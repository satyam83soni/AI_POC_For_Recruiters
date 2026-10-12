"""
FastAPI app: routes, request-id middleware, error handling, startup.

    GET  /health          liveness  -- is the process up?
    GET  /health/ready    readiness -- can it reach Postgres and Redis?
    POST /ingest          add or replace a document
    POST /ask             answer as JSON (cached)
    POST /ask/stream      answer as Server-Sent Events, token by token
"""

# >> Standard library: `json` (JSON.stringify/parse), `logging` (built-in logger),
# >> `time` (timers for request duration), `uuid` (random unique ids for request ids).
import json
import logging
import time
import uuid
# >> asynccontextmanager: a decorator that turns an async generator function into an
# >> "async with"-style setup/teardown block (used for app startup/shutdown).
from contextlib import asynccontextmanager
# >> Annotated[T, extra]: a type hint that attaches extra info to a type; FastAPI reads that
# >> extra info (here, a Depends) to know how to fill a parameter.
from typing import Annotated

# >> Anthropic's SDK (Claude); imported here only to catch its error classes.
import anthropic
# >> Voyage AI's error classes (the embedding service), also only for error handling.
import voyageai.error
# >> FastAPI: Depends = dependency injection marker, FastAPI = the app class
# >> (like express()), Request = the incoming HTTP request object (like `req`).
from fastapi import Depends, FastAPI, Request
# >> JSONResponse sends JSON with a chosen status code; StreamingResponse sends a body piece by piece.
from fastapi.responses import JSONResponse, StreamingResponse
# >> SQLAlchemy (Python's main DB toolkit/ORM): `text` wraps a raw SQL string.
from sqlalchemy import text
# >> AsyncSession: an async DB session (a unit of work / connection handle), used as a type hint.
from sqlalchemy.ext.asyncio import AsyncSession

# >> Our own modules: `cache` (Redis answer cache) and `embeddings` (Voyage client).
from app import cache, embeddings
# >> The validated settings object from app/config.py.
from app.config import settings
# >> DB pieces: the shared connection engine/pool, a per-request session provider, table setup.
from app.db import engine, get_session, init_db
# >> The compiled LangGraph pipeline (retrieve chunks -> generate answer).
from app.graph import rag_graph
# >> Function that chunks, embeds and stores a document.
from app.ingest import ingest_document
# >> Our logging helpers: log() with extra fields, the request-id ContextVar, and setup.
from app.logging import log, request_id_var, setup_logging
# >> Pydantic models describing request/response JSON shapes.
from app.schemas import AskRequest, AskResponse, IngestRequest, IngestResponse, Source

# >> Configure JSON logging right at import time, using the level from settings (e.g. "INFO").
setup_logging(settings.log_level)
# >> Get (or create) a logger named "app" for this module's log lines.
logger = logging.getLogger("app")

# >> A reusable type alias: "an AsyncSession that FastAPI gets by calling get_session()".
# >> Writing `session: SessionDep` in a route makes FastAPI inject a fresh DB session per request.
SessionDep = Annotated[AsyncSession, Depends(get_session)]


# >> Decorator (`@...` wraps the function below): makes lifespan usable as an async context manager.
# >> Code before `yield` runs at startup, code after `yield` runs at shutdown.
@asynccontextmanager
# >> `async def` = an async function (returns an awaitable, like a JS async function).
async def lifespan(app: FastAPI):
    # Startup: everything slow happens once here, never per request.
    # >> `await` waits for the async call to finish: create DB tables/extensions if missing.
    await init_db()
    # >> Create the Voyage embeddings client once (a normal, non-async call).
    embeddings.init_client()
    # >> `not x` is true when x is None or an empty string, i.e. the Voyage key is missing.
    if not settings.voyage_api_key:
        # >> Log a warning (logging.WARNING passed as the level argument) instead of crashing.
        log(logger, "VOYAGE_API_KEY not set: ingest and retrieval will fail",
            logging.WARNING)
    # >> Same check for the Anthropic (Claude) API key.
    if not settings.anthropic_api_key:
        # Without this line, a missing key surfaces later as an opaque
        # TypeError ("Could not resolve authentication method") inside the
        # generate node, on the first real question.
        # >> Warn that answering will fail without the key.
        log(logger, "ANTHROPIC_API_KEY not set: retrieval works, answers will fail",
            logging.WARNING)
    # >> Log startup done; `model=` becomes an extra JSON field via log()'s **fields.
    log(logger, "startup complete", model=settings.llm_model)
    # >> `yield` pauses here while the app serves requests; it resumes when the server shuts down.
    yield
    # Shutdown: close pools cleanly.
    # >> Close all pooled Postgres connections.
    await engine.dispose()
    # >> Close the Redis client connection.
    await cache.client.aclose()


# >> Create the FastAPI app (like `const app = express()`), wiring in the startup/shutdown hook.
app = FastAPI(title="RAG POC", lifespan=lifespan)


# >> Register an HTTP middleware: runs around EVERY request (like `app.use((req, res, next) => ...)`).
@app.middleware("http")
# >> `call_next(request)` passes the request on to the route and returns its response (like next()).
async def request_context(request: Request, call_next):
    # Reuse the caller's id if given, so one id can follow a request across
    # services; otherwise mint one.
    # >> `headers.get(...)` returns None if missing; `a or b` picks b when a is empty/None.
    # >> uuid4().hex is a random 32-char hex string; `[:12]` slices off the first 12 characters.
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    # >> Store the id in the ContextVar for this request; `token` lets us undo it later.
    token = request_id_var.set(rid)
    # >> High-precision timer start (seconds as a float), to measure request duration.
    start = time.perf_counter()
    # >> try/except/finally works like JS try/catch/finally.
    try:
        # >> Run the actual route handler and wait for its response.
        response = await call_next(request)
    # >> Catch any error the route didn't handle itself.
    except Exception:
        # >> Log at ERROR level including the full stack trace.
        logger.exception("unhandled error")
        # >> Build a generic 500 JSON response (like `res.status(500).json({...})`).
        response = JSONResponse({"detail": "internal error", "request_id": rid},
                                status_code=500)
    # >> Always runs, error or not.
    finally:
        # >> Restore the ContextVar to its previous value so the id doesn't leak.
        request_id_var.reset(token)
    # >> Echo the request id back to the client as a response header.
    response.headers["x-request-id"] = rid
    # The reset above already happened, so pass the id explicitly here.
    # >> Write one access-log line; `extra={"fields": {...}}` adds these keys to the JSON line.
    logger.info("request", extra={"fields": {
        # >> The request id, HTTP method (GET/POST) and URL path.
        "request_id": rid, "method": request.method, "path": request.url.path,
        # >> The HTTP status code that was sent.
        "status": response.status_code,
        # >> Elapsed time: seconds * 1000 = milliseconds, rounded to a whole number.
        "ms": round((time.perf_counter() - start) * 1000)}})
    # >> Hand the response back to FastAPI to send.
    return response


# --- upstream LLM errors -> honest HTTP statuses ----------------------------
# The SDK has already retried these by the time they reach us.

# >> Register a global error handler: whenever a route raises anthropic.RateLimitError,
# >> FastAPI calls this function instead of returning a plain 500 (like Express error middleware).
@app.exception_handler(anthropic.RateLimitError)
# >> `exc` is the caught exception object.
async def llm_rate_limited(request: Request, exc: anthropic.RateLimitError):
    # >> Respond 503 (service unavailable) with the request id, plus a `retry-after: 10` header
    # >> telling the client to retry in 10 seconds.
    return JSONResponse({"detail": "LLM rate limited, retry shortly",
                         "request_id": request_id_var.get()}, status_code=503,
                        headers={"retry-after": "10"})


# >> Handler for when the Claude API call times out.
@app.exception_handler(anthropic.APITimeoutError)
async def llm_timeout(request: Request, exc: anthropic.APITimeoutError):
    # >> Respond 504 (gateway timeout): an upstream service was too slow.
    return JSONResponse({"detail": "LLM timed out",
                         "request_id": request_id_var.get()}, status_code=504)


# >> Handler for when the Claude API can't be reached at all (network/DNS problems).
@app.exception_handler(anthropic.APIConnectionError)
async def llm_unreachable(request: Request, exc: anthropic.APIConnectionError):
    # >> Respond 502 (bad gateway): the upstream service failed.
    return JSONResponse({"detail": "LLM unreachable",
                         "request_id": request_id_var.get()}, status_code=502)


# >> Handler for Voyage (embedding API) rate limiting.
@app.exception_handler(voyageai.error.RateLimitError)
async def embed_rate_limited(request: Request, exc: voyageai.error.RateLimitError):
    # >> Same as the LLM case: 503 plus a retry-after header of 10 seconds.
    return JSONResponse({"detail": "embedding API rate limited, retry shortly",
                         "request_id": request_id_var.get()}, status_code=503,
                        headers={"retry-after": "10"})


# >> Handler for a bad/missing Voyage API key.
@app.exception_handler(voyageai.error.AuthenticationError)
async def embed_auth_failed(request: Request, exc: voyageai.error.AuthenticationError):
    # Our misconfiguration, not the caller's fault -> still a 5xx, but say WHAT,
    # so on-call reads the cause in the response instead of digging for it.
    # >> Log an ERROR line pointing at the config problem.
    logger.error("embedding API auth failed: check VOYAGE_API_KEY")
    # >> Respond 500 with a message that says the server is misconfigured.
    return JSONResponse({"detail": "embedding API auth failed (server misconfigured)",
                         "request_id": request_id_var.get()}, status_code=500)


# >> Handler for when the Voyage API can't be reached.
@app.exception_handler(voyageai.error.APIConnectionError)
async def embed_unreachable(request: Request, exc: voyageai.error.APIConnectionError):
    # >> Respond 502 (bad gateway).
    return JSONResponse({"detail": "embedding API unreachable",
                         "request_id": request_id_var.get()}, status_code=502)


# --- routes -----------------------------------------------------------------

# >> Register a GET /health route (like `app.get('/health', handler)` in Express).
@app.get("/health")
# >> `-> dict` is the return type hint; FastAPI turns a returned dict into JSON automatically.
async def health() -> dict:
    # >> Always reply {"status": "ok"}: proves the process is alive.
    return {"status": "ok"}


# >> GET /health/ready: checks the app's dependencies (DB, Redis, API keys).
@app.get("/health/ready")
async def ready() -> JSONResponse:
    # >> Empty dict to collect one result per dependency.
    checks = {}
    # >> Try the Postgres check; any error is caught below.
    try:
        # >> `async with` = open a DB connection and auto-close it when the block ends
        # >> (even on error), like a try/finally that releases the connection.
        async with engine.connect() as conn:
            # >> Run a trivial query to prove the database answers.
            await conn.execute(text("SELECT 1"))
        # >> Reached only if the query worked.
        checks["postgres"] = "ok"
    # >> `as exc` binds the caught error to a variable.
    except Exception as exc:
        # >> `type(exc).__name__` = the error's class name as text, e.g. "ConnectionRefusedError".
        checks["postgres"] = type(exc).__name__
    # >> Same pattern for Redis.
    try:
        # >> PING Redis; raises if Redis is down.
        await cache.client.ping()
        # >> Redis answered.
        checks["redis"] = "ok"
    # >> Catch any Redis error.
    except Exception as exc:
        # >> Record the error class name instead of "ok".
        checks["redis"] = type(exc).__name__
    # >> Python's inline if/else: `a if cond else b` is like JS `cond ? a : b`.
    checks["anthropic_key"] = "ok" if settings.anthropic_api_key else "missing"
    # >> Same for the Voyage key.
    checks["voyage_key"] = "ok" if settings.voyage_api_key else "missing"
    # >> `all(...)` is true only if every item is true; `v == "ok" for v in checks.values()` is a
    # >> generator expression over the dict's values (like Object.values(checks).every(v => v === "ok")).
    healthy = all(v == "ok" for v in checks.values())
    # >> Return all checks as JSON: status 200 if healthy, 503 if anything failed.
    return JSONResponse(checks, status_code=200 if healthy else 503)


# >> POST /ingest; `response_model` tells FastAPI to validate/serialize the return value as
# >> IngestResponse (and document it in /docs).
@app.post("/ingest", response_model=IngestResponse)
# >> `body: IngestRequest` = FastAPI parses + validates the JSON body into that model.
# >> `session: SessionDep` = FastAPI injects a DB session via get_session().
async def ingest(body: IngestRequest, session: SessionDep):
    # >> Call the ingest logic with keyword arguments (`name=value`, like passing an options object)
    # >> and return its result as the response.
    return await ingest_document(session, source=body.source, title=body.title,
                                 text=body.text)


# >> Helper (leading `_` = "private by convention"): convert retrieved chunks to Source models.
def _sources(chunks) -> list[Source]:
    # >> List comprehension: `[expr for c in chunks]` builds a new list, like chunks.map(c => ...).
    return [Source(source=c.source, title=c.title, similarity=c.similarity)
            for c in chunks]


# >> POST /ask: answer a question and return it as one JSON response.
@app.post("/ask", response_model=AskResponse)
async def ask(body: AskRequest):
    # >> Look in the Redis cache for an earlier answer to the same question + top_k.
    hit = await cache.get_answer(body.question, body.top_k)
    # >> A non-empty dict is truthy; None (cache miss) is falsy.
    if hit:
        # >> Note the cache hit in the logs.
        log(logger, "cache hit")
        # >> `**hit` unpacks the dict as keyword args (like spreading {...hit, cached: true}).
        return AskResponse(**hit, cached=True)

    # >> Cache miss: run the whole LangGraph pipeline asynchronously with the starting state
    # >> (a dict), and get back the final state dict.
    state = await rag_graph.ainvoke({"question": body.question, "top_k": body.top_k})
    # >> Build the result dict: the answer text, plus sources converted to plain dicts
    # >> (`model_dump()` turns a Pydantic model into a dict, like toJSON()).
    result = {"answer": state["answer"],
              "sources": [s.model_dump() for s in _sources(state["chunks"])]}

    # Don't cache "not found": the document may be ingested a minute later.
    # >> An empty list is falsy, so this only runs if some chunks were found.
    if state["chunks"]:
        # >> Save the result in Redis for next time.
        await cache.set_answer(body.question, body.top_k, result)
    # >> Return the fresh answer, marked as not cached.
    return AskResponse(**result, cached=False)


# >> Helper that formats one Server-Sent Event message as text.
def _sse(event: str, data: dict) -> str:
    # >> f-string: `f"...{x}..."` inserts values, like a JS template literal `...${x}...`.
    # >> SSE format is "event: NAME\ndata: JSON\n\n" (a blank line ends one event).
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# >> POST /ask/stream: same question, but the answer is streamed token by token.
@app.post("/ask/stream")
async def ask_stream(body: AskRequest):
    # >> Read the current request id now, while we're still inside the middleware's context.
    rid = request_id_var.get()

    # >> A nested async generator function: because it contains `yield`, calling it returns an
    # >> object you can loop over with `async for` (like an `async function*` in JS).
    async def events():
        # This generator runs AFTER the handler has returned, while the
        # response is being sent. An exception here cannot become a 500 --
        # the 200 status line is already on the wire. So errors are caught
        # and sent as an `error` event the client can show.
        # >> Re-set the request id inside the generator so its log lines carry it.
        token = request_id_var.set(rid)
        # >> Catch errors so they can be sent to the client as an event.
        try:
            # >> `async for` loops over an async stream (like `for await` in JS). astream() runs the
            # >> graph and yields (mode, chunk) pairs; the pair is unpacked into two variables.
            async for mode, chunk in rag_graph.astream(
                # >> The starting state: the question and how many chunks to retrieve.
                {"question": body.question, "top_k": body.top_k},
                # >> Ask for "custom" events (our tokens) and "updates" (each node's output).
                stream_mode=["custom", "updates"],
            ):
                # Chunks are typed loosely (Any) by LangGraph; check the shape
                # before indexing instead of trusting it.
                # >> `isinstance(x, dict)` checks x's type (like `typeof x === "object"`).
                if mode == "custom" and isinstance(chunk, dict):
                    # >> `yield` sends one value out of the generator: here a "token" SSE event.
                    yield _sse("token", chunk)
                # >> `elif` = else if. `"retrieve" in chunk` checks the dict has that key.
                elif mode == "updates" and isinstance(chunk, dict) and "retrieve" in chunk:
                    # >> The retrieve node finished: convert its chunks to Source models.
                    srcs = _sources(chunk["retrieve"]["chunks"])
                    # >> Send them as a "sources" event (models converted to plain dicts).
                    yield _sse("sources", {"sources": [s.model_dump() for s in srcs]})
            # >> Stream finished normally: send a final "done" event.
            yield _sse("done", {"request_id": rid})
        # >> Something failed mid-stream.
        except Exception as exc:
            # >> Log the error with its stack trace.
            logger.exception("stream failed")
            # >> Tell the client via an "error" event containing the error's class name.
            yield _sse("error", {"detail": type(exc).__name__, "request_id": rid})
        # >> Always runs at the end of the stream.
        finally:
            # >> Restore the ContextVar.
            request_id_var.reset(token)

    # >> Return a streaming response: FastAPI sends each value events() yields as it's produced.
    # >> `text/event-stream` is the SSE content type; the request id goes in a header too.
    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"x-request-id": rid})
