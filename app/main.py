"""
FastAPI app: routes, request-id middleware, error handling, startup.

    GET  /health          liveness  -- is the process up?
    GET  /health/ready    readiness -- can it reach Postgres and Redis?
    POST /ingest          add or replace a document
    POST /ask             answer as JSON (cached)
    POST /ask/stream      answer as Server-Sent Events, token by token
"""

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated

import anthropic
import voyageai.error
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app import cache, embeddings, llm
from app.config import settings
from app.db import engine, get_session, init_db
from app.graph import rag_graph
from app.ingest import ingest_document
from app.logging import log, request_id_var, setup_logging
from app.schemas import AskRequest, AskResponse, IngestRequest, IngestResponse, Source

setup_logging(settings.log_level)
logger = logging.getLogger("app")

SessionDep = Annotated[AsyncSession, Depends(get_session)]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: everything slow happens once here, never per request.
    await init_db()
    embeddings.init_client()
    if not settings.voyage_api_key:
        log(logger, "VOYAGE_API_KEY not set: ingest and retrieval will fail",
            logging.WARNING)
    if not settings.anthropic_api_key:
        # Without this line, a missing key surfaces later as an opaque
        # TypeError ("Could not resolve authentication method") inside the
        # generate node, on the first real question.
        log(logger, "ANTHROPIC_API_KEY not set: retrieval works, answers will fail",
            logging.WARNING)
    log(logger, "startup complete", model=settings.llm_model,
        pipeline=settings.answer_pipeline, extract_model=settings.extract_model)
    yield
    # Shutdown: close pools cleanly.
    await engine.dispose()
    await cache.client.aclose()


app = FastAPI(title="RAG POC", lifespan=lifespan)


@app.middleware("http")
async def request_context(request: Request, call_next):
    # Reuse the caller's id if given, so one id can follow a request across
    # services; otherwise mint one.
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    token = request_id_var.set(rid)
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("unhandled error")
        response = JSONResponse({"detail": "internal error", "request_id": rid},
                                status_code=500)
    finally:
        request_id_var.reset(token)
    response.headers["x-request-id"] = rid
    headers_ms = round((time.perf_counter() - start) * 1000)

    def log_request() -> None:
        # The reset above already happened, so pass the id explicitly here.
        logger.info("request", extra={"fields": {
            "request_id": rid, "method": request.method, "path": request.url.path,
            "status": response.status_code,
            # headers_ms: when the status line went out. ms: when the LAST byte
            # did. For JSON they're ~equal; for /ask/stream, headers go out in
            # milliseconds and the body streams for seconds, so logging at
            # call_next() time reported a 3.8 s stream as "ms: 2".
            "headers_ms": headers_ms,
            "ms": round((time.perf_counter() - start) * 1000)}})

    body = getattr(response, "body_iterator", None)
    if body is None:  # our own JSONResponse from the except branch: no stream
        log_request()
        return response

    async def logged_body():
        # Runs while the server sends the body; the finally fires after the
        # last chunk -- or when the client disconnects mid-stream.
        try:
            async for chunk in body:
                yield chunk
        finally:
            log_request()

    response.body_iterator = logged_body()  # type: ignore[attr-defined]
    return response


# --- upstream LLM errors -> honest HTTP statuses ----------------------------
# The SDK has already retried these by the time they reach us.

@app.exception_handler(anthropic.RateLimitError)
async def llm_rate_limited(request: Request, exc: anthropic.RateLimitError):
    return JSONResponse({"detail": "LLM rate limited, retry shortly",
                         "request_id": request_id_var.get()}, status_code=503,
                        headers={"retry-after": "10"})


@app.exception_handler(anthropic.APITimeoutError)
async def llm_timeout(request: Request, exc: anthropic.APITimeoutError):
    return JSONResponse({"detail": "LLM timed out",
                         "request_id": request_id_var.get()}, status_code=504)


@app.exception_handler(anthropic.APIConnectionError)
async def llm_unreachable(request: Request, exc: anthropic.APIConnectionError):
    return JSONResponse({"detail": "LLM unreachable",
                         "request_id": request_id_var.get()}, status_code=502)


@app.exception_handler(voyageai.error.RateLimitError)
async def embed_rate_limited(request: Request, exc: voyageai.error.RateLimitError):
    return JSONResponse({"detail": "embedding API rate limited, retry shortly",
                         "request_id": request_id_var.get()}, status_code=503,
                        headers={"retry-after": "10"})


@app.exception_handler(voyageai.error.AuthenticationError)
async def embed_auth_failed(request: Request, exc: voyageai.error.AuthenticationError):
    # Our misconfiguration, not the caller's fault -> still a 5xx, but say WHAT,
    # so on-call reads the cause in the response instead of digging for it.
    logger.error("embedding API auth failed: check VOYAGE_API_KEY")
    return JSONResponse({"detail": "embedding API auth failed (server misconfigured)",
                         "request_id": request_id_var.get()}, status_code=500)


@app.exception_handler(voyageai.error.APIConnectionError)
async def embed_unreachable(request: Request, exc: voyageai.error.APIConnectionError):
    return JSONResponse({"detail": "embedding API unreachable",
                         "request_id": request_id_var.get()}, status_code=502)


# --- routes -----------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/health/ready")
async def ready() -> JSONResponse:
    checks = {}
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        checks["postgres"] = "ok"
    except Exception as exc:
        checks["postgres"] = type(exc).__name__
    try:
        await cache.client.ping()
        checks["redis"] = "ok"
    except Exception as exc:
        checks["redis"] = type(exc).__name__
    checks["anthropic_key"] = "ok" if settings.anthropic_api_key else "missing"
    checks["voyage_key"] = "ok" if settings.voyage_api_key else "missing"
    healthy = all(v == "ok" for v in checks.values())
    return JSONResponse(checks, status_code=200 if healthy else 503)


@app.post("/ingest", response_model=IngestResponse)
async def ingest(body: IngestRequest, session: SessionDep):
    return await ingest_document(session, source=body.source, title=body.title,
                                 text=body.text)


def _sources(chunks) -> list[Source]:
    # Order is a contract: build_context numbers chunks [1], [2]... in this
    # same order and the client maps [n] to sources[n-1]. Never re-sort here.
    return [Source(source=c.source, title=c.title, similarity=c.similarity)
            for c in chunks]


@app.post("/ask", response_model=AskResponse)
async def ask(body: AskRequest):
    # One key for both read and write: it pins the corpus version this answer
    # is computed against (see cache.answer_key).
    key = await cache.answer_key(body.question, body.top_k)
    hit = await cache.get_answer(key) if key else None
    if hit:
        log(logger, "cache hit")
        return AskResponse(**hit, cached=True)

    state = await rag_graph.ainvoke({"question": body.question, "top_k": body.top_k})
    result = {"answer": state["answer"],
              "sources": [s.model_dump() for s in _sources(state["chunks"])]}

    # Don't cache "not found": the document may be ingested a minute later,
    # or (verified pipeline) the extraction failed validation this time only.
    if key and state["chunks"] and state["answer"] != llm.NOT_FOUND:
        await cache.set_answer(key, result)
    return AskResponse(**result, cached=False)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/ask/stream")
async def ask_stream(body: AskRequest):
    rid = request_id_var.get()

    async def events():
        # This generator runs AFTER the handler has returned, while the
        # response is being sent. An exception here cannot become a 500 --
        # the 200 status line is already on the wire. So errors are caught
        # and sent as an `error` event the client can show.
        token = request_id_var.set(rid)
        try:
            async for mode, chunk in rag_graph.astream(
                {"question": body.question, "top_k": body.top_k},
                stream_mode=["custom", "updates"],
            ):
                # Chunks are typed loosely (Any) by LangGraph; check the shape
                # before indexing instead of trusting it.
                if mode == "custom" and isinstance(chunk, dict):
                    yield _sse("token", chunk)
                elif mode == "updates" and isinstance(chunk, dict) and "retrieve" in chunk:
                    srcs = _sources(chunk["retrieve"]["chunks"])
                    yield _sse("sources", {"sources": [s.model_dump() for s in srcs]})
            yield _sse("done", {"request_id": rid})
        except Exception as exc:
            logger.exception("stream failed")
            yield _sse("error", {"detail": type(exc).__name__, "request_id": rid})
        finally:
            request_id_var.reset(token)

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"x-request-id": rid})
