"""
Embeddings via the Voyage AI API (Anthropic's recommended embeddings partner;
Anthropic itself has no embeddings endpoint).

Text goes out over HTTP, 1024 floats come back per input. Postgres never makes
embeddings -- it only stores and compares the numbers this module produces.
"""

# >> Python's built-in logging library (logger objects, log levels).
import logging
# >> Built-in time module; time.perf_counter() is used as a stopwatch.
import time

# >> Voyage AI's async SDK client: sends text to Voyage's API and gets embedding vectors back.
from voyageai.client_async import AsyncClient

# >> Our app settings (API key, model name, vector size, batch size...).
from app.config import settings
# >> Our structured-log helper: log(logger, message, **fields).
from app.logging import log

# >> A logger named after this module ("app.embeddings").
logger = logging.getLogger(__name__)

# >> Module-level variable holding the shared client; None until init_client() runs.
# >> The leading _ means "private to this module" by convention (not enforced).
_client: AsyncClient | None = None


# >> Called once at app startup to create the Voyage client. `-> None` = returns nothing.
def init_client() -> None:
    """Create one client at startup and reuse it (it owns a connection pool)."""
    # >> `global` lets this function assign to the module-level _client instead of making a
    # >> new local variable (in JS you'd just assign to the outer `let`).
    global _client
    # >> Create the async Voyage client and store it in the module variable.
    _client = AsyncClient(
        # >> Secret key for the Voyage API.
        api_key=settings.voyage_api_key,
        # The Voyage SDK defaults to max_retries=0 -- unlike the Anthropic SDK,
        # which retries twice. Left at 0, a single 429 fails the request.
        # >> How many times to retry a failed call automatically.
        max_retries=settings.embed_max_retries,
        # >> Give up on a call after this many seconds.
        timeout=settings.embed_timeout_s,
    # >> End of the constructor call.
    )
    # >> Log that the client is ready, with the model name and vector size as fields.
    log(logger, "embedding client ready", model=settings.embed_model,
        # >> (dim = number of floats per vector)
        dim=settings.embed_dim)


# >> Small getter that returns the client or fails loudly if startup forgot to create it.
def _get() -> AsyncClient:
    # >> `is None` = identity check against None (like `=== null` in JS).
    if _client is None:
        # >> `raise` throws an exception (like `throw new Error(...)` in JS).
        raise RuntimeError("embedding client not initialised; call init_client() at startup")
    # >> Return the shared client.
    return _client


# >> Core function: embeds a list of texts, returns one vector (list of floats) per text.
async def _embed(texts: list[str], input_type: str) -> list[list[float]]:
    # >> Get the shared Voyage client.
    client = _get()
    # >> Will hold every vector, in the same order as `texts`.
    vectors: list[list[float]] = []
    # One API call per batch, not per text: 100 chunks = 1 request, not 100.
    # >> range(start, stop, step): i = 0, batch_size, 2*batch_size, ... while i < len(texts)
    # >> (like for (let i = 0; i < texts.length; i += size)).
    for i in range(0, len(texts), settings.embed_batch_size):
        # >> Slice: texts[i : i + n] = items from index i up to (not incl.) i+n, like .slice(i, i+n).
        batch = texts[i : i + settings.embed_batch_size]
        # >> Start the stopwatch for this API call.
        start = time.perf_counter()
        # >> Call Voyage's embed endpoint for this batch and wait for the result.
        result = await client.embed(
            # >> The list of texts to turn into vectors.
            batch,
            # >> Which Voyage embedding model to use.
            model=settings.embed_model,
            # "document" for what we store, "query" for what users ask. Voyage
            # prepends a different instruction for each; mixing them up still
            # returns vectors, just worse matches -- a silent quality bug.
            # >> input_type: "document" (text being stored) or "query" (a user's question).
            input_type=input_type,
            # Must equal the vector(N) column size, or every INSERT fails.
            # >> output_dimension: how many numbers each vector should contain.
            output_dimension=settings.embed_dim,
            # Default is True: over-long input is silently cut off. We would
            # rather get an error than embed half a chunk without knowing.
            # >> truncation=False: raise an error instead of silently cutting long text.
            truncation=False,
        # >> End of the embed call.
        )
        # The SDK types this as floats OR ints (ints only for quantised
        # output_dtype, which we don't request). float() makes it explicit.
        # >> For each vector v in the result, build [float(x) for x in v] (a list comprehension,
        # >> like v.map(x => Number(x))); .extend adds them all to `vectors` (like push(...arr)).
        vectors.extend([float(x) for x in v] for v in result.embeddings)
        # >> Log the call: type, how many texts, tokens billed...
        log(logger, "embed call", input_type=input_type, n=len(batch),
            # >> ...and how long it took in milliseconds.
            tokens=result.total_tokens, ms=round((time.perf_counter() - start) * 1000))
    # >> All vectors, one per input text.
    return vectors


# >> Public helper for ingest: embed document chunks (input_type "document").
async def embed_passages(texts: list[str]) -> list[list[float]]:
    # >> Delegate to _embed and return its list of vectors.
    return await _embed(texts, "document")


# >> Public helper for search: embed one user question (input_type "query").
async def embed_query(text: str) -> list[float]:
    # >> Wrap the single text in a list, embed it, then take the first (only) vector with [0].
    return (await _embed([text], "query"))[0]
