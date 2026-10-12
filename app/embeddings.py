"""
Embeddings via the Voyage AI API (Anthropic's recommended embeddings partner;
Anthropic itself has no embeddings endpoint).

Text goes out over HTTP, 1024 floats come back per input. Postgres never makes
embeddings -- it only stores and compares the numbers this module produces.
"""

import logging
import time

from voyageai.client_async import AsyncClient

from app.config import settings
from app.logging import log

logger = logging.getLogger(__name__)

_client: AsyncClient | None = None


def init_client() -> None:
    """Create one client at startup and reuse it (it owns a connection pool)."""
    global _client
    _client = AsyncClient(
        api_key=settings.voyage_api_key,
        # The Voyage SDK defaults to max_retries=0 -- unlike the Anthropic SDK,
        # which retries twice. Left at 0, a single 429 fails the request.
        max_retries=settings.embed_max_retries,
        timeout=settings.embed_timeout_s,
    )
    log(logger, "embedding client ready", model=settings.embed_model,
        dim=settings.embed_dim)


def _get() -> AsyncClient:
    if _client is None:
        raise RuntimeError("embedding client not initialised; call init_client() at startup")
    return _client


async def _embed(texts: list[str], input_type: str) -> list[list[float]]:
    client = _get()
    vectors: list[list[float]] = []
    # One API call per batch, not per text: 100 chunks = 1 request, not 100.
    for i in range(0, len(texts), settings.embed_batch_size):
        batch = texts[i : i + settings.embed_batch_size]
        start = time.perf_counter()
        result = await client.embed(
            batch,
            model=settings.embed_model,
            # "document" for what we store, "query" for what users ask. Voyage
            # prepends a different instruction for each; mixing them up still
            # returns vectors, just worse matches -- a silent quality bug.
            input_type=input_type,
            # Must equal the vector(N) column size, or every INSERT fails.
            output_dimension=settings.embed_dim,
            # Default is True: over-long input is silently cut off. We would
            # rather get an error than embed half a chunk without knowing.
            truncation=False,
        )
        # The SDK types this as floats OR ints (ints only for quantised
        # output_dtype, which we don't request). float() makes it explicit.
        vectors.extend([float(x) for x in v] for v in result.embeddings)
        log(logger, "embed call", input_type=input_type, n=len(batch),
            tokens=result.total_tokens, ms=round((time.perf_counter() - start) * 1000))
    return vectors


async def embed_passages(texts: list[str]) -> list[list[float]]:
    return await _embed(texts, "document")


async def embed_query(text: str) -> list[float]:
    return (await _embed([text], "query"))[0]
