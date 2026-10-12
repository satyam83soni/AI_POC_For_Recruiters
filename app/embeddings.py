"""
Local embeddings with fastembed (ONNX, CPU).

The model call is SYNCHRONOUS and CPU-bound. Called directly inside an
`async def`, it would freeze the event loop -- every other request, health
checks included, would wait. So each call goes through asyncio.to_thread.
"""

import asyncio
import logging
import time

from fastembed import TextEmbedding

from app.config import settings
from app.logging import log

logger = logging.getLogger(__name__)

_model: TextEmbedding | None = None


def load_model() -> None:
    """Load once at startup. First run downloads ~130MB to the cache dir."""
    global _model
    start = time.perf_counter()
    _model = TextEmbedding(model_name=settings.embed_model)
    log(logger, "embedding model loaded", model=settings.embed_model,
        ms=round((time.perf_counter() - start) * 1000))


def _get() -> TextEmbedding:
    if _model is None:
        raise RuntimeError("embedding model not loaded; call load_model() at startup")
    return _model


# bge models are trained asymmetrically: documents and queries get different
# prefixes. passage_embed / query_embed apply the right one. Embedding a query
# as a passage still "works" but retrieval quality quietly drops.

async def embed_passages(texts: list[str]) -> list[list[float]]:
    model = _get()
    vectors = await asyncio.to_thread(lambda: list(model.passage_embed(texts)))
    return [v.tolist() for v in vectors]


async def embed_query(text: str) -> list[float]:
    model = _get()
    vectors = await asyncio.to_thread(lambda: list(model.query_embed(text)))
    return vectors[0].tolist()
