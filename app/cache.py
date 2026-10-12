"""
Redis answer cache.

The key includes a corpus version. Every ingest bumps it, so answers computed
against old documents stop matching instead of being served stale for an hour.
"""

import hashlib
import json
import logging

import redis.asyncio as redis

from app.config import settings
from app.logging import log

logger = logging.getLogger(__name__)

client = redis.from_url(settings.redis_url, decode_responses=True)

CORPUS_VERSION_KEY = "rag:corpus_version"


def _normalise(question: str) -> str:
    # "What is the refund policy?" and "what is the refund policy" should hit
    # the same entry.
    return " ".join(question.lower().split()).rstrip("?")


async def _key(question: str, top_k: int) -> str:
    version = await client.get(CORPUS_VERSION_KEY) or "0"
    raw = f"{settings.llm_model}|{top_k}|v{version}|{_normalise(question)}"
    return "rag:answer:" + hashlib.sha256(raw.encode()).hexdigest()


async def get_answer(question: str, top_k: int) -> dict | None:
    try:
        cached = await client.get(await _key(question, top_k))
    except redis.RedisError as exc:
        # Cache is an optimisation. If Redis is down, answer without it
        # rather than failing the request.
        log(logger, "cache read failed", logging.WARNING, error=type(exc).__name__)
        return None
    return json.loads(cached) if cached else None


async def set_answer(question: str, top_k: int, value: dict) -> None:
    try:
        await client.set(await _key(question, top_k), json.dumps(value),
                         ex=settings.cache_ttl_s)
    except redis.RedisError as exc:
        log(logger, "cache write failed", logging.WARNING, error=type(exc).__name__)


async def bump_corpus_version() -> None:
    await client.incr(CORPUS_VERSION_KEY)
