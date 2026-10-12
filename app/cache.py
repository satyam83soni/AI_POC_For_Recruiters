"""
Redis answer cache.

The key includes a corpus version. Every ingest bumps it, so answers computed
against old documents stop matching instead of being served stale for an hour.
"""

# >> hashlib: built-in hashing (like Node's `crypto.createHash`).
import hashlib
# >> json: built-in JSON tools; json.dumps = JSON.stringify, json.loads = JSON.parse.
import json
# >> logging: built-in logging module.
import logging

# >> The async Redis client library, imported under the short name `redis` (like ioredis).
import redis.asyncio as redis

# >> App settings (redis_url, cache TTL, model name).
from app.config import settings
# >> Project helper for structured log lines.
from app.logging import log

# >> Logger named after this module ("app.cache").
logger = logging.getLogger(__name__)

# >> Create one Redis client from a URL; decode_responses=True returns str instead of raw bytes.
client = redis.from_url(settings.redis_url, decode_responses=True)

# >> The Redis key that stores the current corpus version number (UPPER_CASE = constant by convention).
CORPUS_VERSION_KEY = "rag:corpus_version"


# >> Normal (non-async) helper; leading `_` means "private to this module" by convention.
def _normalise(question: str) -> str:
    # "What is the refund policy?" and "what is the refund policy" should hit
    # the same entry.
    # >> lowercase -> .split() splits on any whitespace -> " ".join glues with single spaces
    # >> -> .rstrip("?") removes trailing question marks.
    return " ".join(question.lower().split()).rstrip("?")


# >> Async helper that builds the Redis key for a question.
async def _key(question: str, top_k: int) -> str:
    # >> Read the current corpus version from Redis; `or "0"` uses "0" if the key doesn't exist (None).
    version = await client.get(CORPUS_VERSION_KEY) or "0"
    # >> f-string (like a JS template literal `${...}`): combine model, top_k, version and question.
    raw = f"{settings.llm_model}|{top_k}|v{version}|{_normalise(question)}"
    # >> Hash it to a fixed-length key, e.g. "rag:answer:3fa9...".
    return "rag:answer:" + hashlib.sha256(raw.encode()).hexdigest()


# >> Look up a cached answer; returns a dict or None (`dict | None` = "either type").
async def get_answer(question: str, top_k: int) -> dict | None:
    # >> try/except = try/catch in JS.
    try:
        # >> Build the key, then GET it from Redis (None if missing or expired).
        cached = await client.get(await _key(question, top_k))
    # >> Catch any Redis error (connection refused, timeout, ...) and keep it in `exc`.
    except redis.RedisError as exc:
        # Cache is an optimisation. If Redis is down, answer without it
        # rather than failing the request.
        # >> Log a warning with the error's class name (type(exc).__name__, e.g. "ConnectionError").
        log(logger, "cache read failed", logging.WARNING, error=type(exc).__name__)
        # >> Act as a cache miss.
        return None
    # >> If found, parse the JSON string back into a dict; otherwise return None.
    return json.loads(cached) if cached else None


# >> Save an answer dict in the cache; returns nothing.
async def set_answer(question: str, top_k: int, value: dict) -> None:
    # >> Wrap in try so a Redis failure never breaks the request.
    try:
        # >> SET key = JSON string of the answer...
        await client.set(await _key(question, top_k), json.dumps(value),
                         # >> ...with ex = expiry in seconds (TTL); Redis deletes it after that.
                         ex=settings.cache_ttl_s)
    # >> On a Redis error...
    except redis.RedisError as exc:
        # >> ...just log a warning and carry on.
        log(logger, "cache write failed", logging.WARNING, error=type(exc).__name__)


# >> Called after each ingest to invalidate all cached answers.
async def bump_corpus_version() -> None:
    # >> INCR: atomically add 1 to the number stored at the key (starts at 0 if missing).
    # >> New version -> new cache keys, so old answers are never matched again.
    await client.incr(CORPUS_VERSION_KEY)
