"""
All settings, read once from the environment / .env and validated at startup.

Every knob that changes RAG behaviour lives here, not scattered in code. In an
incident, "what is chunk_size set to in prod?" should be one lookup.
"""

# >> `Literal` is a type hint meaning "only these exact values are allowed" (like a TS union of
# >> string literals: 'low' | 'medium'). Pydantic enforces it at runtime.
from typing import Literal

# >> pydantic-settings: BaseSettings is a Pydantic model that fills its fields from env vars / .env.
# >> SettingsConfigDict is the typed dict used to configure that behaviour.
from pydantic_settings import BaseSettings, SettingsConfigDict


# >> Define a class `Settings` that inherits from BaseSettings (the part in parentheses is the parent
# >> class, like `class Settings extends BaseSettings` in JS). Each field below = one env var.
class Settings(BaseSettings):
    # >> `model_config` is a special Pydantic class attribute: read values from the `.env` file too,
    # >> and silently ignore env vars that don't match any field (instead of raising an error).
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- infra -------------------------------------------------------------
    # >> `name: type = default` declares a field with a type hint and a default value.
    # >> Env var DATABASE_URL (case-insensitive) overrides it. This is the Postgres connection string.
    database_url: str = "postgresql+asyncpg://rag:rag@localhost:5433/rag"
    # >> Redis connection string (used as the answer cache). Overridable via REDIS_URL.
    redis_url: str = "redis://localhost:6380/0"
    # >> DB connection pool: keep 5 open connections, allow 5 extra temporary ones under load,
    # >> and wait at most 10 seconds for a free connection before failing (`float` = decimal number).
    db_pool_size: int = 5
    db_max_overflow: int = 5
    db_pool_timeout_s: float = 10.0

    # --- LLM ---------------------------------------------------------------
    # pydantic-settings reads .env into THIS object, not into os.environ, so
    # the key is passed to the client explicitly rather than relying on the
    # SDK finding ANTHROPIC_API_KEY in the environment.
    # >> `str | None` means "a string or None" (None is Python's null). Default None = key not set.
    anthropic_api_key: str | None = None
    # >> Which Claude model to call for answers.
    llm_model: str = "claude-opus-5-5"
    # Grounded Q&A doesn't need deep reasoning; low effort = faster + cheaper.
    # Literal: LLM_EFFORT=hgh in .env fails at startup, not on the first request.
    # >> Reasoning-effort setting; only one of the listed strings is accepted, default "low".
    llm_effort: Literal["low", "medium", "high", "xhigh", "max"] = "low"
    # >> Max tokens the LLM may generate, request timeout in seconds, and how many times the SDK
    # >> retries a failed call.
    llm_max_tokens: int = 2048
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 2

    # --- embeddings (Voyage AI) ---------------------------------------------
    # >> API key for Voyage AI (the embedding service); None if not configured.
    voyage_api_key: str | None = None
    # >> Name of the Voyage model that turns text into vectors (lists of numbers).
    embed_model: str = "voyage-3.5-lite"
    # The dimension MUST match the vector(N) column. Change it without
    # recreating the table and re-ingesting, and every insert/query fails.
    # voyage-3.5-lite supports 256 / 512 / 1024 / 2048.
    # >> Length of each embedding vector (how many numbers per chunk).
    embed_dim: int = 1024
    # >> How many texts to embed per API call, the per-call timeout, and the retry count.
    embed_batch_size: int = 128
    embed_timeout_s: float = 30.0
    embed_max_retries: int = 3

    # --- chunking ----------------------------------------------------------
    # >> Documents are split into pieces of ~800 characters (the `#` part after it is a normal comment).
    chunk_size: int = 800      # characters, not tokens
    # >> Neighbouring chunks share 120 characters so a sentence cut at a border isn't lost.
    chunk_overlap: int = 120

    # --- retrieval ---------------------------------------------------------
    # >> Default number of most-similar chunks to fetch for each question.
    top_k: int = 4
    # Cosine similarity floor. Below this a chunk is "not really relevant",
    # and answering from it is how a RAG system hallucinates confidently.
    # Calibrate, don't guess: each embedding model has its own score range.
    # (With the earlier local bge-small model an off-topic question scored
    # 0.41, so a 0.35 floor let it through.) Re-measure with
    # `scripts/smoke.py --retrieval` whenever the embed model changes.
    # >> Chunks scoring below 0.55 similarity are thrown away as irrelevant.
    min_similarity: float = 0.55
    # Budget guard: hard cap on context characters sent to the LLM.
    # >> `12_000` is just 12000; Python allows `_` in numbers for readability (like JS `12_000`).
    max_context_chars: int = 12_000

    # --- cache -------------------------------------------------------------
    # >> How long a cached answer lives in Redis: 3600 seconds = 1 hour (TTL = time to live).
    cache_ttl_s: int = 3600

    # >> Minimum log level to print ("DEBUG", "INFO", "WARNING", ...). Overridable via LOG_LEVEL.
    log_level: str = "INFO"


# >> Create ONE Settings object now, at import time: this is when env/.env is read and validated.
# >> Other modules do `from app.config import settings` to share it (like a singleton module export).
settings = Settings()
