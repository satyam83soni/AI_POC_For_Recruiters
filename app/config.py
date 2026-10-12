"""
All settings, read once from the environment / .env and validated at startup.

Every knob that changes RAG behaviour lives here, not scattered in code. In an
incident, "what is chunk_size set to in prod?" should be one lookup.
"""

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- infra -------------------------------------------------------------
    database_url: str = "postgresql+asyncpg://rag:rag@localhost:5433/rag"
    redis_url: str = "redis://localhost:6380/0"
    db_pool_size: int = 5
    db_max_overflow: int = 5
    db_pool_timeout_s: float = 10.0

    # --- LLM ---------------------------------------------------------------
    # pydantic-settings reads .env into THIS object, not into os.environ, so
    # the key is passed to the client explicitly rather than relying on the
    # SDK finding ANTHROPIC_API_KEY in the environment.
    anthropic_api_key: str | None = None
    llm_model: str = "claude-opus-5-5"
    # Grounded Q&A doesn't need deep reasoning; low effort = faster + cheaper.
    # Literal: LLM_EFFORT=hgh in .env fails at startup, not on the first request.
    llm_effort: Literal["low", "medium", "high", "xhigh", "max"] = "low"
    llm_max_tokens: int = 2048
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 2

    # --- embeddings (Voyage AI) ---------------------------------------------
    voyage_api_key: str | None = None
    embed_model: str = "voyage-3.5-lite"
    # The dimension MUST match the vector(N) column. Change it without
    # recreating the table and re-ingesting, and every insert/query fails.
    # voyage-3.5-lite supports 256 / 512 / 1024 / 2048.
    embed_dim: int = 1024
    embed_batch_size: int = 128
    embed_timeout_s: float = 30.0
    embed_max_retries: int = 3

    # --- chunking ----------------------------------------------------------
    chunk_size: int = 800      # characters, not tokens
    chunk_overlap: int = 120

    # --- retrieval ---------------------------------------------------------
    top_k: int = 4
    # Cosine similarity floor. Below this a chunk is "not really relevant",
    # and answering from it is how a RAG system hallucinates confidently.
    # Calibrate, don't guess: each embedding model has its own score range.
    # (With the earlier local bge-small model an off-topic question scored
    # 0.41, so a 0.35 floor let it through.) Re-measure with
    # `scripts/smoke.py --retrieval` whenever the embed model changes.
    min_similarity: float = 0.55
    # Budget guard: hard cap on context characters sent to the LLM.
    max_context_chars: int = 12_000

    # --- cache -------------------------------------------------------------
    cache_ttl_s: int = 3600

    log_level: str = "INFO"


settings = Settings()
