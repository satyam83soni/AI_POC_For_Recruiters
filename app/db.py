"""Engine, sessions, tables, and one-time schema setup."""

from collections.abc import AsyncGenerator
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, Index, String, Text, func, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from app.config import settings

engine = create_async_engine(
    settings.database_url,
    pool_size=settings.db_pool_size,
    max_overflow=settings.db_max_overflow,
    # How long a request waits for a free connection before raising
    # TimeoutError ("QueuePool limit ... reached"). A leak shows up here.
    pool_timeout=settings.db_pool_timeout_s,
    pool_pre_ping=True,
)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    # `async with` returns the connection to the pool even if the handler raises.
    async with SessionLocal() as session:
        yield session


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    # Where it came from (file name / URL). Re-ingesting the same source
    # replaces it instead of duplicating it.
    source: Mapped[str] = mapped_column(String(500), unique=True)
    title: Mapped[str] = mapped_column(String(500))
    # sha256 of the text: same hash on re-ingest -> skip the embedding work.
    content_hash: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan", passive_deletes=True
    )


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    chunk_index: Mapped[int]
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(settings.embed_dim))

    document: Mapped[Document] = relationship(back_populates="chunks")

    __table_args__ = (
        # Approximate nearest-neighbour index. Without it every query compares
        # the question against EVERY chunk (a sequential scan) -- fine at 50
        # rows, a slow-query incident at 5 million.
        #
        # The opclass must match the operator the query uses:
        #   vector_cosine_ops  <->  <=>  (cosine_distance)
        #   vector_l2_ops      <->  <->  (l2_distance)
        # Mismatch them and Postgres silently ignores the index.
        Index(
            "ix_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


async def init_db() -> None:
    """Create the extension and tables if missing. Idempotent."""
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
