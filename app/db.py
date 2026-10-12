"""Engine, sessions, tables, and one-time schema setup."""

# >> AsyncGenerator: a type-hint name for an async function that uses `yield` (used on get_session).
from collections.abc import AsyncGenerator
# >> datetime: Python's built-in date+time type (like JS `Date`); used as the type of created_at.
from datetime import datetime

# >> Vector: pgvector's column type for SQLAlchemy -- stores an array of floats (an embedding).
from pgvector.sqlalchemy import Vector
# >> Column building blocks from SQLAlchemy: DateTime/String/Text = column types, ForeignKey = link
# >> to another table, Index = DB index, func = SQL functions (NOW()), text = raw SQL string.
from sqlalchemy import DateTime, ForeignKey, Index, String, Text, func, text
# >> Async (await-able) versions of SQLAlchemy: the session type, a session factory, and the engine.
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
# >> ORM pieces for declaring tables as Python classes (similar to defining Prisma models).
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# >> Import the app's settings object (values read from env vars / .env).
from app.config import settings

# >> Create the engine = a connection pool to Postgres (like `new pg.Pool({...})` in Node).
engine = create_async_engine(
    # >> The Postgres connection URL (postgresql+asyncpg://user:pass@host/db).
    settings.database_url,
    # >> How many connections the pool keeps open permanently.
    pool_size=settings.db_pool_size,
    # >> How many extra temporary connections may be opened when all pooled ones are busy.
    max_overflow=settings.db_max_overflow,
    # How long a request waits for a free connection before raising
    # TimeoutError ("QueuePool limit ... reached"). A leak shows up here.
    # >> Seconds to wait for a free connection before giving up with an error.
    pool_timeout=settings.db_pool_timeout_s,
    # >> Before handing out a connection, send a tiny "are you alive?" query; drops dead connections.
    pool_pre_ping=True,
)
# >> A factory that makes new sessions (a session = one unit of work/transaction on a connection).
# >> expire_on_commit=False: objects stay readable after commit; autoflush=False: no auto-writes.
SessionLocal = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)


# >> `async def` = an async function (like JS `async function`). The return hint says it is an async
# >> generator that yields one AsyncSession. FastAPI uses it as a dependency (Depends(get_session)).
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    # `async with` returns the connection to the pool even if the handler raises.
    # >> Open a session; `async with` closes it automatically afterwards (like try/finally release()).
    async with SessionLocal() as session:
        # >> `yield` hands the session to the route handler and pauses here until the request ends.
        yield session


# >> Base class all table classes inherit from; SQLAlchemy uses it to collect table definitions.
class Base(DeclarativeBase):
    # >> `pass` = empty body (Python needs at least one statement in a block).
    pass


# >> Declare the `documents` table as a Python class (one instance = one row).
class Document(Base):
    # >> `__tablename__` tells SQLAlchemy the real table name in Postgres.
    __tablename__ = "documents"

    # >> `Mapped[int]` = type hint saying this column holds an int; mapped_column makes it the primary key
    # >> (auto-increment id).
    id: Mapped[int] = mapped_column(primary_key=True)
    # Where it came from (file name / URL). Re-ingesting the same source
    # replaces it instead of duplicating it.
    # >> VARCHAR(500) column for the source name; unique=True means no two rows may share it.
    source: Mapped[str] = mapped_column(String(500), unique=True)
    # >> VARCHAR(500) column for a human-readable title.
    title: Mapped[str] = mapped_column(String(500))
    # sha256 of the text: same hash on re-ingest -> skip the embedding work.
    # >> 64-char column holding the sha256 hex fingerprint of the document's text.
    content_hash: Mapped[str] = mapped_column(String(64))
    # >> Timestamp-with-timezone column; `datetime` is the Python type it maps to.
    created_at: Mapped[datetime] = mapped_column(
        # >> server_default=func.now(): Postgres fills it with NOW() when the row is inserted.
        DateTime(timezone=True), server_default=func.now()
    )

    # >> Not a real column: a Python-side link to this document's Chunk rows (a list of Chunk objects).
    # >> "Chunk" is in quotes because that class is defined further down the file.
    chunks: Mapped[list["Chunk"]] = relationship(
        # >> back_populates: pairs with Chunk.document; cascade: deleting a Document deletes its chunks;
        # >> passive_deletes=True: let Postgres's ON DELETE CASCADE do it instead of loading each chunk.
        back_populates="document", cascade="all, delete-orphan", passive_deletes=True
    )


# >> Declare the `chunks` table: each row is one piece of a document plus its embedding vector.
class Chunk(Base):
    # >> Real table name in Postgres.
    __tablename__ = "chunks"

    # >> Auto-increment integer primary key.
    id: Mapped[int] = mapped_column(primary_key=True)
    # >> Integer column pointing at documents.id (a foreign key).
    document_id: Mapped[int] = mapped_column(
        # >> ondelete="CASCADE": Postgres deletes chunks when their document is deleted;
        # >> index=True: adds a normal index so lookups by document_id are fast.
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    # >> Integer column (position of the chunk inside its document); mapped_column is implied.
    chunk_index: Mapped[int]
    # >> TEXT column (unlimited length) holding the chunk's actual text.
    content: Mapped[str] = mapped_column(Text)
    # >> pgvector column of `embed_dim` floats -- the embedding used for similarity search.
    embedding: Mapped[list[float]] = mapped_column(Vector(settings.embed_dim))

    # >> Python-side link back to the parent Document object (pairs with Document.chunks).
    document: Mapped[Document] = relationship(back_populates="chunks")

    # >> `__table_args__` = extra table-level settings (here: a tuple holding one index definition).
    __table_args__ = (
        # Approximate nearest-neighbour index. Without it every query compares
        # the question against EVERY chunk (a sequential scan) -- fine at 50
        # rows, a slow-query incident at 5 million.
        #
        # The opclass must match the operator the query uses:
        #   vector_cosine_ops  <->  <=>  (cosine_distance)
        #   vector_l2_ops      <->  <->  (l2_distance)
        # Mismatch them and Postgres silently ignores the index.
        # >> Define a database index on the embedding column.
        Index(
            # >> The index's name in Postgres.
            "ix_chunks_embedding_hnsw",
            # >> The column being indexed.
            "embedding",
            # >> Use HNSW: a graph-based index for fast approximate "find nearest vectors" search.
            postgresql_using="hnsw",
            # >> HNSW tuning: m = links per graph node, ef_construction = build-time search effort.
            postgresql_with={"m": 16, "ef_construction": 64},
            # >> Operator class: build the index for cosine distance (the `<=>` operator).
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


# >> Async function run once at startup to make sure the DB schema exists; returns nothing (None).
async def init_db() -> None:
    """Create the extension and tables if missing. Idempotent."""
    # >> engine.begin(): grab a connection and start a transaction; commits automatically at the end.
    async with engine.begin() as conn:
        # >> Run raw SQL to enable the pgvector extension (no-op if already enabled).
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        # >> run_sync runs a normal (non-async) function on this connection; create_all creates every
        # >> table/index declared on Base that doesn't exist yet.
        await conn.run_sync(Base.metadata.create_all)
