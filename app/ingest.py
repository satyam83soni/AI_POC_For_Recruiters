"""
Ingest: text -> chunks -> embeddings -> rows in Postgres.

Idempotent by source: same source + same content hash is a no-op; same source
with changed content replaces the old chunks.
"""

# >> hashlib: built-in hashing module (like Node's `crypto.createHash`).
import hashlib
# >> logging: built-in logging module.
import logging
# >> time: built-in module, used for perf_counter() timing.
import time

# >> A LangChain helper that splits long text into smaller overlapping chunks.
from langchain_text_splitters import RecursiveCharacterTextSplitter
# >> select(): SQLAlchemy's SELECT query builder.
from sqlalchemy import select
# >> AsyncSession: the type of a DB session (used as a type hint for the parameter).
from sqlalchemy.ext.asyncio import AsyncSession

# >> This project's cache module (Redis) and embeddings module (text -> vectors).
from app import cache, embeddings
# >> App settings (chunk size, overlap, ...).
from app.config import settings
# >> The table classes from app/db.py.
from app.db import Chunk, Document
# >> Project helper for structured log lines.
from app.logging import log

# >> Logger named after this module ("app.ingest").
logger = logging.getLogger(__name__)

# Tries to split on paragraphs first, then lines, then sentences, then words,
# so chunks break at natural boundaries rather than mid-word. Overlap repeats
# the tail of one chunk at the head of the next, so a fact that straddles a
# boundary still appears whole in at least one chunk.
# >> Create one splitter, reused for every document (module-level = created once on import).
splitter = RecursiveCharacterTextSplitter(
    # >> Max characters per chunk.
    chunk_size=settings.chunk_size,
    # >> How many characters each chunk repeats from the end of the previous one.
    chunk_overlap=settings.chunk_overlap,
    # >> Split points to try, in order: markdown "## " headings, blank lines, lines, sentences, words,
    # >> and finally "" (any character) as a last resort.
    separators=["\n## ", "\n\n", "\n", ". ", " ", ""],
)


# >> Async function that stores one document. Returns a dict (a JS-object-like key/value map).
async def ingest_document(
    # >> `*` means everything after it must be passed by name: source=..., title=..., text=...
    session: AsyncSession, *, source: str, title: str, text: str
) -> dict:
    # >> Start timestamp (high-precision seconds).
    start = time.perf_counter()
    # >> .encode() turns the string into bytes; sha256(...).hexdigest() = 64-char hex hash
    # >> (like crypto.createHash('sha256').update(text).digest('hex')).
    content_hash = hashlib.sha256(text.encode()).hexdigest()

    # >> Look up an existing Document with this source; session.scalar returns the first object or None.
    existing = await session.scalar(select(Document).where(Document.source == source))
    # >> If it exists and its text hasn't changed (same hash)...
    if existing and existing.content_hash == content_hash:
        # >> ...log that we're skipping it...
        log(logger, "ingest skipped, unchanged", source=source)
        # >> ...and return early without doing any work.
        return {"source": source, "status": "unchanged", "chunks": 0}

    # >> Split the text into a list of chunk strings.
    pieces = splitter.split_text(text)
    # >> Get one embedding vector per chunk (a list of lists of floats), in the same order.
    vectors = await embeddings.embed_passages(pieces)
    # >> Milliseconds spent so far (splitting + embedding), rounded to an integer.
    embed_ms = round((time.perf_counter() - start) * 1000)

    # >> If an older version of this document exists (content changed)...
    if existing:
        # Cascade removes the old chunks with it.
        # >> ...mark it for deletion (its chunks go too, via ON DELETE CASCADE)...
        await session.delete(existing)
        # >> ...and flush = send the pending DELETE to Postgres now (still inside the transaction,
        # >> not committed), so the unique `source` is free for the new row below.
        await session.flush()

    # >> Create a new Document object in Python memory (not in the DB yet).
    doc = Document(source=source, title=title, content_hash=content_hash)
    # >> Attach a list of Chunk objects to it (list comprehension, like .map()).
    doc.chunks = [
        # >> One Chunk per piece: its position, text and vector.
        Chunk(chunk_index=i, content=piece, embedding=vec)
        # >> zip pairs pieces[k] with vectors[k] (strict=True: error if lengths differ);
        # >> enumerate adds the index i (like arr.forEach((x, i) => ...)).
        for i, (piece, vec) in enumerate(zip(pieces, vectors, strict=True))
    ]
    # >> Add the document (and, through the relationship, its chunks) to the session for saving.
    session.add(doc)
    # >> Commit the transaction: INSERTs (and the earlier DELETE) become permanent in Postgres.
    await session.commit()

    # Only after the commit succeeds: invalidate cached answers.
    # >> Increase the corpus version in Redis so old cached answers are no longer used.
    await cache.bump_corpus_version()

    # >> Inline if/else (like `existing ? "replaced" : "created"`).
    status = "replaced" if existing else "created"
    # >> Log the result with counts and timings (total_ms = time since start, in ms).
    log(logger, "ingest done", source=source, status=status, chunks=len(pieces),
        embed_ms=embed_ms, total_ms=round((time.perf_counter() - start) * 1000))
    # >> Return a summary dict (FastAPI sends this back as JSON).
    return {"source": source, "status": status, "chunks": len(pieces)}
