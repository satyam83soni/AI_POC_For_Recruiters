"""
Ingest: text -> chunks -> embeddings -> rows in Postgres.

Idempotent by source:
  - same text, same title   -> no-op
  - same text, new title    -> UPDATE the title only (no re-embedding)
  - changed text            -> replace the document and all its chunks
  - new source              -> create
"""

import hashlib
import logging
import time

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app import cache, embeddings
from app.config import settings
from app.db import Chunk, Document
from app.logging import log

logger = logging.getLogger(__name__)

# Tries to split on paragraphs first, then lines, then sentences, then words,
# so chunks break at natural boundaries rather than mid-word. Overlap repeats
# the tail of one chunk at the head of the next, so a fact that straddles a
# boundary still appears whole in at least one chunk.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap,
    separators=["\n## ", "\n\n", "\n", ". ", " ", ""],
)


async def ingest_document(
    session: AsyncSession, *, source: str, title: str, text: str
) -> dict:
    start = time.perf_counter()
    # The hash covers the TEXT only, because only the text decides the chunks
    # and embeddings. Metadata like the title is compared separately, so a
    # title change never pays for re-embedding.
    content_hash = hashlib.sha256(text.encode()).hexdigest()

    existing = await session.scalar(select(Document).where(Document.source == source))
    # The SELECT above opened a transaction and took a pool connection. End it
    # now so no connection is held during the slow embedding API call below.
    # (expire_on_commit=False keeps `existing` readable after this.)
    await session.commit()

    if existing and existing.content_hash == content_hash:
        if existing.title == title:
            log(logger, "ingest skipped, unchanged", source=source)
            return {"source": source, "status": "unchanged", "chunks": 0}

        # Same text, new title: one UPDATE on the existing row. Chunks and
        # embeddings are untouched. Don't read existing.chunks here -- it is
        # not loaded, and lazy-loading it in async code raises MissingGreenlet.
        existing.title = title
        await session.commit()
        await cache.bump_corpus_version()  # cached answers include source titles
        log(logger, "ingest done", source=source, status="title_updated",
            total_ms=round((time.perf_counter() - start) * 1000))
        return {"source": source, "status": "title_updated", "chunks": 0}

    # New or changed text: split and embed OUTSIDE any transaction.
    pieces = splitter.split_text(text)
    vectors = await embeddings.embed_passages(pieces)
    embed_ms = round((time.perf_counter() - start) * 1000)

    # One short write transaction: delete the old version (if any) and insert
    # the new one together, so readers never see the document missing.
    if existing:
        # Cascade removes the old chunks with it.
        await session.delete(existing)
        await session.flush()

    doc = Document(source=source, title=title, content_hash=content_hash)
    doc.chunks = [
        Chunk(chunk_index=i, content=piece, embedding=vec)
        for i, (piece, vec) in enumerate(zip(pieces, vectors, strict=True))
    ]
    session.add(doc)
    await session.commit()

    # Only after the commit succeeds: invalidate cached answers.
    await cache.bump_corpus_version()

    status = "replaced" if existing else "created"
    log(logger, "ingest done", source=source, status=status, chunks=len(pieces),
        embed_ms=embed_ms, total_ms=round((time.perf_counter() - start) * 1000))
    return {"source": source, "status": status, "chunks": len(pieces)}
