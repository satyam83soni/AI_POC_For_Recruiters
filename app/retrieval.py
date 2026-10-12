"""Vector search: question -> the top-k most similar chunks."""

import logging
import time
from dataclasses import dataclass

from sqlalchemy import select

from app import embeddings
from app.config import settings
from app.db import Chunk, Document, SessionLocal
from app.logging import log

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    chunk_id: int
    source: str
    title: str
    content: str
    similarity: float


async def retrieve(question: str, top_k: int) -> list[RetrievedChunk]:
    t0 = time.perf_counter()
    qvec = await embeddings.embed_query(question)
    t1 = time.perf_counter()

    # <=> is cosine DISTANCE (0 = identical, 2 = opposite).
    # similarity = 1 - distance, so higher is better.
    #
    # ORDER BY distance LIMIT k is the exact shape the HNSW index serves.
    # Put the similarity threshold in WHERE instead and Postgres may give up
    # on the index; filtering the k results afterwards keeps it on the index.
    distance = Chunk.embedding.cosine_distance(qvec)
    stmt = (
        select(Chunk.id, Chunk.content, Document.source, Document.title,
               (1 - distance).label("similarity"))
        .join(Document, Chunk.document_id == Document.id)
        .order_by(distance)
        .limit(top_k)
    )

    # Its own short-lived session: holds a pool connection only for the query,
    # not for the many seconds the LLM takes to answer afterwards.
    async with SessionLocal() as session:
        rows = (await session.execute(stmt)).all()
    t2 = time.perf_counter()

    hits = [
        RetrievedChunk(r.id, r.source, r.title, r.content, round(float(r.similarity), 4))
        for r in rows
    ]
    kept = [h for h in hits if h.similarity >= settings.min_similarity]

    log(logger, "retrieval done",
        embed_ms=round((t1 - t0) * 1000), search_ms=round((t2 - t1) * 1000),
        candidates=len(hits), kept=len(kept),
        top_similarity=hits[0].similarity if hits else None)
    return kept
