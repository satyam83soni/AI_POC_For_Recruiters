"""Vector search: question -> the top-k most similar chunks."""

# >> logging: Python's built-in logging module (like a built-in winston/pino).
import logging
# >> time: built-in module; used here for perf_counter() timing.
import time
# >> dataclass: decorator that auto-generates __init__ etc. for a class that just holds fields.
from dataclasses import dataclass

# >> select(): SQLAlchemy's query builder for SELECT statements (like knex's `.select()`).
from sqlalchemy import select

# >> This project's embeddings module (turns text into vectors).
from app import embeddings
# >> App settings (min_similarity etc.).
from app.config import settings
# >> The table classes and the session factory defined in app/db.py.
from app.db import Chunk, Document, SessionLocal
# >> Project helper that writes a structured log line.
from app.logging import log

# >> Create a logger named after this module ("app.retrieval").
logger = logging.getLogger(__name__)


# >> @dataclass turns the class below into a simple record type with an auto constructor.
@dataclass
# >> One search result (like a TS interface + constructor in one).
class RetrievedChunk:
    # >> Field: database id of the chunk (int).
    chunk_id: int
    # >> Fields: the document's source and title (strings).
    source: str
    title: str
    # >> Field: the chunk's text.
    content: str
    # >> Field: how similar it is to the question (float, higher = more similar).
    similarity: float


# >> Async function: takes the question and how many results to fetch, returns a list of RetrievedChunk.
async def retrieve(question: str, top_k: int) -> list[RetrievedChunk]:
    # >> perf_counter(): a high-precision clock in seconds (like performance.now()); start timestamp.
    t0 = time.perf_counter()
    # >> Turn the question into an embedding vector (await = wait for the async call).
    qvec = await embeddings.embed_query(question)
    # >> Timestamp after embedding.
    t1 = time.perf_counter()

    # <=> is cosine DISTANCE (0 = identical, 2 = opposite).
    # similarity = 1 - distance, so higher is better.
    #
    # ORDER BY distance LIMIT k is the exact shape the HNSW index serves.
    # Put the similarity threshold in WHERE instead and Postgres may give up
    # on the index; filtering the k results afterwards keeps it on the index.
    # >> Build (not run) the SQL expression `embedding <=> :qvec` -- pgvector's cosine distance.
    distance = Chunk.embedding.cosine_distance(qvec)
    # >> Build the query; the outer parentheses just let it span several lines.
    stmt = (
        # >> SELECT these columns...
        select(Chunk.id, Chunk.content, Document.source, Document.title,
               # >> ...plus (1 - distance) named "similarity" via .label (SQL `AS similarity`).
               (1 - distance).label("similarity"))
        # >> JOIN documents ON chunks.document_id = documents.id.
        .join(Document, Chunk.document_id == Document.id)
        # >> ORDER BY distance (closest first) -- this is what lets Postgres use the HNSW index.
        .order_by(distance)
        # >> LIMIT top_k rows.
        .limit(top_k)
    )

    # Its own short-lived session: holds a pool connection only for the query,
    # not for the many seconds the LLM takes to answer afterwards.
    # >> Open a session just for this query; it is closed (connection returned) when the block ends.
    async with SessionLocal() as session:
        # >> Run the query, then .all() turns the result into a list of row objects.
        rows = (await session.execute(stmt)).all()
    # >> Timestamp after the DB search.
    t2 = time.perf_counter()

    # >> List comprehension (like rows.map(...)): build one RetrievedChunk per row.
    hits = [
        # >> round(x, 4): round similarity to 4 decimals; float() converts the DB number to a Python float.
        RetrievedChunk(r.id, r.source, r.title, r.content, round(float(r.similarity), 4))
        # >> Loop over every row (r) returned by the query.
        for r in rows
    ]
    # >> Keep only hits at or above the minimum similarity (like hits.filter(...)).
    kept = [h for h in hits if h.similarity >= settings.min_similarity]

    # >> Write a log line with the message and timing/count details as key=value fields.
    log(logger, "retrieval done",
        # >> Durations in ms: (seconds difference) * 1000, rounded to a whole number.
        embed_ms=round((t1 - t0) * 1000), search_ms=round((t2 - t1) * 1000),
        # >> len() = length of a list (like arr.length): how many found vs kept.
        candidates=len(hits), kept=len(kept),
        # >> Best similarity, or None if there were no hits (Python's inline if/else, like `a ? b : c`).
        top_similarity=hits[0].similarity if hits else None)
    # >> Return the filtered results.
    return kept
