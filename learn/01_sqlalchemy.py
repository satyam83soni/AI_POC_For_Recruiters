r"""
SQLAlchemy 2.0 async + pgvector -- ONLY the parts rag-poc uses.

Run (Docker must be up):   .\.venv\Scripts\python.exe learn\01_sqlalchemy.py

Uses its own throwaway table `learn_notes`, dropped at the end, so it never
touches the real documents/chunks tables.

Node mental model:
    engine        = new pg.Pool()             one per app, owns connections
    session       = one transaction            borrow a connection, query, give back
    model class   = a Prisma/TypeORM model     a Python class mapped to a table
    select(...)   = a query builder (Knex)     builds SQL, nothing runs until execute
"""

import asyncio

from pgvector.sqlalchemy import Vector
from sqlalchemy import String, select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

URL = "postgresql+asyncpg://rag:rag@localhost:5433/rag"


# --- 1. A model = a table --------------------------------------------------
class Base(DeclarativeBase):
    pass


class Note(Base):
    __tablename__ = "learn_notes"

    id: Mapped[int] = mapped_column(primary_key=True)     # SERIAL PRIMARY KEY
    topic: Mapped[str] = mapped_column(String(50))         # VARCHAR(50) NOT NULL
    body: Mapped[str | None]                               # TEXT NULL (| None = nullable)
    # vector(3): tiny on purpose so you can read the numbers. rag-poc uses 384.
    embedding: Mapped[list[float]] = mapped_column(Vector(3))


def title(s: str) -> None:
    print(f"\n=== {s} ===")


async def main() -> None:
    # --- 2. Engine (pool) and session factory -------------------------------
    # echo=True prints every SQL statement. Turn it on whenever you're unsure
    # what your Python turned into.
    engine = create_async_engine(URL, echo=False)
    Session = async_sessionmaker(engine, expire_on_commit=False)

    async with engine.begin() as conn:                     # begin() = BEGIN ... COMMIT
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.drop_all)        # start clean
        await conn.run_sync(Base.metadata.create_all)      # CREATE TABLE learn_notes

    # --- 3. Insert: add -> commit -------------------------------------------
    title("insert")
    async with Session() as session:                       # like: const client = await pool.connect()
        session.add_all([
            Note(topic="refund", body="7 day refund window", embedding=[1.0, 0.0, 0.0]),
            Note(topic="refund", body="UPI refunds in 48h", embedding=[0.9, 0.1, 0.0]),
            Note(topic="certificate", body="need 70% to pass", embedding=[0.0, 1.0, 0.0]),
            Note(topic="pricing", body="Pro is Rs 799/month", embedding=[0.0, 0.0, 1.0]),
        ])
        await session.commit()                             # nothing is saved until this
        print("4 rows inserted")
    # leaving `async with` = client.release(). Forget this and the pool leaks.

    async with Session() as session:
        # --- 4. Select: build, execute, unwrap ------------------------------
        title("select ... where ... order by")
        stmt = select(Note).where(Note.topic == "refund").order_by(Note.id)
        print("SQL it builds:\n ", str(stmt).replace("\n", " "))
        result = await session.execute(stmt)
        for note in result.scalars():                      # scalars(): Row(Note,) -> Note
            print("  ", note.id, note.topic, note.body)

        # --- 5. The four ways to unwrap a result ----------------------------
        title("unwrapping results")
        one = await session.scalar(select(Note).where(Note.id == 1))
        print("scalar()            ->", one.body)          # first value or None
        missing = await session.scalar(select(Note).where(Note.id == 999))
        print("scalar() no match   ->", missing)
        by_pk = await session.get(Note, 3)                 # lookup by primary key
        print("get(Note, 3)        ->", by_pk.body)
        rows = (await session.execute(select(Note.topic, Note.body))).all()
        print("select(cols).all()  ->", rows[0], "<- a Row, use row.topic")

        # --- 6. Vector search: exactly what app/retrieval.py does ----------
        title("vector search  ORDER BY embedding <=> query LIMIT 2")
        query_vec = [0.95, 0.05, 0.0]                      # "a refund-ish question"
        distance = Note.embedding.cosine_distance(query_vec)
        stmt = (
            select(Note.body, (1 - distance).label("similarity"))
            .order_by(distance)                            # closest first
            .limit(2)
        )
        for r in (await session.execute(stmt)).all():
            print(f"   sim={r.similarity:.3f}  {r.body}")

        # --- 7. flush vs commit, and rollback -------------------------------
        title("flush vs commit")
        n = Note(topic="tmp", body="not committed", embedding=[0.5, 0.5, 0.0])
        session.add(n)
        await session.flush()                              # INSERT sent, id assigned...
        print("after flush, id =", n.id, "(row exists only inside this transaction)")
        await session.rollback()                           # ...and now undone
        gone = await session.scalar(select(Note).where(Note.body == "not committed"))
        print("after rollback   ->", gone)

        # --- 8. EXPLAIN: how you prove a query is slow and why --------------
        title("EXPLAIN (what you'd run in a slow-query incident)")
        plan = await session.execute(text(
            "EXPLAIN SELECT body FROM learn_notes "
            "ORDER BY embedding <=> '[0.95,0.05,0]' LIMIT 2"))
        for (line,) in plan:
            print("  ", line)
        print("   ^ 'Seq Scan' = compares against every row. Fine for 4 rows,")
        print("     an incident at millions. rag-poc's HNSW index fixes that.")

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()                                 # close the pool
    print("\ncleaned up learn_notes")


# TRY: change query_vec in section 6 to [0, 0, 1]. Which note comes first now?
# TRY: delete `await session.commit()` in section 3. What does section 4 print?

if __name__ == "__main__":
    asyncio.run(main())
