# RAG POC

A small retrieval-augmented Q&A service on the LearnTube AI Backend stack:
FastAPI, LangGraph, LangChain text splitters, Claude (answers), Voyage AI (embeddings), Postgres + pgvector, Redis, Docker.

Built to learn where this kind of system breaks, then drill debugging it against a 30-minute clock.

## Run it

```powershell
docker compose up -d --wait                        # Postgres :5433, Redis :6380
copy .env.example .env                             # then set ANTHROPIC_API_KEY and VOYAGE_API_KEY
.\.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000
.\.venv\Scripts\python.exe scripts\ingest_docs.py  # load data/docs
.\.venv\Scripts\python.exe scripts\smoke.py        # 6 known questions, expect 6/6
```

Swagger UI: http://127.0.0.1:8000/docs

## One request, end to end

```
POST /ask {"question": ...}
  │
  ├─ middleware          assign request_id, time the request          app/main.py
  ├─ cache.get_answer    Redis, key = model|top_k|corpus_version|q    app/cache.py
  │
  └─ rag_graph (LangGraph)                                            app/graph.py
       retrieve ─► embed question (Voyage API, input_type=query)      app/embeddings.py
                   ORDER BY embedding <=> q LIMIT k (HNSW index)      app/retrieval.py
                   drop chunks below min_similarity
          │
          ├─ no chunks ─► not_found  (no LLM call)
          └─ chunks ────► extract   cheap model (Haiku 5.5) -> JSON    app/llm.py
                            │       {found, answer, citations}, schema
                            │       enforced by structured outputs
                            ▼
                          validate  plain code: cited [n] exist, match     app/graph.py
                            │       the citations list
                            ├─ invalid, retry left ─► extract (told what failed)
                            ├─ invalid / found=false ─► not_found
                            ▼
                          verify    heavy model (Opus 5.5) -> verdict      app/llm.py
                                    {supported, issues, corrected_answer}
                                    supported -> draft; corrected -> fix
                                    (re-checked); else not_found
```

`ANSWER_PIPELINE=stream` switches back to the single-model path: Opus writes the
answer directly and `/ask/stream` sends it token by token. In the verified
pipeline the answer only exists once the verifier passes it, so the stream
carries it as one `token` event.

Ingest: `POST /ingest` → split (800 chars, 120 overlap) → embed → replace the document's chunks → bump corpus version so cached answers go stale.

## Where to look first when it breaks

| Symptom | First evidence |
|---|---|
| Every request slow at once | event loop blocked: a sync call in `async def` |
| 503 on ingest or ask, `embed call` slow | Voyage rate limit / latency: `embed call` log lines |
| `expected 1024 dimensions, not N` | `embed_dim` vs the `vector(N)` column vs the model |
| Hangs, then pool timeout | `pool_timeout` errors, a session not closed or held across the LLM call |
| Wrong or "not found" answers | `retrieval done` log: `top_similarity`, `kept` |
| "Not found" although retrieval kept chunks | `extraction validated` (`problems`) and `answer verified` (`outcome`, `issues`) logs |
| Slow search | `search_ms` in logs, `EXPLAIN ANALYZE` on the vector query |
| 503 / 504 from /ask | LLM rate limit / timeout handlers in `main.py` |
| Stream stops mid-answer | `event: error` in the SSE stream + `stream failed` log |
| Stale answers after ingest | cache key / corpus version |

Every log line is JSON with a `request_id`. To follow one request:

```powershell
Select-String <request_id> server.log
```
