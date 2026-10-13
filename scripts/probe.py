"""
Concurrency probe: one real question, plus fast background traffic at the same time.

    python scripts/probe.py

While ONE /ask request is in flight, this fires a cheap GET /health every 100 ms
and records how long each one takes. /health touches nothing (no DB, no Redis,
no APIs), so on a healthy async server it answers in a few milliseconds no
matter what else is running.

Uses 1 embedding call, so it is safe under the Voyage free-tier rate limit.
"""

import asyncio
import statistics
import time

import httpx

API = "http://127.0.0.1:8000"
QUESTION = "What does the Pro plan include, and how many devices can I download on?"


async def ask(client: httpx.AsyncClient, done: asyncio.Event) -> None:
    start = time.perf_counter()
    # A unique suffix per run so the answer cache never short-circuits the probe.
    question = f"{QUESTION} (ref {time.time_ns() % 100000})"
    r = await client.post(f"{API}/ask", json={"question": question}, timeout=120)
    ms = round((time.perf_counter() - start) * 1000)
    print(f"  /ask            {r.status_code}  {ms:>6} ms  cached={r.json().get('cached')}"
          f"  request_id={r.headers.get('x-request-id')}")
    done.set()


async def health_loop(client: httpx.AsyncClient, done: asyncio.Event) -> list[float]:
    timings: list[float] = []
    while not done.is_set():
        start = time.perf_counter()
        await client.get(f"{API}/health", timeout=30)
        timings.append((time.perf_counter() - start) * 1000)
        await asyncio.sleep(0.1)
    return timings


async def main() -> None:
    async with httpx.AsyncClient() as client:
        await client.get(f"{API}/health")  # warm the connection
        done = asyncio.Event()
        _, timings = await asyncio.gather(ask(client, done), health_loop(client, done))

    slow = [t for t in timings if t > 100]
    print(f"  /health x{len(timings):<4}      median {statistics.median(timings):6.0f} ms"
          f"   max {max(timings):6.0f} ms   slower than 100 ms: {len(slow)}")
    print("\n  every /health latency, in order (ms):")
    print("  " + " ".join(f"{t:.0f}" for t in timings))


if __name__ == "__main__":
    asyncio.run(main())
