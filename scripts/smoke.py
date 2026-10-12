"""
Smoke check: known questions with known answers.

    python scripts/smoke.py              full check via POST /ask (needs API key)
    python scripts/smoke.py --retrieval  retrieval only, no LLM, no server needed

This is your first move after any change, and the first move in a debugging
drill: is the system answering correctly RIGHT NOW, and if not, is it the
retrieval half or the generation half that broke?
"""

import asyncio
import json
import sys
import time
import urllib.request

# (question, file the answer lives in, a fact the answer must contain)
CASES = [
    ("How many days do I have to ask for a refund?", "refund-policy.md", "7"),
    ("What score do I need on the final assessment to get a certificate?",
     "certificates.md", "70"),
    ("How much does the yearly Pro plan cost?", "pricing-plans.md", "6,999"),
    ("How long are AI tutor conversations kept?", "ai-tutor.md", "90"),
    ("What should I check first when tutor latency alerts?",
     "oncall-runbook.md", "status page"),
    ("Who won the 2011 cricket world cup?", None, "couldn't find"),
]


async def retrieval_only() -> int:
    sys.path.insert(0, ".")
    from app import embeddings
    from app.retrieval import retrieve

    embeddings.load_model()
    failures = 0
    for question, expected_file, _ in CASES:
        hits = await retrieve(question, 4)
        top = hits[0].source if hits else None
        ok = top == expected_file
        failures += not ok
        sim = f"{hits[0].similarity:.3f}" if hits else "  -  "
        print(f"{'PASS' if ok else 'FAIL'}  top={str(top):<20} sim={sim}  {question}")
    return failures


def full() -> int:
    failures = 0
    for question, expected_file, fact in CASES:
        body = json.dumps({"question": question}).encode()
        req = urllib.request.Request("http://127.0.0.1:8000/ask", data=body,
                                     headers={"content-type": "application/json"})
        start = time.perf_counter()
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
        ms = round((time.perf_counter() - start) * 1000)
        sources = [s["source"] for s in data["sources"]]
        ok = fact.lower() in data["answer"].lower() and (
            expected_file is None or expected_file in sources)
        failures += not ok
        tag = "cached" if data["cached"] else f"{ms}ms"
        print(f"{'PASS' if ok else 'FAIL'}  {tag:>7}  {question}")
        if not ok:
            print(f"      answer : {data['answer'][:160]!r}")
            print(f"      sources: {sources}")
    return failures


if __name__ == "__main__":
    failed = asyncio.run(retrieval_only()) if "--retrieval" in sys.argv else full()
    print(f"\n{len(CASES) - failed}/{len(CASES)} passed")
    sys.exit(1 if failed else 0)
