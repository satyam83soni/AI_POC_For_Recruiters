"""
Smoke check: known questions with known answers.

    python scripts/smoke.py              full check via POST /ask (needs API key)
    python scripts/smoke.py --retrieval  retrieval only, no LLM, no server needed

This is your first move after any change, and the first move in a debugging
drill: is the system answering correctly RIGHT NOW, and if not, is it the
retrieval half or the generation half that broke?
"""

# >> asyncio = Python's built-in async runtime (the event loop Node gives you for free).
import asyncio
# >> json = built-in JSON encode/decode (json.dumps ~ JSON.stringify, json.loads ~ JSON.parse).
import json
# >> sys = interpreter access: sys.argv (like process.argv), sys.exit, sys.path.
import sys
# >> time = built-in time module; perf_counter() is used as a stopwatch.
import time
# >> urllib.request = built-in HTTP client (a low-level `fetch`), so no extra dependency needed.
import urllib.request

# (question, file the answer lives in, a fact the answer must contain)
# >> A list of tuples (fixed-size, read-only arrays). Each one is a test case.
CASES = [
    # >> Case: refund window; answer must be in refund-policy.md and mention "7".
    ("How many days do I have to ask for a refund?", "refund-policy.md", "7"),
    # >> Case: certificate pass mark (this tuple is split over two lines).
    ("What score do I need on the final assessment to get a certificate?",
     "certificates.md", "70"),
    # >> Cases: Pro plan price and how long tutor chats are kept.
    ("How much does the yearly Pro plan cost?", "pricing-plans.md", "6,999"),
    ("How long are AI tutor conversations kept?", "ai-tutor.md", "90"),
    # >> Case: on-call runbook question (split over two lines).
    ("What should I check first when tutor latency alerts?",
     "oncall-runbook.md", "status page"),
    # >> Off-topic case: no file should match (None) and the answer must say "couldn't find".
    ("Who won the 2011 cricket world cup?", None, "couldn't find"),
]


# >> Mode 1: test only the search half (no LLM, no server). Returns the number of failures.
async def retrieval_only() -> int:
    # >> Add the current folder to Python's import search path so `import app...` works
    # >> when running this script from the project root.
    sys.path.insert(0, ".")
    # >> Import inside the function so the default (full) mode doesn't need the app's deps.
    from app import embeddings
    # >> The async vector-search function.
    from app.retrieval import retrieve

    # >> Create the Voyage client (normally done at server startup).
    embeddings.init_client()
    # >> Failure counter.
    failures = 0
    # >> Loop over cases, unpacking each tuple into 3 names; `_` = "ignore this value".
    for question, expected_file, _ in CASES:
        # >> Search for the top 4 chunks for this question.
        hits = await retrieve(question, 4)
        # >> File of the best hit, or None if there were no hits (inline if/else).
        top = hits[0].source if hits else None
        # >> Passed if the best hit came from the expected file (== compares values).
        ok = top == expected_file
        # >> `not ok` is True/False, which counts as 1/0, so this adds 1 on failure.
        failures += not ok
        # >> Similarity formatted to 3 decimals (:.3f), or a placeholder if no hits.
        sim = f"{hits[0].similarity:.3f}" if hits else "  -  "
        # >> Print a result row; {str(top):<20} left-aligns the file name in 20 characters.
        print(f"{'PASS' if ok else 'FAIL'}  top={str(top):<20} sim={sim}  {question}")
    # >> Return the number of failed cases.
    return failures


# >> Mode 2: full end-to-end test through the running server's POST /ask endpoint.
def full() -> int:
    # >> Failure counter.
    failures = 0
    # >> Loop over cases, unpacking all three values.
    for question, expected_file, fact in CASES:
        # >> Build the JSON request body and .encode() it from text into bytes for sending.
        body = json.dumps({"question": question}).encode()
        # >> Build a request object (passing `data` makes it a POST), like preparing a fetch call...
        req = urllib.request.Request("http://127.0.0.1:8000/ask", data=body,
                                     # >> ...with a JSON content-type header.
                                     headers={"content-type": "application/json"})
        # >> Start the stopwatch.
        start = time.perf_counter()
        # >> Send the request; `with` auto-closes the response when done. Timeout: 120 seconds.
        with urllib.request.urlopen(req, timeout=120) as resp:
            # >> Read the response bytes and parse the JSON into a dict.
            data = json.loads(resp.read())
        # >> Elapsed time in milliseconds.
        ms = round((time.perf_counter() - start) * 1000)
        # >> List comprehension: pull the "source" field out of each source (like .map(s => s.source)).
        sources = [s["source"] for s in data["sources"]]
        # >> Pass if the fact appears in the answer (case-insensitive; `in` = substring check)...
        ok = fact.lower() in data["answer"].lower() and (
            # >> ...and the expected file is among the sources (skipped when no file is expected).
            expected_file is None or expected_file in sources)
        # >> Add 1 on failure.
        failures += not ok
        # >> Show "cached" if served from cache, else the time like "850ms".
        tag = "cached" if data["cached"] else f"{ms}ms"
        # >> Print the row; {tag:>7} right-aligns the tag in 7 characters.
        print(f"{'PASS' if ok else 'FAIL'}  {tag:>7}  {question}")
        # >> On failure, print details to help debug.
        if not ok:
            # >> First 160 characters of the answer; !r prints it in quoted repr() form.
            print(f"      answer : {data['answer'][:160]!r}")
            # >> The list of source files that came back.
            print(f"      sources: {sources}")
    # >> Return the number of failed cases.
    return failures


# >> True only when this file is run directly (python scripts/smoke.py), not when imported.
if __name__ == "__main__":
    # >> If "--retrieval" was passed on the command line, run the async mode with asyncio.run
    # >> (starts an event loop and waits for it); otherwise run the full HTTP check.
    failed = asyncio.run(retrieval_only()) if "--retrieval" in sys.argv else full()
    # >> Print the summary, e.g. "5/6 passed" (\n adds a blank line first).
    print(f"\n{len(CASES) - failed}/{len(CASES)} passed")
    # >> Exit code 1 if anything failed, 0 if all passed (like process.exit(code)), useful for CI.
    sys.exit(1 if failed else 0)
