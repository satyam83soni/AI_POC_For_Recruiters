"""
Load every .md file in data/docs into the running API via POST /ingest.

    python scripts/ingest_docs.py            (server must be running on :8000)

Goes through the HTTP API on purpose, so ingest is exercised exactly the way a
real client would use it. stdlib only -- no extra dependency for a script.
"""

import json
import pathlib
import sys
import urllib.error
import urllib.request

API = "http://127.0.0.1:8000"
DOCS = pathlib.Path(__file__).resolve().parent.parent / "data" / "docs"


def post(path: str, body: dict) -> dict:
    req = urllib.request.Request(
        API + path, data=json.dumps(body).encode(),
        headers={"content-type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def main() -> int:
    files = sorted(DOCS.glob("*.md"))
    if not files:
        print(f"no .md files in {DOCS}")
        return 1
    for f in files:
        text = f.read_text(encoding="utf-8")
        title = text.splitlines()[0].lstrip("# ").strip()
        try:
            result = post("/ingest", {"source": f.name, "title": title, "text": text})
        except urllib.error.HTTPError as exc:
            print(f"FAIL {f.name}: {exc.code} {exc.read().decode()[:200]}")
            return 1
        print(f"{result['status']:<9} {result['chunks']:>3} chunks  {f.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
