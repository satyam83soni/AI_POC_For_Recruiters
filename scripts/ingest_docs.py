"""
Load every .md file in data/docs into the running API via POST /ingest.

    python scripts/ingest_docs.py            (server must be running on :8000)

Goes through the HTTP API on purpose, so ingest is exercised exactly the way a
real client would use it. stdlib only -- no extra dependency for a script.
"""

# >> json = built-in JSON encode/decode (dumps ~ JSON.stringify, loads ~ JSON.parse).
import json
# >> pathlib = built-in, object-based file paths (nicer than string paths; like Node's `path` + `fs`).
import pathlib
# >> sys = interpreter access; used here for sys.exit (like process.exit).
import sys
# >> urllib.error holds HTTP error types; urllib.request is the built-in HTTP client (low-level fetch).
import urllib.error
import urllib.request

# >> Base URL of the locally running API server.
API = "http://127.0.0.1:8000"
# >> __file__ = this script's path; .resolve() makes it absolute; .parent.parent goes up two folders
# >> to the project root; the `/` operator joins path parts -> <project>/data/docs.
DOCS = pathlib.Path(__file__).resolve().parent.parent / "data" / "docs"


# >> Helper: POST a dict as JSON to API + path and return the parsed JSON reply (like fetch + .json()).
def post(path: str, body: dict) -> dict:
    # >> Build the HTTP request object...
    req = urllib.request.Request(
        # >> ...full URL, and the body as JSON text encoded to bytes...
        API + path, data=json.dumps(body).encode(),
        # >> ...JSON content-type header and an explicit POST method.
        headers={"content-type": "application/json"}, method="POST")
    # >> Send it (timeout 120 s); `with` closes the response automatically afterwards.
    with urllib.request.urlopen(req, timeout=120) as resp:
        # >> Read the response bytes, parse JSON into a dict and return it.
        return json.loads(resp.read())


# >> Main script logic; returns an exit code (0 = success, 1 = failure).
def main() -> int:
    # >> .glob("*.md") finds all .md files in the folder; sorted() returns them in a stable order.
    files = sorted(DOCS.glob("*.md"))
    # >> `not files` is True when the list is empty.
    if not files:
        # >> Tell the user where we looked.
        print(f"no .md files in {DOCS}")
        # >> Exit code 1 = failure.
        return 1
    # >> Loop over each file path.
    for f in files:
        # >> Read the whole file as text (UTF-8), like fs.readFileSync(f, "utf8").
        text = f.read_text(encoding="utf-8")
        # >> Title = first line, with leading "#" / spaces removed (lstrip) and outer spaces trimmed.
        title = text.splitlines()[0].lstrip("# ").strip()
        # >> try/except = try/catch in JS.
        try:
            # >> Send this document to POST /ingest; f.name is just the file name (e.g. "faq.md").
            result = post("/ingest", {"source": f.name, "title": title, "text": text})
        # >> Catch HTTP errors (4xx/5xx responses) and name the error object `exc`.
        except urllib.error.HTTPError as exc:
            # >> Print status code and first 200 chars of the error body (bytes decoded to text).
            print(f"FAIL {f.name}: {exc.code} {exc.read().decode()[:200]}")
            # >> Stop on the first failure.
            return 1
        # >> Print status (left-aligned in 9 chars), chunk count (right-aligned in 3) and file name.
        print(f"{result['status']:<9} {result['chunks']:>3} chunks  {f.name}")
    # >> All files ingested: success.
    return 0


# >> True only when the file is run directly (python scripts/ingest_docs.py), not when imported.
if __name__ == "__main__":
    # >> Run main() and exit with its return code (like process.exit(main())).
    sys.exit(main())
