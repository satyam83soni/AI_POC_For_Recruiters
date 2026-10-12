r"""
LangChain text splitter -- the only LangChain piece rag-poc uses, and why
chunk size quietly decides whether RAG answers correctly.

Run (no Docker, no API key):   .\.venv\Scripts\python.exe learn\03_chunking.py
"""

import pathlib

from langchain_text_splitters import RecursiveCharacterTextSplitter

DOC = (pathlib.Path(__file__).resolve().parent.parent
       / "data" / "docs" / "refund-policy.md").read_text(encoding="utf-8")


SMART = ["\n## ", "\n\n", "\n", ". ", " ", ""]


def show(size: int, overlap: int, separators: list[str] = SMART) -> None:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,             # max CHARACTERS per chunk (not tokens, not words)
        chunk_overlap=overlap,       # chars repeated from the previous chunk's tail
        # Try to split at the first separator; if a piece is still too big,
        # fall back to the next one. So: sections, then paragraphs, then lines,
        # then sentences, then words, then characters.
        separators=separators,
    )
    chunks = splitter.split_text(DOC)
    print(f"\n=== chunk_size={size}, overlap={overlap} -> {len(chunks)} chunks ===")
    for i, c in enumerate(chunks):
        flat = c.replace("\n", " | ")
        print(f"  [{i}] {len(c):>4} chars  {flat[:90]}{'...' if len(flat) > 90 else ''}")


print(f"document is {len(DOC)} characters")

show(800, 120)    # rag-poc's setting: a few big chunks, each a full section or two
show(200, 0)      # tiny: headings get orphaned, facts get separated
show(5000, 0)     # one giant chunk: retrieval can't tell sections apart

# Overlap only matters when a split lands MID-TEXT. With the smart separators
# above, splits land on paragraph breaks, so overlap has nothing to repeat.
# Splitting on spaces only forces mid-sentence cuts -- now compare:
show(150, 0, separators=[" "])     # sentences cut in half, nothing repeated
show(150, 40, separators=[" "])    # each chunk starts with the previous one's tail

print("""
What to notice:
  - 200/0: "## Eligibility" becomes its own 14-character chunk. A chunk with
    no facts in it can still be retrieved, and it wastes a top_k slot.
    Chunk [3] starts with ". After 7 days" -- cut off from what it qualifies.
  - 5000: every question retrieves the same single chunk. Similarity scores
    get blurry, and you send the whole document to the LLM every time (cost).
  - Overlap is insurance for facts that sit on a boundary.

Interview line: "chunk size is a retrieval-quality knob, not a performance
knob. Too small loses context, too big loses precision and costs tokens."

TRY: in the last two runs, find a sentence cut in half at 150/0 and see it
     whole at 150/40.
""")
