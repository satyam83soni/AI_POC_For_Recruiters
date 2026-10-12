r"""
Streaming in rag-poc, layer by layer, with a FAKE LLM so it runs with no keys.

Run:   .\.venv\Scripts\python.exe learn\04_streaming.py

The real chain in rag-poc, innermost first:

  1. Claude's API sends tokens over HTTP           (llm.py: stream.text_stream)
  2. stream_answer() re-yields each token          (llm.py: async generator)
  3. generate_node pushes each token to LangGraph  (graph.py: get_stream_writer)
  4. graph.astream() hands tokens to the route     (main.py: async for mode, chunk)
  5. events() formats each one as SSE text         (main.py: _sse)
  6. StreamingResponse writes it to the socket     (main.py: StreamingResponse)
  7. The browser/curl prints it as it arrives

Every layer is "an async loop over the layer below it". Nothing waits for the
whole answer -- each token passes through all 7 layers before the next exists.
"""

import asyncio
import json
import time
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import threading

import httpx
import uvicorn
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from typing import TypedDict

T0 = time.perf_counter()


def t() -> str:
    return f"{time.perf_counter() - T0:5.2f}s"


# --- LAYER 1+2: a fake Claude that produces one token every 0.3 s ----------
# An ASYNC GENERATOR: `async def` + `yield`. Each `yield` hands one value to
# whoever is looping over it, then PAUSES here until they ask for the next.
# JS equivalent: `async function* fakeClaude() { yield "The"; ... }`
async def fake_claude(question: str) -> AsyncIterator[str]:
    for token in ["Refunds ", "are ", "allowed ", "within ", "7 ", "days ", "[1]."]:
        await asyncio.sleep(0.3)          # pretend the network is slow
        print(f"  {t()}  [1 llm]     produced {token!r}")
        yield token


# --- LAYER 3: a LangGraph node that forwards tokens as it gets them --------
class State(TypedDict, total=False):
    question: str
    answer: str


async def generate(state: State) -> dict:
    write = get_stream_writer()           # a function: write(x) sends x out of astream()
    parts = []
    async for token in fake_claude(state["question"]):
        parts.append(token)
        print(f"  {t()}  [3 node]    write({token!r})")
        write({"token": token})           # leaves the graph IMMEDIATELY, node keeps running
    return {"answer": "".join(parts)}     # the full answer, at the very end


g = StateGraph(State)
g.add_node("generate", generate)
g.add_edge(START, "generate")
g.add_edge("generate", END)
graph = g.compile()


# --- LAYER 5: Server-Sent Events format ------------------------------------
# SSE is just TEXT over a normal HTTP response that is never "finished" until
# the server stops writing. Each event is:
#     event: <name>\n
#     data: <json>\n
#     \n                 <- a blank line ends one event
def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# --- LAYER 4+5+6: the FastAPI route ----------------------------------------
app = FastAPI()


@app.get("/stream")
async def stream():
    async def events():                   # another async generator
        async for mode, chunk in graph.astream({"question": "refund?"},
                                               stream_mode=["custom", "updates"]):
            if mode == "custom":          # something a node passed to write()
                print(f"  {t()}  [4 route]   got custom {chunk}")
                yield sse("token", chunk)
            elif mode == "updates":       # a node FINISHED and returned its dict
                print(f"  {t()}  [4 route]   node finished: {list(chunk)}")
        yield sse("done", {})

    # StreamingResponse loops over events() and writes each yielded string to
    # the socket as soon as it is yielded. The handler itself returns at once.
    return StreamingResponse(events(), media_type="text/event-stream")


# --- LAYER 7: a client reading the stream as it arrives --------------------
# A REAL server on a real socket. (FastAPI's TestClient buffers the whole
# response in memory, which would hide exactly what this demo is showing.)
def main() -> None:
    server = uvicorn.Server(uvicorn.Config(app, port=8765, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.05)

    global T0
    T0 = time.perf_counter()
    print("Watch the timestamps: each token goes through every layer, including")
    print("the client, ~0.3s apart -- long before the full answer exists.\n")
    # httpx .stream() = fetch() + response.body.getReader() in JS: read as it arrives.
    with httpx.stream("GET", "http://127.0.0.1:8765/stream", timeout=10) as resp:
        for line in resp.iter_lines():
            if line:
                print(f"  {t()}  [7 client]  {line}")
    server.should_exit = True
    print("\nWhat the client received is plain text: `event:` / `data:` / blank line.")


# TRY: in fake_claude, raise ValueError("boom") after the 3rd token. What does
#      the client see? (rag-poc catches this and sends `event: error` instead.)
# TRY: replace `async for token in fake_claude(...)` with collecting all tokens
#      first into a list, then writing them. Watch the timestamps change.

if __name__ == "__main__":
    main()
