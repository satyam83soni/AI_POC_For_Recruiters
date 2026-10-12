r"""
LangGraph -- ONLY the parts rag-poc uses, plus the one failure it's famous for.

Run (no Docker, no API key needed):   .\.venv\Scripts\python.exe learn\02_langgraph.py

Node mental model: a graph is a state machine. Like a Redux reducer with steps:
    state  = one object passed through every step
    node   = an async function: (state) -> {only the keys it changed}
    edge   = "after this node, go to that one"
    conditional edge = a router function that returns the next node's name
"""

import asyncio
from typing import TypedDict

from langgraph.config import get_stream_writer
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph


# --- 1. State: a typed dict, like a TS interface ---------------------------
class State(TypedDict, total=False):        # total=False: keys may be missing
    question: str
    docs: list[str]
    answer: str


FAKE_DB = {
    "refund": ["Refunds within 7 days.", "UPI refunds arrive in 48h."],
    "certificate": ["Score 70% to pass."],
}


# --- 2. Nodes: return ONLY what changed; LangGraph merges it ----------------
async def retrieve(state: State) -> dict:
    words = state["question"].lower().split()
    docs = [d for key, ds in FAKE_DB.items() if key in words for d in ds]
    print(f"  [retrieve] found {len(docs)} docs")
    return {"docs": docs}                   # not {**state, "docs": ...}, just the delta


async def generate(state: State) -> dict:
    write = get_stream_writer()             # pushes data to astream(stream_mode="custom")
    answer = ""
    for word in f"Based on the docs: {' '.join(state['docs'])}".split():
        write({"token": word + " "})        # no-op under ainvoke, live under astream
        answer += word + " "
    print("  [generate] done")
    return {"answer": answer.strip()}


async def not_found(state: State) -> dict:
    print("  [not_found] skipping the LLM")
    return {"answer": "I couldn't find that."}


# --- 3. Router for the conditional edge: returns a NAME --------------------
def route(state: State) -> str:
    return "generate" if state["docs"] else "not_found"


def build():
    g = StateGraph(State)
    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate)
    g.add_node("not_found", not_found)
    g.add_edge(START, "retrieve")
    g.add_conditional_edges("retrieve", route)   # route() picks the next node
    g.add_edge("generate", END)
    g.add_edge("not_found", END)
    return g.compile()                            # compile() checks the wiring


async def main() -> None:
    graph = build()

    # --- 4. ainvoke: run to the end, get the final state --------------------
    print("\n=== ainvoke: on-topic ===")
    final = await graph.ainvoke({"question": "what is the refund policy"})
    print("  final state keys:", list(final))
    print("  answer:", final["answer"])

    print("\n=== ainvoke: off-topic (conditional edge takes the other branch) ===")
    final = await graph.ainvoke({"question": "who won the world cup"})
    print("  answer:", final["answer"])

    # --- 5. astream: watch it run (this is how /ask/stream works) -----------
    # "updates" = which node just finished + what it returned
    # "custom"  = whatever nodes pushed with get_stream_writer()
    print("\n=== astream(stream_mode=['custom', 'updates']) ===")
    async for mode, chunk in graph.astream(
        {"question": "certificate rules"}, stream_mode=["custom", "updates"]
    ):
        print(f"  {mode:<8} {chunk}")

    # --- 6. The classic LangGraph incident: a loop that never ends ----------
    # Agents loop ("retry until good enough"). If the exit condition can never
    # be true, the graph spins until recursion_limit (default 25) and raises.
    print("\n=== a loop with a broken exit condition ===")

    async def improve(state: State) -> dict:
        return {"answer": state.get("answer", "") + "+"}

    def good_enough(state: State) -> str:
        return "done" if len(state["answer"]) > 100 else "again"   # needs 100 iterations

    lg = StateGraph(State)
    lg.add_node("improve", improve)
    lg.add_edge(START, "improve")
    lg.add_conditional_edges("improve", good_enough, {"again": "improve", "done": END})
    loop = lg.compile()
    try:
        await loop.ainvoke({"answer": ""}, config={"recursion_limit": 10})
    except GraphRecursionError as exc:
        print("  GraphRecursionError:", str(exc).splitlines()[0])
        print("  ^ what you'd see in logs. Fix: cap iterations in STATE, not only the limit.")


# TRY: in route(), return "generte" (typo). What error, and when does it fire?
# TRY: make retrieve() return {**state, "docs": docs}. Does anything break? Why not?

if __name__ == "__main__":
    asyncio.run(main())
