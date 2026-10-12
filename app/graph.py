r"""
The RAG flow as a LangGraph state machine.

    START -> retrieve -> (any relevant chunks?) --yes--> generate -> END
                                               \--no---> not_found -> END

Why a graph for two steps: the shape is what LearnTube-style agents grow from.
Adding "grade the chunks", "rewrite the query and retry" or a tool call is one
more node and one more edge, not a rewrite of nested ifs.
"""

# >> `typing` = Python's type-hint toolkit. TypedDict describes a dict with fixed keys (like a TS
# >> interface for a plain object); NotRequired marks a key as optional (like `key?:` in TS).
from typing import NotRequired, TypedDict

# >> LangGraph helper: inside a running node, returns a function that pushes custom data out to
# >> whoever is streaming the graph (used here to stream LLM tokens live).
from langgraph.config import get_stream_writer
# >> StateGraph = the graph builder. START / END = special built-in node names marking where the
# >> graph begins and where it stops.
from langgraph.graph import END, START, StateGraph

# >> Import our own module app/llm.py as a namespace object (use it as llm.stream_answer, etc.).
from app import llm
# >> RetrievedChunk = the type of one search hit; retrieve = async function that searches the DB.
from app.retrieval import RetrievedChunk, retrieve


# >> Define the graph's shared "state" shape. Every node reads this dict and returns updates to it.
class RAGState(TypedDict):
    # Required: the caller must pass these in.
    # >> The user's question text (str = string).
    question: str
    # >> How many chunks to fetch from the vector search (int = integer).
    top_k: int
    # Filled in by nodes as the graph runs, so absent at the start.
    # >> Optional key: a list of RetrievedChunk objects, filled in by the retrieve node.
    chunks: NotRequired[list[RetrievedChunk]]
    # >> Optional key: the final answer text, filled in by generate or not_found.
    answer: NotRequired[str]


# >> Node 1. `async def` = an async function (like `async function` in JS); it gets the current
# >> state and returns a dict of the keys it wants to change ("-> dict" is just a type hint).
async def retrieve_node(state: RAGState) -> dict:
    # A node returns only the keys it changes; LangGraph merges them in.
    # >> Run the vector search (await = wait for the async call), store the hits under "chunks".
    return {"chunks": await retrieve(state["question"], state["top_k"])}


# >> Node 2: ask Claude, stream tokens out as they arrive, and save the full answer in state.
async def generate_node(state: RAGState) -> dict:
    # The writer pushes tokens out to graph.astream(stream_mode="custom")
    # while the node is still running. Under ainvoke() it is a no-op.
    # >> Get the stream-writer function for this run; calling write(x) emits x to the stream.
    write = get_stream_writer()
    # >> Empty list that will collect every token (`: list[str]` is just a type hint).
    parts: list[str] = []
    # >> `async for` = loop over an async generator, like `for await (const token of ...)` in JS.
    # >> state.get("chunks", []) reads the key, or uses [] if it's missing (no KeyError).
    async for token in llm.stream_answer(state["question"], state.get("chunks", [])):
        # >> Remember this token so we can build the full answer later (append = JS push).
        parts.append(token)
        # >> Emit the token right now to anyone streaming the graph (e.g. the SSE endpoint).
        write({"token": token})
    # >> "".join(parts) glues all tokens into one string (like parts.join("") in JS).
    return {"answer": "".join(parts)}


# >> Node 3: used when retrieval found nothing; answers "not found" without calling the LLM.
async def not_found_node(state: RAGState) -> dict:
    # No LLM call at all: nothing relevant was retrieved, so there is nothing
    # to ground an answer in. Also saves the cost of a pointless request.
    # >> Get the writer and immediately call it, streaming the fixed "not found" message.
    get_stream_writer()({"token": llm.NOT_FOUND})
    # >> Save the same message as the final answer in state.
    return {"answer": llm.NOT_FOUND}


# >> Router function: decides which node runs after "retrieve". Returns the next node's name.
def route_after_retrieve(state: RAGState) -> str:
    # >> Python's inline if (like JS `cond ? a : b`): an empty list or missing key counts as false.
    return "generate" if state.get("chunks") else "not_found"


# >> Builds and compiles the graph once; returns a runnable graph object.
def build_graph():
    # >> Create a graph builder whose shared state has the RAGState shape.
    g = StateGraph(RAGState)
    # >> add_node(name, function): register each step under a name we can point edges at.
    g.add_node("retrieve", retrieve_node)
    # >> Register the "generate" step.
    g.add_node("generate", generate_node)
    # >> Register the "not_found" step.
    g.add_node("not_found", not_found_node)
    # >> add_edge(a, b): after a finishes, run b. The graph always starts at "retrieve".
    g.add_edge(START, "retrieve")
    # >> add_conditional_edges: after "retrieve", call route_after_retrieve(state) and use its
    # >> return value to pick the next node...
    g.add_conditional_edges("retrieve", route_after_retrieve,
                            # >> ...via this map of {router return value: node name}.
                            {"generate": "generate", "not_found": "not_found"})
    # >> After "generate" finishes, the graph ends.
    g.add_edge("generate", END)
    # >> After "not_found" finishes, the graph ends too.
    g.add_edge("not_found", END)
    # >> compile() checks the wiring and turns the builder into a runnable graph
    # >> (it gets .ainvoke() for one result and .astream() for streaming).
    return g.compile()


# >> Build the graph once when this module is imported; the rest of the app reuses this object.
rag_graph = build_graph()
