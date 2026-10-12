"""
The RAG flow as a LangGraph state machine.

    START -> retrieve -> (any relevant chunks?) --yes--> generate -> END
                                               \--no---> not_found -> END

Why a graph for two steps: the shape is what LearnTube-style agents grow from.
Adding "grade the chunks", "rewrite the query and retry" or a tool call is one
more node and one more edge, not a rewrite of nested ifs.
"""

from typing import TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from app import llm
from app.retrieval import RetrievedChunk, retrieve


class RAGState(TypedDict, total=False):
    question: str
    top_k: int
    chunks: list[RetrievedChunk]
    answer: str


async def retrieve_node(state: RAGState) -> dict:
    # A node returns only the keys it changes; LangGraph merges them in.
    return {"chunks": await retrieve(state["question"], state["top_k"])}


async def generate_node(state: RAGState) -> dict:
    # The writer pushes tokens out to graph.astream(stream_mode="custom")
    # while the node is still running. Under ainvoke() it is a no-op.
    write = get_stream_writer()
    parts: list[str] = []
    async for token in llm.stream_answer(state["question"], state["chunks"]):
        parts.append(token)
        write({"token": token})
    return {"answer": "".join(parts)}


async def not_found_node(state: RAGState) -> dict:
    # No LLM call at all: nothing relevant was retrieved, so there is nothing
    # to ground an answer in. Also saves the cost of a pointless request.
    get_stream_writer()({"token": llm.NOT_FOUND})
    return {"answer": llm.NOT_FOUND}


def route_after_retrieve(state: RAGState) -> str:
    return "generate" if state["chunks"] else "not_found"


def build_graph():
    g = StateGraph(RAGState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("generate", generate_node)
    g.add_node("not_found", not_found_node)
    g.add_edge(START, "retrieve")
    g.add_conditional_edges("retrieve", route_after_retrieve,
                            {"generate": "generate", "not_found": "not_found"})
    g.add_edge("generate", END)
    g.add_edge("not_found", END)
    return g.compile()


rag_graph = build_graph()
