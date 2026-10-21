r"""
The RAG flow as a LangGraph state machine.

    START -> retrieve --no chunks--> not_found -> END
                 |
                 |-- answer_pipeline="stream" --> generate -> END
                 |
                 \-- answer_pipeline="verified" (default):

        extract (cheap model, JSON schema) -> validate
            validate --invalid, retries left--> extract   (told what was wrong)
            validate --invalid, out of retries--> not_found
            validate --found=false--> not_found
            validate --ok--> verify (heavy model) -> END

Why a graph: the shape is what LearnTube-style agents grow from. The
extract/validate/verify loop above was added as nodes and edges, not as a
rewrite of nested ifs.
"""

import logging
from typing import NotRequired, TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from app import llm
from app.config import settings
from app.logging import log
from app.retrieval import RetrievedChunk, retrieve

logger = logging.getLogger(__name__)


class RAGState(TypedDict):
    # Required: the caller must pass these in.
    question: str
    top_k: int
    # Filled in by nodes as the graph runs, so absent at the start.
    chunks: NotRequired[list[RetrievedChunk]]
    answer: NotRequired[str]
    # Verified pipeline only.
    context: NotRequired[str]
    n_passages: NotRequired[int]
    extraction: NotRequired[llm.ExtractedAnswer | None]
    problems: NotRequired[list[str]]
    attempts: NotRequired[int]


async def retrieve_node(state: RAGState) -> dict:
    # A node returns only the keys it changes; LangGraph merges them in.
    return {"chunks": await retrieve(state["question"], state["top_k"])}


async def generate_node(state: RAGState) -> dict:
    # The writer pushes tokens out to graph.astream(stream_mode="custom")
    # while the node is still running. Under ainvoke() it is a no-op.
    write = get_stream_writer()
    parts: list[str] = []
    async for token in llm.stream_answer(state["question"], state.get("chunks", [])):
        parts.append(token)
        write({"token": token})
    return {"answer": "".join(parts)}


async def not_found_node(state: RAGState) -> dict:
    # No LLM call at all: nothing relevant was retrieved, so there is nothing
    # to ground an answer in. Also saves the cost of a pointless request.
    get_stream_writer()({"token": llm.NOT_FOUND})
    return {"answer": llm.NOT_FOUND}


async def extract_node(state: RAGState) -> dict:
    context, n_passages = (state["context"], state["n_passages"]) if "context" in state \
        else llm.build_context(state["chunks"])
    attempts = state.get("attempts", 0)
    # On a retry, feed back what failed validation last time.
    previous = state.get("problems") if attempts else None
    extraction = await llm.extract_answer(state["question"], context, previous)
    return {"extraction": extraction, "attempts": attempts + 1,
            "context": context, "n_passages": n_passages}


async def validate_node(state: RAGState) -> dict:
    # Plain code, no LLM: cheap, deterministic, and it runs before we pay
    # for the heavy model.
    ex = state["extraction"]
    problems = (["no output matching the schema (refused or cut off)"] if ex is None
                else llm.check_extraction(ex, state["n_passages"]))
    log(logger, "extraction validated", attempt=state["attempts"],
        valid=not problems, found=ex.found if ex else None, problems=problems)
    return {"problems": problems}


async def verify_node(state: RAGState) -> dict:
    draft = state["extraction"].answer
    verdict = await llm.verify_answer(state["question"], state["context"], draft)

    if verdict is None:
        answer, outcome = llm.NOT_FOUND, "no_verdict"
    elif verdict.supported:
        answer, outcome = draft, "supported"
    elif verdict.corrected_answer.strip() and not llm.check_citations(
            verdict.corrected_answer, state["n_passages"]):
        # The heavy model's correction goes through the same citation rules
        # as the cheap model's draft -- it is not trusted blindly either.
        answer, outcome = verdict.corrected_answer, "corrected"
    else:
        answer, outcome = llm.NOT_FOUND, "rejected"

    log(logger, "answer verified", outcome=outcome,
        issues=verdict.issues if verdict else None)
    get_stream_writer()({"token": answer})
    return {"answer": answer}


def route_after_retrieve(state: RAGState) -> str:
    if not state.get("chunks"):
        return "not_found"
    return "generate" if settings.answer_pipeline == "stream" else "extract"


def route_after_validate(state: RAGState) -> str:
    if state["problems"]:
        retries_left = state["attempts"] <= settings.extract_max_retries
        return "extract" if retries_left else "not_found"
    if not state["extraction"].found:
        return "not_found"  # the cheap model says it isn't there: no need to verify
    return "verify"


def build_graph():
    g = StateGraph(RAGState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("generate", generate_node)
    g.add_node("extract", extract_node)
    g.add_node("validate", validate_node)
    g.add_node("verify", verify_node)
    g.add_node("not_found", not_found_node)
    g.add_edge(START, "retrieve")
    g.add_conditional_edges("retrieve", route_after_retrieve,
                            {"generate": "generate", "extract": "extract",
                             "not_found": "not_found"})
    g.add_edge("extract", "validate")
    g.add_conditional_edges("validate", route_after_validate,
                            {"extract": "extract", "verify": "verify",
                             "not_found": "not_found"})
    g.add_edge("generate", END)
    g.add_edge("verify", END)
    g.add_edge("not_found", END)
    return g.compile()


rag_graph = build_graph()
