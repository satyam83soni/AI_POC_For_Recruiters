"""
The Claude calls: build a grounded prompt from retrieved chunks, then either

    stream_answer   heavy model writes the answer, streamed token by token
    extract_answer  cheap model returns the answer as schema-shaped JSON
    verify_answer   heavy model judges that JSON against the same context

plus check_citations / check_extraction, the rules a JSON schema can't express.
"""

import logging
import re
import time
from collections.abc import AsyncIterator

import anthropic
from pydantic import BaseModel, Field, ValidationError

from app.config import settings
from app.logging import log
from app.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)

# One client for the whole process: it owns a connection pool. Creating one
# per request throws that pool away every time.
#
# max_retries: the SDK already retries 429 / 5xx / connection errors with
# exponential backoff. timeout: without one, a hung upstream hangs our request.
client = anthropic.AsyncAnthropic(
    api_key=settings.anthropic_api_key,
    timeout=settings.llm_timeout_s,
    max_retries=settings.llm_max_retries,
)

SYSTEM_PROMPT = """You answer questions using only the numbered context passages provided.

Cite the passages you used like [1] or [2][3]. If the context does not contain the answer, say you could not find it in the documents. Do not fill gaps from general knowledge, because a confident wrong answer about a company policy is worse than no answer."""

NOT_FOUND = "I couldn't find that in the documents."


def build_context(chunks: list[RetrievedChunk]) -> tuple[str, int]:
    """Number the chunks and stop at the character budget (cost guard).

    Returns the context and how many passages made it in: citations are only
    valid up to that number, which can be less than len(chunks).
    """
    parts: list[str] = []
    used = 0
    for i, c in enumerate(chunks, start=1):
        block = f"[{i}] (source: {c.source})\n{c.content}"
        if used + len(block) > settings.max_context_chars:
            log(logger, "context truncated by budget", logging.WARNING,
                kept=i - 1, total=len(chunks))
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts), len(parts)


async def stream_answer(question: str, chunks: list[RetrievedChunk]) -> AsyncIterator[str]:
    context, _ = build_context(chunks)
    user_message = f"<context>\n{context}\n</context>\n\nQuestion: {question}"

    start = time.perf_counter()
    first_token_at: float | None = None

    # fallbacks="default": if Claude's safety classifier declines, the API
    # re-runs the request on a fallback model server-side instead of
    # returning a refusal. Needs the beta header below.
    async with client.beta.messages.stream(
        model=settings.llm_model,
        max_tokens=settings.llm_max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
        output_config={"effort": settings.llm_effort},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    ) as stream:
        async for text in stream.text_stream:
            if first_token_at is None:
                first_token_at = time.perf_counter()
            yield text
        final = await stream.get_final_message()

    if final.stop_reason == "refusal":
        yield NOT_FOUND
    elif final.stop_reason == "max_tokens":
        log(logger, "answer truncated at max_tokens", logging.WARNING,
            max_tokens=settings.llm_max_tokens)

    log(logger, "llm done",
        model=final.model, stop_reason=final.stop_reason,
        input_tokens=final.usage.input_tokens, output_tokens=final.usage.output_tokens,
        ttft_ms=round((first_token_at - start) * 1000) if first_token_at else None,
        total_ms=round((time.perf_counter() - start) * 1000),
        # final.id (msg_...) is on every response, streamed or not. The
        # private _request_id is not set on beta stream results, and reading
        # it crashed the request AFTER Claude had already answered.
        message_id=final.id)


# --- verified pipeline: cheap model extracts, heavy model verifies ----------
#
# Two layers of checking, on purpose:
#   1. Structured outputs (output_format=...) make the API return JSON that
#      matches the schema -- the SHAPE is guaranteed.
#   2. check_extraction enforces what a schema can't say: cited passage
#      numbers exist, and the [n] in the text match the citations list.
# Then the heavy model checks the MEANING: is every claim in the cited text?

class ExtractedAnswer(BaseModel):
    found: bool = Field(description="False if the passages do not contain the answer.")
    answer: str = Field(description="Concise answer with inline citations like [1] or [2][3]. "
                                    "Empty when found is false.")
    citations: list[int] = Field(description="Every passage number cited in answer.")


class Verdict(BaseModel):
    supported: bool = Field(description="True only if every claim in the draft is stated "
                                        "in the passages it cites.")
    issues: list[str] = Field(description="Each unsupported or wrong claim. Empty if supported.")
    corrected_answer: str = Field(description="If unsupported and the passages DO contain the "
                                              "answer: the correct answer with inline [n] "
                                              "citations. Otherwise empty.")


EXTRACT_SYSTEM = """Extract the answer to the question from the numbered context passages only.

If the passages do not contain the answer, set found to false and leave answer and citations empty. Otherwise write a concise answer, cite the passages you used inline like [1] or [2][3], and list every cited number in citations. Use nothing outside the passages: a confident wrong answer about a company policy is worse than no answer."""

VERIFY_SYSTEM = """You check a draft answer against numbered context passages, using nothing outside them.

The draft is supported only if every claim in it is stated in the passages it cites. If it is not, list each problem in issues; then, if the passages do contain the correct answer, write it in corrected_answer with inline [n] citations, otherwise leave corrected_answer empty. If the draft is supported, leave issues and corrected_answer empty."""

_CITATION = re.compile(r"\[(\d+)\]")


def check_citations(text: str, n_passages: int) -> list[str]:
    """Problems with the [n] markers in an answer text; empty = valid."""
    cited = {int(n) for n in _CITATION.findall(text)}
    problems = []
    if not cited:
        problems.append("answer cites no passage")
    bad = sorted(n for n in cited if not 1 <= n <= n_passages)
    if bad:
        problems.append(f"cites passages {bad} but only 1-{n_passages} exist")
    return problems


def check_extraction(ex: ExtractedAnswer, n_passages: int) -> list[str]:
    """Rules the JSON schema can't express. Empty list = valid."""
    if not ex.found:
        return []  # "not in the documents" is a valid, checkable outcome
    if not ex.answer.strip():
        return ["found is true but answer is empty"]
    problems = check_citations(ex.answer, n_passages)
    in_text = {int(n) for n in _CITATION.findall(ex.answer)}
    if in_text != set(ex.citations):
        problems.append(f"citations list {sorted(set(ex.citations))} does not match "
                        f"the [n] in the answer {sorted(in_text)}")
    return problems


def _log_call(step: str, msg, start: float) -> None:
    log(logger, "llm done", step=step,
        model=msg.model, stop_reason=msg.stop_reason,
        input_tokens=msg.usage.input_tokens, output_tokens=msg.usage.output_tokens,
        total_ms=round((time.perf_counter() - start) * 1000),
        message_id=msg.id)


def _unparseable(step: str, exc: ValidationError, start: float) -> None:
    # parse() validates the JSON inside the call, so a refusal (plain text)
    # or a reply cut off at max_tokens (half a JSON object) raises HERE --
    # there is no message left to check stop_reason on. Treat it as "no
    # answer to trust" rather than letting it become a 500.
    log(logger, "llm output failed schema", logging.WARNING, step=step,
        errors=exc.error_count(),
        total_ms=round((time.perf_counter() - start) * 1000))
    return None


async def extract_answer(question: str, context: str,
                         previous_problems: list[str] | None = None) -> ExtractedAnswer | None:
    """Cheap model -> ExtractedAnswer, or None if there's nothing parseable."""
    user_message = f"<context>\n{context}\n</context>\n\nQuestion: {question}"
    if previous_problems:
        # A retry: say exactly what failed, so the model fixes THAT.
        user_message += ("\n\nYour previous answer was rejected:\n- "
                         + "\n- ".join(previous_problems))

    start = time.perf_counter()
    # Not streamed: nothing reads this JSON until it is complete.
    # No fallbacks param: server-side fallbacks aren't offered for Haiku.
    try:
        msg = await client.messages.parse(
            model=settings.extract_model,
            max_tokens=settings.extract_max_tokens,
            system=EXTRACT_SYSTEM,
            messages=[{"role": "user", "content": user_message}],
            output_config={"effort": settings.extract_effort},
            output_format=ExtractedAnswer,
        )
    except ValidationError as exc:
        return _unparseable("extract", exc, start)
    _log_call("extract", msg, start)
    return msg.parsed_output


async def verify_answer(question: str, context: str, draft: str) -> Verdict | None:
    """Heavy model judges the draft. None = no verdict (refused / cut off)."""
    user_message = (f"<context>\n{context}\n</context>\n\nQuestion: {question}"
                    f"\n\n<draft_answer>\n{draft}\n</draft_answer>")
    start = time.perf_counter()
    try:
        msg = await client.beta.messages.parse(
            model=settings.llm_model,
            max_tokens=settings.llm_max_tokens,
            system=VERIFY_SYSTEM,
            messages=[{"role": "user", "content": user_message}],
            output_config={"effort": settings.llm_effort},
            output_format=Verdict,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
    except ValidationError as exc:
        return _unparseable("verify", exc, start)
    _log_call("verify", msg, start)
    return msg.parsed_output
