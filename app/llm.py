"""
The Claude call: build a grounded prompt from retrieved chunks, stream tokens.
"""

import logging
import time
from collections.abc import AsyncIterator

import anthropic

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


def build_context(chunks: list[RetrievedChunk]) -> str:
    """Number the chunks and stop at the character budget (cost guard)."""
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
    return "\n\n".join(parts)


async def stream_answer(question: str, chunks: list[RetrievedChunk]) -> AsyncIterator[str]:
    context = build_context(chunks)
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
