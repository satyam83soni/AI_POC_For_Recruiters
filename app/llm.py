"""
The Claude call: build a grounded prompt from retrieved chunks, stream tokens.
"""

# >> Python's built-in logging library (like a built-in winston/pino); used for the WARNING level.
import logging
# >> Built-in time module; we use time.perf_counter() as a high-precision stopwatch.
import time
# >> AsyncIterator = type hint for "something you can `async for` over" (our token stream).
from collections.abc import AsyncIterator

# >> The official Anthropic SDK (the Python version of @anthropic-ai/sdk) for calling Claude.
import anthropic

# >> Our app settings object (API keys, model name, limits) loaded from env / .env.
from app.config import settings
# >> Our own helper that writes a structured log line with extra key=value fields.
from app.logging import log
# >> The type of one retrieved search hit (has .source, .content, .similarity).
from app.retrieval import RetrievedChunk

# >> Create a logger named after this module ("app.llm"); __name__ = the module's import name.
logger = logging.getLogger(__name__)

# One client for the whole process: it owns a connection pool. Creating one
# per request throws that pool away every time.
#
# max_retries: the SDK already retries 429 / 5xx / connection errors with
# exponential backoff. timeout: without one, a hung upstream hangs our request.
# >> Create the async Claude client once, at import time (async = its methods must be awaited).
client = anthropic.AsyncAnthropic(
    # >> The secret API key used to authenticate with Anthropic.
    api_key=settings.anthropic_api_key,
    # >> Give up on a request after this many seconds.
    timeout=settings.llm_timeout_s,
    # >> How many times the SDK retries automatically on rate limits / server errors.
    max_retries=settings.llm_max_retries,
# >> End of the constructor call.
)

# >> The system prompt (Claude's standing instructions). Triple quotes = a multi-line string,
# >> like a JS template literal without ${} interpolation.
SYSTEM_PROMPT = """You answer questions using only the numbered context passages provided.

Cite the passages you used like [1] or [2][3]. If the context does not contain the answer, say you could not find it in the documents. Do not fill gaps from general knowledge, because a confident wrong answer about a company policy is worse than no answer."""

# >> Fixed reply used when nothing relevant was found (UPPER_CASE = a constant by convention).
NOT_FOUND = "I couldn't find that in the documents."


# >> Turns the list of chunks into one numbered text block to paste into the prompt.
def build_context(chunks: list[RetrievedChunk]) -> str:
    """Number the chunks and stop at the character budget (cost guard)."""
    # >> Collects each formatted chunk.
    parts: list[str] = []
    # >> Running total of characters used so far.
    used = 0
    # >> enumerate(..., start=1) gives (number, item) pairs counting from 1, like
    # >> chunks.forEach((c, idx) => ...) with i = idx + 1.
    for i, c in enumerate(chunks, start=1):
        # >> f-string = template literal: f"[{i}]" is like `[${i}]`. \n is a newline.
        block = f"[{i}] (source: {c.source})\n{c.content}"
        # >> If adding this block would exceed the character budget...
        if used + len(block) > settings.max_context_chars:
            # >> ...log a warning saying how many chunks we kept out of the total...
            log(logger, "context truncated by budget", logging.WARNING,
                # >> (extra fields attached to the log line)
                kept=i - 1, total=len(chunks))
            # >> ...and stop the loop early.
            break
        # >> Keep this block.
        parts.append(block)
        # >> Add its length to the running total (len() = .length in JS).
        used += len(block)
    # >> Join the blocks with a blank line between them and return one big string.
    return "\n\n".join(parts)


# >> An async generator: because the body uses `yield`, calling it returns a stream of strings
# >> you consume with `async for` (like an `async function*` in JS).
async def stream_answer(question: str, chunks: list[RetrievedChunk]) -> AsyncIterator[str]:
    # >> Build the numbered context text from the chunks.
    context = build_context(chunks)
    # >> The user message: context wrapped in <context> tags, then the question.
    user_message = f"<context>\n{context}\n</context>\n\nQuestion: {question}"

    # >> Start the stopwatch (seconds as a float, only useful for measuring differences).
    start = time.perf_counter()
    # >> When the first token arrived; `float | None` means "a number or None (JS null)".
    first_token_at: float | None = None

    # fallbacks="default": if Claude's safety classifier declines, the API
    # re-runs the request on a fallback model server-side instead of
    # returning a refusal. Needs the beta header below.
    # >> `async with` opens a streaming request and closes it automatically at the end of the
    # >> block, even on error (like try/finally). client.beta.* = API calls using beta features.
    async with client.beta.messages.stream(
        # >> Which Claude model to use.
        model=settings.llm_model,
        # >> Hard cap on how many tokens Claude may write in its answer.
        max_tokens=settings.llm_max_tokens,
        # >> The system prompt defined above.
        system=SYSTEM_PROMPT,
        # >> The conversation: one user turn containing context + question.
        messages=[{"role": "user", "content": user_message}],
        # >> output_config.effort: how much effort/thinking Claude spends (e.g. low/medium/high).
        output_config={"effort": settings.llm_effort},
        # >> Opt in to the beta feature that enables server-side fallback.
        betas=["server-side-fallback-2026-07-01"],
        # >> Use the default fallback model if the main model refuses.
        fallbacks="default",
    # >> `as stream` names the open stream object so we can read from it inside the block.
    ) as stream:
        # >> stream.text_stream yields just the text pieces as they arrive; `async for` = `for await`.
        async for text in stream.text_stream:
            # >> Only true for the very first token...
            if first_token_at is None:
                # >> ...record when it arrived (to measure time-to-first-token).
                first_token_at = time.perf_counter()
            # >> Hand this text piece to whoever is looping over stream_answer(), then continue.
            yield text
        # >> After the stream ends, get the full final message (stop_reason, usage, model, ...).
        final = await stream.get_final_message()

    # >> stop_reason says why Claude stopped; "refusal" = it declined to answer.
    if final.stop_reason == "refusal":
        # >> Send our "not found" text instead of leaving the answer empty.
        yield NOT_FOUND
    # >> "max_tokens" = the answer was cut off because it hit the max_tokens cap.
    elif final.stop_reason == "max_tokens":
        # >> Log a warning that the answer was truncated...
        log(logger, "answer truncated at max_tokens", logging.WARNING,
            # >> ...including the limit we used.
            max_tokens=settings.llm_max_tokens)

    # >> Log one summary line about the call (keyword args become log fields):
    log(logger, "llm done",
        # >> which model actually answered, and why it stopped;
        model=final.model, stop_reason=final.stop_reason,
        # >> usage = token counts (what you are billed for): prompt tokens in, answer tokens out;
        input_tokens=final.usage.input_tokens, output_tokens=final.usage.output_tokens,
        # >> time to first token in ms (None if no token ever arrived);
        ttft_ms=round((first_token_at - start) * 1000) if first_token_at else None,
        # >> total time in ms (round() rounds to the nearest whole number);
        total_ms=round((time.perf_counter() - start) * 1000),
        # >> Anthropic's request ID, handy when asking their support about a specific call.
        request_id=final._request_id)
