"""
Structured JSON logs with a request id on every line.

In a debugging round the logs are your main evidence. Two properties make them
usable: one JSON object per line (so you can grep/jq them), and a request_id on
every line (so you can pull out everything one request did).
"""

# >> Standard-library modules: `json` to turn dicts into JSON strings (like JSON.stringify),
# >> `logging` = Python's built-in logger system, `sys` gives access to stdout,
# >> `time` for formatting timestamps.
import json
import logging
import sys
import time
# >> ContextVar: a variable whose value is separate per async task/request
# >> (like AsyncLocalStorage in Node).
from contextvars import ContextVar

# A ContextVar is per-asyncio-task, so concurrent requests never see each
# other's id -- unlike a global, which they would all overwrite.
# >> Create the request-id holder. `ContextVar[str]` is a type hint (it holds strings);
# >> "request_id" is its debug name; if nothing was set, `.get()` returns "-".
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


# >> A custom formatter class: inherits logging.Formatter and overrides how one log line is printed.
class JsonFormatter(logging.Formatter):
    # >> `def` defines a method; `self` is the object itself (like `this`, but passed explicitly).
    # >> `record` is one log event; `-> str` says the method returns a string.
    def format(self, record: logging.LogRecord) -> str:
        # >> Build a dict (like a JS object) with the standard fields of every log line.
        payload = {
            # >> Timestamp: record.created is seconds since 1970; gmtime converts it to UTC parts,
            # >> strftime formats it as text like "2026-10-12T09:30:00".
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            # >> Level name such as "INFO" or "ERROR".
            "level": record.levelname,
            # >> Name of the logger that wrote this line (e.g. "app").
            "logger": record.name,
            # >> The current request's id, read from the ContextVar ("-" outside a request).
            "request_id": request_id_var.get(),
            # >> The final log message text (with any %-style args filled in).
            "msg": record.getMessage(),
        }
        # logger.info("x", extra={"fields": {...}}) -> merged into the line
        # >> `getattr(obj, "fields", {})` reads obj.fields, or returns {} if it doesn't exist.
        # >> `dict.update(...)` copies those keys into payload (like Object.assign(payload, fields)).
        payload.update(getattr(record, "fields", {}))
        # >> If this log call carried an exception (e.g. logger.exception), `exc_info` is truthy.
        if record.exc_info:
            # >> Add the formatted stack trace as text under the "exc" key.
            payload["exc"] = self.formatException(record.exc_info)
        # >> Return the dict as one JSON line; `default=str` turns non-JSON values (dates etc.) to text.
        return json.dumps(payload, default=str)


# >> Function called once at startup to configure logging. `-> None` = returns nothing.
def setup_logging(level: str) -> None:
    # >> A handler decides WHERE logs go; this one writes to standard output (the console).
    handler = logging.StreamHandler(sys.stdout)
    # >> Tell that handler to format each line with our JsonFormatter.
    handler.setFormatter(JsonFormatter())
    # >> getLogger() with no name returns the root logger, the parent of every logger.
    root = logging.getLogger()
    # >> `[:] = [...]` replaces the CONTENTS of the existing list in place, so the root logger
    # >> now has exactly one handler (ours), removing any defaults.
    root.handlers[:] = [handler]
    # >> Set the minimum level (e.g. "INFO"): anything less important is dropped.
    root.setLevel(level)
    # Quiet noisy libraries; our own lines are the ones that matter.
    # >> Loop over a tuple (fixed list in parentheses) of noisy library logger names...
    for name in ("httpx", "httpx2", "uvicorn.access"):
        # >> ...and only let WARNING or worse through for each of them.
        logging.getLogger(name).setLevel(logging.WARNING)


# >> Helper to log a message plus extra key/value fields. `level` defaults to INFO;
# >> `**fields` collects any extra keyword args into a dict (like `...rest` of an options object).
# >> e.g. log(logger, "hi", model="x") -> fields == {"model": "x"}.
def log(logger: logging.Logger, msg: str, level: int = logging.INFO, **fields) -> None:
    # >> Write the log line; `extra=` attaches fields to the record so JsonFormatter merges them in.
    logger.log(level, msg, extra={"fields": fields})
