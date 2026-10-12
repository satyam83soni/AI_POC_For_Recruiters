"""
Structured JSON logs with a request id on every line.

In a debugging round the logs are your main evidence. Two properties make them
usable: one JSON object per line (so you can grep/jq them), and a request_id on
every line (so you can pull out everything one request did).
"""

import json
import logging
import sys
import time
from contextvars import ContextVar

# A ContextVar is per-asyncio-task, so concurrent requests never see each
# other's id -- unlike a global, which they would all overwrite.
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "request_id": request_id_var.get(),
            "msg": record.getMessage(),
        }
        # logger.info("x", extra={"fields": {...}}) -> merged into the line
        payload.update(getattr(record, "fields", {}))
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(level: str) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    # Quiet noisy libraries; our own lines are the ones that matter.
    for name in ("httpx", "httpx2", "uvicorn.access"):
        logging.getLogger(name).setLevel(logging.WARNING)


def log(logger: logging.Logger, msg: str, level: int = logging.INFO, **fields) -> None:
    logger.log(level, msg, extra={"fields": fields})
