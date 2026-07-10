"""Structured JSON logging for the API service."""

import json
import logging
import sys
from typing import Any

from core.config import settings


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "extra_fields", None)
        if isinstance(extra, dict):
            payload.update(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(settings.log_level.upper())
    # keep noisy libraries at WARNING
    for name in ("httpx", "httpcore", "LiteLLM", "chromadb", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)


def log_stage(logger: logging.Logger, msg: str, **fields: Any) -> None:
    """Log with structured extra fields (e.g. per-stage latency)."""
    logger.info(msg, extra={"extra_fields": fields})
