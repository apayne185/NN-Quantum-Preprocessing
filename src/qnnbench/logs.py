"""Logging setup: JSON lines for machines (containers, log shippers), plain text for terminals.

    configure_logging()                     # QNNBENCH_LOG_FORMAT=json|text, QNNBENCH_LOG_LEVEL
    log.info("epoch done", extra={"epoch": 3, "test_acc": 0.91})

Fields passed through ``extra`` become top-level keys in JSON output, so
dashboards can filter on them without parsing message strings.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone

_RESERVED = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname.lower(),
            "logger": record.name,
            "msg": record.getMessage(),
        }
        payload.update({k: v for k, v in record.__dict__.items() if k not in _RESERVED})
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        extras = {k: v for k, v in record.__dict__.items() if k not in _RESERVED}
        line = f"{record.levelname[0]} {record.name}: {record.getMessage()}"
        if extras:
            line += "  " + " ".join(f"{k}={_short(v)}" for k, v in extras.items())
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


def _short(v) -> str:
    return f"{v:.4g}" if isinstance(v, float) else str(v)


def configure_logging(fmt: str | None = None, level: str | None = None) -> None:
    """Idempotent root-logger setup for the qnnbench CLIs and the server."""
    fmt = fmt or os.environ.get("QNNBENCH_LOG_FORMAT", "text")
    if fmt not in ("json", "text"):
        raise ValueError(f"QNNBENCH_LOG_FORMAT must be 'json' or 'text', got {fmt!r}")
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter() if fmt == "json" else TextFormatter())
    logger = logging.getLogger("qnnbench")
    logger.handlers[:] = [handler]
    logger.setLevel(level or os.environ.get("QNNBENCH_LOG_LEVEL", "INFO"))
    logger.propagate = False
