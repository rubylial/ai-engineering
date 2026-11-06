"""Logging estructurado en JSON con structlog.

Cada línea de log lleva `request_id`, `thread_id`, `trace_id` y `span_id`
cuando existen, así un log se cruza con su traza de Phoenix por el mismo id.
Los logs de librerías (uvicorn, httpx, langchain) pasan por el mismo
formateador, por eso todo el proceso emite JSON homogéneo.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from opentelemetry import trace
from structlog.typing import EventDict, WrappedLogger

_NOISY_LOGGERS = ("httpx", "httpcore", "anthropic", "openai", "urllib3", "pinecone")


def add_trace_context(_: WrappedLogger, __: str, event_dict: EventDict) -> EventDict:
    """Agrega trace_id/span_id de OpenTelemetry si hay un span activo."""
    span_context = trace.get_current_span().get_span_context()
    if span_context.is_valid:
        event_dict.setdefault("trace_id", format(span_context.trace_id, "032x"))
        event_dict.setdefault("span_id", format(span_context.span_id, "016x"))
    return event_dict


def configure_logging(level: str = "INFO", json_logs: bool = True) -> None:
    """Configura structlog + logging estándar. Idempotente."""
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        add_trace_context,
        structlog.processors.StackInfoRenderer(),
    ]
    structlog.configure(
        processors=[*shared_processors, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    renderer: Any = (
        structlog.processors.JSONRenderer(ensure_ascii=False)
        if json_logs
        else structlog.dev.ConsoleRenderer()
    )
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.format_exc_info,
            renderer,
        ],
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
    # El access log lo emite nuestro middleware (con request_id); el de uvicorn sobra.
    logging.getLogger("uvicorn.access").handlers = []
    logging.getLogger("uvicorn.access").propagate = False
    for name in ("uvicorn", "uvicorn.error"):
        logging.getLogger(name).handlers = []
        logging.getLogger(name).propagate = True
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.stdlib.get_logger(name)
