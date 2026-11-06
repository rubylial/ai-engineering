"""Identificadores de correlación por request (request_id / thread_id).

Se guardan en `ContextVar` para que cualquier capa (middleware, handlers de
error, nodos del grafo) los lea sin pasarlos a mano. asyncio copia el contexto
al crear tareas, así que las tareas hijas (streaming SSE, nodos de LangGraph)
heredan los mismos ids.
"""

from __future__ import annotations

import re
import uuid
from contextvars import ContextVar

import structlog

_REQUEST_ID: ContextVar[str | None] = ContextVar("request_id", default=None)
_THREAD_ID: ContextVar[str | None] = ContextVar("thread_id", default=None)

# Un X-Request-ID externo solo se acepta si es corto y seguro para logs.
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


def new_request_id() -> str:
    return f"req_{uuid.uuid4().hex}"


def new_thread_id() -> str:
    return f"thr_{uuid.uuid4().hex[:20]}"


def sanitize_request_id(candidate: str | None) -> str:
    """Reutiliza el id del cliente si es válido; si no, genera uno nuevo."""
    if candidate and _REQUEST_ID_RE.match(candidate):
        return candidate
    return new_request_id()


def get_request_id() -> str | None:
    return _REQUEST_ID.get()


def get_thread_id() -> str | None:
    return _THREAD_ID.get()


def bind_request_id(request_id: str) -> None:
    _REQUEST_ID.set(request_id)
    structlog.contextvars.bind_contextvars(request_id=request_id)


def bind_thread_id(thread_id: str) -> None:
    _THREAD_ID.set(thread_id)
    structlog.contextvars.bind_contextvars(thread_id=thread_id)


def clear_context() -> None:
    _REQUEST_ID.set(None)
    _THREAD_ID.set(None)
    structlog.contextvars.clear_contextvars()
