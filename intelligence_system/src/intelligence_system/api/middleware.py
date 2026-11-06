"""Middleware ASGI puro: request_id, access log JSON y red de seguridad para errores.

- Toma `X-Request-ID` del cliente si es válido o genera uno, lo guarda en
  contextvars (logs) y lo devuelve en la respuesta.
- Emite una línea de access log por request con método, ruta, status y duración.
- Si algo explota sin manejar, responde 500 en formato problem+json en vez de
  dejar que el servidor devuelva un error sin request_id.
"""

from __future__ import annotations

import time

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from intelligence_system.api.errors import problem_response
from intelligence_system.observability.context import (
    bind_request_id,
    clear_context,
    sanitize_request_id,
)
from intelligence_system.observability.logging import get_logger

log = get_logger("intelligence_system.http")
_QUIET_PATHS = frozenset({"/health"})


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = sanitize_request_id(Headers(scope=scope).get("x-request-id"))
        clear_context()
        bind_request_id(request_id)
        started = time.perf_counter()
        status_code = 500
        response_started = False

        async def send_with_id(message: Message) -> None:
            nonlocal status_code, response_started
            if message["type"] == "http.response.start":
                response_started = True
                status_code = message["status"]
                MutableHeaders(scope=message).append("X-Request-ID", request_id)
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        except Exception:
            log.exception("http.unhandled_error", path=scope.get("path"))
            if response_started:
                raise
            response = problem_response(
                status=500,
                type_uri="/errors/internal-error",
                title="Error interno",
                detail="Error inesperado. Usá el request_id para buscarlo en los logs.",
            )
            await response(scope, receive, send_with_id)
        finally:
            path = scope.get("path", "")
            log_method = log.debug if path in _QUIET_PATHS else log.info
            log_method(
                "http.request",
                method=scope.get("method"),
                path=path,
                status=status_code,
                duration_ms=round((time.perf_counter() - started) * 1000, 1),
            )
            clear_context()
