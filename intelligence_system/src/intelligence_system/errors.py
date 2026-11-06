"""Errores de dominio con su representación HTTP (RFC 7807 simplificado).

La API los traduce a `{type, title, status, detail, request_id, trace_id}`.
Los servicios lanzan estos errores; nunca construyen respuestas HTTP.
"""

from __future__ import annotations


class AppError(Exception):
    status_code = 500
    slug = "internal-error"
    title = "Error interno"

    def __init__(self, detail: str, *, trace_id: str | None = None) -> None:
        super().__init__(detail)
        self.detail = detail
        # Traza del trabajo que falló (el span ya cerró cuando se arma la respuesta).
        self.trace_id = trace_id

    @property
    def type_uri(self) -> str:
        return f"/errors/{self.slug}"


class InvalidRequestError(AppError):
    status_code = 422
    slug = "validation-error"
    title = "La solicitud no es válida"


class AuthError(AppError):
    status_code = 401
    slug = "unauthorized"
    title = "No autorizado"


class NotFoundError(AppError):
    status_code = 404
    slug = "not-found"
    title = "Recurso no encontrado"


class NotReadyError(AppError):
    status_code = 503
    slug = "not-ready"
    title = "El servicio no está listo"


class UpstreamError(AppError):
    status_code = 502
    slug = "upstream-error"
    title = "Falló un proveedor externo"


class GraphTimeoutError(AppError):
    status_code = 504
    slug = "graph-timeout"
    title = "Se excedió el tiempo de procesamiento"


class AgentLoopError(AppError):
    status_code = 500
    slug = "agent-loop"
    title = "El agente superó el límite de pasos"
