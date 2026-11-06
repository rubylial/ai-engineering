"""Respuestas de error uniformes `{type, title, status, detail, request_id, trace_id}`."""

from __future__ import annotations

from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from intelligence_system.errors import AppError, InvalidRequestError
from intelligence_system.observability.context import get_request_id
from intelligence_system.observability.logging import get_logger
from intelligence_system.observability.tracing import current_trace_id

log = get_logger(__name__)
PROBLEM_JSON = "application/problem+json"


def problem_response(
    *,
    status: int,
    type_uri: str,
    title: str,
    detail: str,
    errors: list[dict[str, str]] | None = None,
    headers: dict[str, str] | None = None,
    trace_id: str | None = None,
) -> JSONResponse:
    body: dict[str, Any] = {
        "type": type_uri,
        "title": title,
        "status": status,
        "detail": detail,
        "request_id": get_request_id(),
        "trace_id": trace_id or current_trace_id(),
    }
    if errors:
        body["errors"] = errors
    return JSONResponse(body, status_code=status, media_type=PROBLEM_JSON, headers=headers)


def _validation_errors(exc: RequestValidationError) -> list[dict[str, str]]:
    return [
        {"loc": ".".join(str(p) for p in err.get("loc", ())), "msg": str(err.get("msg", ""))}
        for err in exc.errors()
    ]


async def _app_error(_: Request, exc: AppError) -> JSONResponse:
    log.warning("http.app_error", error=exc.slug, status=exc.status_code, detail=exc.detail[:300])
    return problem_response(
        status=exc.status_code,
        type_uri=exc.type_uri,
        title=exc.title,
        detail=exc.detail,
        trace_id=exc.trace_id,
    )


async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    errors = _validation_errors(exc)
    error = InvalidRequestError("; ".join(f"{e['loc']}: {e['msg']}" for e in errors)[:500])
    log.info("http.validation_error", errors=errors[:5])
    return problem_response(
        status=error.status_code,
        type_uri=error.type_uri,
        title=error.title,
        detail=error.detail,
        errors=errors,
    )


async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
    try:
        phrase = HTTPStatus(exc.status_code).phrase
    except ValueError:
        phrase = "Error"
    return problem_response(
        status=exc.status_code,
        type_uri=f"/errors/http-{exc.status_code}",
        title=phrase,
        detail=str(exc.detail),
        headers=getattr(exc, "headers", None),
    )


def register_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppError, _app_error)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, _validation_error)  # type: ignore[arg-type]
    app.add_exception_handler(StarletteHTTPException, _http_error)  # type: ignore[arg-type]
