"""Aplicación FastAPI: lifespan con clientes compartidos, middleware, errores y routers."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

from fastapi import FastAPI

from intelligence_system.api.errors import register_error_handlers
from intelligence_system.api.middleware import RequestContextMiddleware
from intelligence_system.api.routers import chat, health, ingest, threads
from intelligence_system.config import Settings, get_settings
from intelligence_system.container import AppContainer, build_container
from intelligence_system.observability.logging import configure_logging, get_logger
from intelligence_system.observability.tracing import setup_tracing, shutdown_tracing

ContainerFactory = Callable[[Settings], AbstractAsyncContextManager[AppContainer]]
log = get_logger(__name__)

DESCRIPTION = """\
Sistema de grado producción que integra **RAG híbrido** (Pinecone denso + BM25),
**agentes con Supervisor** (LangGraph, handoffs con `Command`) y **trazabilidad
completa** en Arize Phoenix.

- Autenticación: header `X-API-Key`.
- Errores: `application/problem+json` con `request_id` y `trace_id`.
- Cada request devuelve `X-Request-ID`; las respuestas de chat incluyen `trace_id`
  para abrir la traza en Phoenix.
"""

TAGS = [
    {"name": "health", "description": "Liveness y readiness."},
    {"name": "chat", "description": "Conversación con el sistema multi-agente."},
    {"name": "ingest", "description": "Ingesta idempotente de documentos al índice."},
    {"name": "threads", "description": "Historial persistido por thread_id."},
]


def create_app(
    settings: Settings | None = None,
    *,
    container_factory: ContainerFactory | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_json)
    factory = container_factory or build_container

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        provider = setup_tracing(settings)
        app.state.started_at = time.monotonic()
        log.info("app.starting", env=settings.app_env, version=settings.app_version)
        try:
            async with factory(settings) as container:
                app.state.container = container
                yield
        finally:
            log.info("app.stopping")
            shutdown_tracing(provider)

    app = FastAPI(
        title="The Intelligence System",
        version=settings.app_version,
        description=DESCRIPTION,
        openapi_tags=TAGS,
        lifespan=lifespan,
        telemetry={
            "tracing": settings.phoenix_http_spans,
            "operation_spans": settings.phoenix_http_spans,
            "metrics": False,
            "logs": False,
            "auto_configure": False,
        },
    )
    app.add_middleware(RequestContextMiddleware)
    register_error_handlers(app)
    for module in (health, chat, ingest, threads):
        app.include_router(module.router)
    return app
