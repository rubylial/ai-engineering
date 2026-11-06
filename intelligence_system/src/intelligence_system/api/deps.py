"""Dependencias de FastAPI: contenedor, autenticación por X-API-Key y servicios."""

from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import Depends, Request, Security
from fastapi.security import APIKeyHeader

from intelligence_system.agents.service import ChatService
from intelligence_system.container import AppContainer
from intelligence_system.errors import AuthError, NotReadyError
from intelligence_system.rag.ingest import IngestService

api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
    description="Clave definida en la variable API_KEY del servidor.",
)


def get_container(request: Request) -> AppContainer:
    container: AppContainer = request.app.state.container
    return container


ContainerDep = Annotated[AppContainer, Depends(get_container)]


async def require_api_key(
    container: ContainerDep,
    api_key: Annotated[str | None, Security(api_key_header)] = None,
) -> None:
    expected = container.settings.api_key
    if expected is None or not expected.get_secret_value():
        raise NotReadyError("El servidor no tiene API_KEY configurada.")
    if not api_key or not secrets.compare_digest(
        api_key.encode("utf-8"), expected.get_secret_value().encode("utf-8")
    ):
        raise AuthError("Falta el header X-API-Key o no es válido.")


def _not_ready(container: AppContainer, component: str) -> NotReadyError:
    problems = "; ".join(f"{k}: {v}" for k, v in sorted(container.problems.items()))
    return NotReadyError(f"{component} no disponible. {problems or 'Revisá GET /ready.'}"[:500])


def get_chat_service(container: ContainerDep) -> ChatService:
    if container.chat is None:
        raise _not_ready(container, "El agente")
    return container.chat


def get_ingest_service(container: ContainerDep) -> IngestService:
    if container.ingest is None:
        raise _not_ready(container, "La ingesta")
    return container.ingest


ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]
IngestServiceDep = Annotated[IngestService, Depends(get_ingest_service)]
