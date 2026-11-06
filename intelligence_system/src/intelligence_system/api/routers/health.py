"""GET /health (liveness) y GET /ready (readiness con detalle de qué falta)."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from intelligence_system.api.deps import ContainerDep
from intelligence_system.api.schemas import ComponentStatus, HealthResponse, ReadyResponse
from intelligence_system.container import AppContainer

router = APIRouter(tags=["health"])
_CHECK_TIMEOUT_S = 5.0


@router.get("/health", response_model=HealthResponse, summary="Liveness: el proceso responde")
async def health(request: Request, container: ContainerDep) -> HealthResponse:
    started = getattr(request.app.state, "started_at", time.monotonic())
    return HealthResponse(
        app=container.settings.app_name,
        version=container.settings.app_version,
        uptime_s=round(time.monotonic() - started, 1),
    )


async def _timed(name: str, probe: Callable[[], Awaitable[str]]) -> ComponentStatus:
    started = time.perf_counter()
    try:
        detail = await asyncio.wait_for(probe(), timeout=_CHECK_TIMEOUT_S)
        ok = True
    except TimeoutError:
        detail, ok = f"sin respuesta en {_CHECK_TIMEOUT_S:g}s", False
    except Exception as exc:
        detail, ok = f"{type(exc).__name__}: {exc}"[:300], False
    return ComponentStatus(
        name=name, ok=ok, detail=detail, latency_ms=round((time.perf_counter() - started) * 1000, 1)
    )


def _probes(container: AppContainer) -> dict[str, Callable[[], Awaitable[str]]]:
    settings = container.settings
    probes: dict[str, Callable[[], Awaitable[str]]] = {}

    async def redis_probe() -> str:
        if settings.checkpointer_backend == "memory":
            return "checkpointer en memoria (no persistente)"
        if container.redis is None:
            raise RuntimeError("cliente Redis no inicializado")
        await container.redis.ping()
        return settings.redis_url.rsplit("@", 1)[-1]

    async def vector_probe() -> str:
        if container.vector_store is None:
            raise RuntimeError(container.problems.get("vector_store", "vector store no inicializado"))
        status = await container.vector_store.health()
        if not status.ok:
            raise RuntimeError(status.detail)
        return f"{status.detail} · {status.vector_count} vectores"

    async def llm_probe() -> str:
        if container.llm is None:
            raise RuntimeError(container.problems.get("llm", "LLM no inicializado"))
        models = sorted({settings.llm_model_supervisor, settings.llm_model_worker})
        await asyncio.gather(*(container.llm.check_model(m) for m in models))
        return f"{settings.llm_provider}: {', '.join(models)}"

    async def agent_probe() -> str:
        if container.chat is None:
            raise RuntimeError("grafo no construido (depende de LLM, vector store y checkpointer)")
        return "grafo compilado"

    probes["redis"] = redis_probe
    probes["vector_store"] = vector_probe
    probes["llm"] = llm_probe
    probes["agent"] = agent_probe
    return probes


@router.get(
    "/ready",
    response_model=ReadyResponse,
    summary="Readiness: verifica Redis, Pinecone y el LLM, e informa qué falta",
    responses={503: {"model": ReadyResponse, "description": "Algún componente no está listo"}},
)
async def ready(container: ContainerDep) -> JSONResponse:
    probes = _probes(container)
    components = list(await asyncio.gather(*(_timed(n, p) for n, p in probes.items())))
    missing = container.settings.missing_credentials()
    if missing:
        components.insert(0, ComponentStatus(name="config", ok=False, detail=f"faltan: {', '.join(missing)}"))
    is_ready = all(c.ok for c in components)
    body = ReadyResponse(status="ready" if is_ready else "not_ready", components=components, missing=missing)
    return JSONResponse(body.model_dump(), status_code=200 if is_ready else 503)
