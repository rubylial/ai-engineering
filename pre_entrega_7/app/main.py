"""FastAPI de producción: arma el runtime en el lifespan y monta las rutas.

Endpoints (en app/routes.py):
- POST /tasks                 → encola y devuelve job_id (202). No bloquea el event loop.
- GET  /tasks/{id}            → estado del job desde Redis.
- POST /tasks/{id}/approve    → aprueba/rechaza un job en WAITING_APPROVAL y lo reanuda.
- GET  /health, GET /ready    → liveness / readiness.

El trabajo pesado (correr el grafo) lo hace el Worker en segundo plano.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from redis.asyncio import Redis

from app.config import get_settings
from app.graph import build_graph_from_llm
from app.jobs import JobStore
from app.llm import build_chat_model
from app.observability import setup_observability
from app.routes import router
from app.worker import Worker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    setup_observability(settings)

    redis: Redis = Redis.from_url(settings.redis_url, decode_responses=True)
    store = JobStore(redis, ttl_seconds=settings.job_ttl_seconds)

    # Checkpointer de LangGraph sobre Redis (sobrevive reinicios de la API).
    from langgraph.checkpoint.redis.aio import AsyncRedisSaver

    ttl = (
        {"default_ttl": settings.checkpoint_ttl_minutes, "refresh_on_read": True}
        if settings.checkpoint_ttl_minutes
        else None
    )
    checkpointer_cm = AsyncRedisSaver.from_conn_string(settings.redis_url, ttl=ttl)
    checkpointer = await checkpointer_cm.__aenter__()

    graph = build_graph_from_llm(build_chat_model(), checkpointer=checkpointer)
    worker = Worker(
        graph,
        store,
        critical_keywords=settings.critical_keyword_set,
        concurrency=settings.worker_concurrency,
        recursion_limit=settings.recursion_limit,
    )
    await worker.start()

    app.state.redis = redis
    app.state.store = store
    app.state.worker = worker
    log.info("api.ready")

    try:
        yield
    finally:
        await worker.stop()
        await checkpointer_cm.__aexit__(None, None, None)
        await redis.aclose()
        log.info("api.shutdown")


def create_app() -> FastAPI:
    """Factory para uvicorn --factory."""
    app = FastAPI(title="API Agente (M7)", version="0.1.0", lifespan=lifespan)
    app.include_router(router)
    return app


app = create_app()
