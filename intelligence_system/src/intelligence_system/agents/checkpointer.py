"""Checkpointer async de LangGraph: Redis en producción, memoria en tests.

`AsyncRedisSaver` (langgraph-checkpoint-redis) requiere Redis con RedisJSON y
RediSearch: vienen incluidos en Redis 8+ (imagen oficial `redis:8.x`). Su
`__aenter__` ya ejecuta `asetup()` (crea los índices la primera vez).
Las conversaciones sobreviven reinicios de la API porque viven en Redis; con
AOF activado en el contenedor de Redis, sobreviven también a su reinicio.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from intelligence_system.config import Settings
from intelligence_system.observability.logging import get_logger

log = get_logger(__name__)


@asynccontextmanager
async def open_checkpointer(settings: Settings) -> AsyncIterator[BaseCheckpointSaver[Any]]:
    if settings.checkpointer_backend == "memory":
        log.warning("checkpointer.memory", detail="las conversaciones se pierden al reiniciar")
        yield InMemorySaver()
        return

    from langgraph.checkpoint.redis.aio import AsyncRedisSaver

    ttl = (
        {"default_ttl": settings.checkpoint_ttl_minutes, "refresh_on_read": True}
        if settings.checkpoint_ttl_minutes
        else None
    )
    async with AsyncRedisSaver.from_conn_string(settings.redis_url, ttl=ttl) as saver:
        log.info("checkpointer.redis", ttl_minutes=settings.checkpoint_ttl_minutes)
        yield saver
