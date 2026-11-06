"""Observabilidad: Arize Phoenix vía OpenInference.

`setup_observability` registra el TracerProvider de Phoenix e instrumenta LangChain.
El `LangChainInstrumentor` traza automáticamente cada nodo del grafo y cada llamada
al LLM —no hace falta decorar funciones a mano—. Para agrupar en Phoenix todas las
trazas de un mismo job (incluido el turno de aprobación), el worker envuelve cada
ejecución en `job_trace_context(job_id)`, que setea `session.id = job_id`.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

from app.config import Settings

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import TracerProvider

log = logging.getLogger(__name__)


def setup_observability(settings: Settings) -> TracerProvider | None:
    """Registra Phoenix e instrumenta LangChain/LangGraph. Devuelve el provider o None."""
    if not settings.phoenix_enabled:
        log.info("observability.disabled")
        return None

    from openinference.instrumentation.langchain import LangChainInstrumentor
    from phoenix.otel import register

    provider = register(
        endpoint=settings.phoenix_traces_endpoint,
        project_name=settings.phoenix_project_name,
        protocol="http/protobuf",
        batch=settings.phoenix_batch_export,
        set_global_tracer_provider=True,
        verbose=False,
        auto_instrument=False,
    )
    LangChainInstrumentor().instrument(tracer_provider=provider, skip_dep_check=True)
    log.info(
        "observability.enabled endpoint=%s project=%s",
        settings.phoenix_traces_endpoint,
        settings.phoenix_project_name,
    )
    return provider


@contextmanager
def job_trace_context(job_id: str) -> Iterator[None]:
    """Agrupa las trazas de un job bajo session.id = job_id (vista Sessions de Phoenix)."""
    try:
        from openinference.instrumentation import using_attributes
    except ImportError:  # sin openinference instalado (no debería pasar)
        yield
        return
    with using_attributes(session_id=job_id, metadata={"job_id": job_id}):
        yield
