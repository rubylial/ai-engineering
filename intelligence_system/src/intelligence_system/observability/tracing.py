"""Trazabilidad con OpenTelemetry + OpenInference, exportada a Arize Phoenix.

Diseño ("caja de cristal"):
- Cada request abre un span raíz con `request_id`, `thread_id` y `session.id`
  (= thread_id, así Phoenix agrupa los turnos de una conversación).
- `using_attributes` propaga session/metadata a todos los spans que crean los
  instrumentadores de OpenInference (LangGraph/LangChain, Anthropic, OpenAI).
- Cada nodo del grafo abre un span `agent.<worker>` colgado del span del nodo
  de LangGraph, con el atributo `worker`; las llamadas al LLM quedan anidadas
  debajo de su worker.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from openinference.instrumentation import using_attributes
from openinference.semconv.trace import (
    DocumentAttributes,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)
from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode

from intelligence_system.observability.logging import get_logger

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import TracerProvider

    from intelligence_system.config import Settings

_TRACER_NAME = "intelligence_system"
_MAX_ATTR_CHARS = 4000
log = get_logger(__name__)


def setup_tracing(settings: Settings) -> TracerProvider | None:
    """Registra el TracerProvider de Phoenix e instrumenta LangChain y los SDKs de LLM."""
    if not settings.phoenix_enabled:
        log.info("tracing.disabled")
        return None
    from openinference.instrumentation.anthropic import AnthropicInstrumentor
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from openinference.instrumentation.openai import OpenAIInstrumentor
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
    LangChainInstrumentor().instrument(tracer_provider=provider)
    AnthropicInstrumentor().instrument(tracer_provider=provider)
    OpenAIInstrumentor().instrument(tracer_provider=provider)
    log.info(
        "tracing.enabled",
        endpoint=settings.phoenix_traces_endpoint,
        project=settings.phoenix_project_name,
    )
    return provider


def shutdown_tracing(provider: TracerProvider | None) -> None:
    """Vacía el buffer de spans antes de apagar (no perder las últimas trazas)."""
    if provider is None:
        return
    try:
        provider.force_flush()
        provider.shutdown()
    except Exception:  # pragma: no cover - el apagado nunca debe tumbar la app
        log.warning("tracing.shutdown_failed", exc_info=True)


def get_tracer() -> trace.Tracer:
    return trace.get_tracer(_TRACER_NAME)


def format_trace_id(span: Span) -> str | None:
    ctx = span.get_span_context()
    return format(ctx.trace_id, "032x") if ctx.is_valid else None


def current_trace_id() -> str | None:
    return format_trace_id(trace.get_current_span())


def _truncate(value: str) -> str:
    return value if len(value) <= _MAX_ATTR_CHARS else value[:_MAX_ATTR_CHARS] + "…"


@contextmanager
def request_span(
    name: str,
    *,
    request_id: str,
    thread_id: str,
    input_value: str | None = None,
) -> Iterator[Span]:
    """Span raíz de un request + atributos de sesión para todos sus hijos."""
    attributes: dict[str, Any] = {
        SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.CHAIN.value,
        SpanAttributes.SESSION_ID: thread_id,
        "request_id": request_id,
        "thread_id": thread_id,
        SpanAttributes.METADATA: json.dumps({"request_id": request_id, "thread_id": thread_id}),
    }
    if input_value is not None:
        attributes[SpanAttributes.INPUT_VALUE] = _truncate(input_value)
    with (
        get_tracer().start_as_current_span(name, attributes=attributes) as span,
        using_attributes(
            session_id=thread_id,
            metadata={"request_id": request_id, "thread_id": thread_id},
        ),
    ):
        yield span


def _langgraph_node_span() -> Span | None:
    """Span que OpenInference abrió para el nodo de LangGraph en ejecución."""
    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor, get_current_span

        if not LangChainInstrumentor().is_instrumented_by_opentelemetry:
            return None  # sin Phoenix (tests/offline): no hay span del nodo
        return get_current_span()
    except Exception:
        return None


@contextmanager
def agent_span(
    worker: str,
    *,
    request_id: str | None,
    thread_id: str | None,
    step: int,
) -> Iterator[Span]:
    """Span `agent.<worker>` anidado bajo el nodo de LangGraph (si está instrumentado)."""
    parent = _langgraph_node_span()
    context = trace.set_span_in_context(parent) if parent is not None else None
    attributes: dict[str, Any] = {
        SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.AGENT.value,
        "worker": worker,
        "agent.step": step,
    }
    if request_id:
        attributes["request_id"] = request_id
    if thread_id:
        attributes["thread_id"] = thread_id
        attributes[SpanAttributes.SESSION_ID] = thread_id
    with get_tracer().start_as_current_span(
        f"agent.{worker}", context=context, attributes=attributes
    ) as span:
        yield span


@contextmanager
def child_span(name: str, kind: OpenInferenceSpanKindValues, **attributes: Any) -> Iterator[Span]:
    attrs: dict[str, Any] = {SpanAttributes.OPENINFERENCE_SPAN_KIND: kind.value}
    attrs.update({k: v for k, v in attributes.items() if v is not None})
    with get_tracer().start_as_current_span(name, attributes=attrs) as span:
        yield span


def set_io(span: Span, *, input_value: str | None = None, output_value: str | None = None) -> None:
    if input_value is not None:
        span.set_attribute(SpanAttributes.INPUT_VALUE, _truncate(input_value))
    if output_value is not None:
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, _truncate(output_value))


def record_documents(span: Span, documents: Sequence[dict[str, Any]]) -> None:
    """Formato RETRIEVER de OpenInference: Phoenix muestra cada documento recuperado."""
    prefix = SpanAttributes.RETRIEVAL_DOCUMENTS
    for i, doc in enumerate(documents):
        span.set_attribute(f"{prefix}.{i}.{DocumentAttributes.DOCUMENT_ID}", str(doc["id"]))
        span.set_attribute(
            f"{prefix}.{i}.{DocumentAttributes.DOCUMENT_CONTENT}", _truncate(str(doc["content"]))
        )
        if doc.get("score") is not None:
            span.set_attribute(f"{prefix}.{i}.{DocumentAttributes.DOCUMENT_SCORE}", float(doc["score"]))
        if doc.get("metadata"):
            span.set_attribute(
                f"{prefix}.{i}.{DocumentAttributes.DOCUMENT_METADATA}",
                json.dumps(doc["metadata"], ensure_ascii=False),
            )


def mark_error(span: Span, exc: BaseException, *, description: str | None = None) -> None:
    span.record_exception(exc)
    span.set_status(Status(StatusCode.ERROR, description or type(exc).__name__))
