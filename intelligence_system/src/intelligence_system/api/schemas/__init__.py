"""Modelos Pydantic de entrada/salida de la API (con ejemplos para OpenAPI)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intelligence_system.agents.schemas import Citation, StepInfo, ThreadHistory, TurnResult
from intelligence_system.rag.schemas import (
    CATEGORY_PATTERN,
    DOC_ID_PATTERN,
    Document,
    IngestReport,
    MetadataFilter,
)

THREAD_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$"


class ProblemDetail(BaseModel):
    """Error uniforme (RFC 7807): siempre incluye request_id para correlacionar."""

    type: str = Field(examples=["/errors/validation-error"])
    title: str = Field(examples=["La solicitud no es válida"])
    status: int = Field(examples=[422])
    detail: str = Field(examples=["message: String should have at least 1 character"])
    request_id: str | None = Field(default=None, examples=["req_4f1c2a9e0b7d4c55a1e3f8d2c6b9a0e1"])
    trace_id: str | None = None
    errors: list[dict[str, str]] | None = None


ERROR_RESPONSES: dict[int | str, dict[str, object]] = {
    401: {"model": ProblemDetail, "description": "Falta o es inválida la X-API-Key"},
    422: {"model": ProblemDetail, "description": "Entrada inválida"},
    502: {"model": ProblemDetail, "description": "Falló un proveedor externo (LLM o Pinecone)"},
    503: {"model": ProblemDetail, "description": "Un componente requerido no está listo"},
    504: {"model": ProblemDetail, "description": "Se excedió el deadline del turno"},
}


# --------------------------------------------------------------------------- health


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    app: str
    version: str
    uptime_s: float


class ComponentStatus(BaseModel):
    name: str
    ok: bool
    detail: str
    latency_ms: float | None = None


class ReadyResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    components: list[ComponentStatus]
    missing: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- chat


class ChatRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "thread_id": "demo-thread-1",
                    "message": "¿Para qué sirve asyncio.Semaphore cuando llamo a un LLM?",
                },
                {
                    "message": "¿Qué ventaja tiene BM25 sobre la búsqueda semántica?",
                    "filters": {"categories": ["retrieval"]},
                },
            ]
        },
    )

    thread_id: str | None = Field(
        default=None,
        pattern=THREAD_ID_PATTERN,
        description="Conversación a continuar. Si se omite, se crea un thread nuevo.",
    )
    message: str = Field(min_length=1, max_length=4000)
    filters: MetadataFilter | None = None
    deadline_s: float | None = Field(
        default=None,
        gt=0,
        le=300,
        description="Tiempo máximo del turno (por defecto AGENT_REQUEST_TIMEOUT_S).",
    )

    @field_validator("message")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("el mensaje no puede estar vacío")
        return value.strip()


def _step(step: int, node: str, detail: str, duration_ms: float) -> dict[str, object]:
    return {"step": step, "node": node, "status": "ok", "detail": detail, "duration_ms": duration_ms}


class ChatResponse(TurnResult):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "thread_id": "demo-thread-1",
                    "request_id": "req_4f1c2a9e0b7d4c55a1e3f8d2c6b9a0e1",
                    "answer": "Limita cuántas corrutinas corren a la vez, para no recibir 429 [1].",
                    "grounded": True,
                    "citations": [
                        {
                            "marker": 1,
                            "id": "01_asyncio_python#3f9a1c2d4b5e6f70",
                            "kind": "document",
                            "title": "Asyncio en Python 3.12 › Patrones de concurrencia",
                            "source": "01_asyncio_python.md",
                            "snippet": "Un semáforo limita cuántas corrutinas pueden ejecutarse...",
                            "score": 0.83,
                        }
                    ],
                    "steps": [
                        _step(1, "supervisor", "→ researcher (llm)", 812.4),
                        _step(2, "researcher", "2/5 fragmentos relevantes", 1530.0),
                        _step(3, "supervisor", "→ writer (llm)", 640.2),
                        _step(4, "writer", "1 cita(s) válidas", 2101.7),
                    ],
                    "trace_id": "6f1d2c3b4a5968778695a4b3c2d1e0f9",
                }
            ]
        }
    )


# --------------------------------------------------------------------------- ingest


class DocumentIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(pattern=DOC_ID_PATTERN)
    title: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=500_000)
    source: str = Field(default="api", max_length=300)
    category: str = Field(default="general", pattern=CATEGORY_PATTERN)
    tags: list[str] = Field(default_factory=list, max_length=20)

    def to_document(self) -> Document:
        return Document(**self.model_dump())


class IngestRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {"from_corpus": True},
                {
                    "documents": [
                        {
                            "doc_id": "nota-interna-01",
                            "title": "Guía de chunking",
                            "text": "# Guía de chunking\n\nUsamos chunks de 500 tokens con 50 de overlap.",
                            "category": "retrieval",
                        }
                    ]
                },
            ]
        },
    )

    from_corpus: bool = Field(default=False, description="Ingestar el corpus incluido (RAG_CORPUS_DIR).")
    documents: list[DocumentIn] = Field(default_factory=list, max_length=100)

    @model_validator(mode="after")
    def _has_source(self) -> IngestRequest:
        if not self.from_corpus and not self.documents:
            raise ValueError("indicá from_corpus=true o al menos un documento")
        ids = [d.doc_id for d in self.documents]
        if len(ids) != len(set(ids)):
            raise ValueError("doc_id repetidos en la solicitud")
        return self


class IngestResponse(IngestReport):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "documents": 7,
                    "chunks_total": 41,
                    "chunks_upserted": 0,
                    "chunks_unchanged": 41,
                    "chunks_deleted": 0,
                    "duration_ms": 412.7,
                }
            ]
        }
    )


__all__ = [
    "ERROR_RESPONSES",
    "THREAD_ID_PATTERN",
    "ChatRequest",
    "ChatResponse",
    "Citation",
    "ComponentStatus",
    "DocumentIn",
    "HealthResponse",
    "IngestRequest",
    "IngestResponse",
    "ProblemDetail",
    "ReadyResponse",
    "StepInfo",
    "ThreadHistory",
]
