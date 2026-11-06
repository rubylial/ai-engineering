"""Contratos Pydantic del RAG: Document, Chunk, SparseVector, filtros y resultados.

Toda la información que cruza un borde del RAG (ingesta, vector store,
retriever, tools del agente) viaja validada por estos modelos.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DOC_ID_PATTERN = r"^[a-z0-9][a-z0-9_-]{0,63}$"
CATEGORY_PATTERN = r"^[a-z0-9_-]{1,40}$"
MAX_SPARSE_INDEX = 2**32


def content_hash(text: str) -> str:
    """SHA-256 del texto normalizado (NFC, espacios colapsados): base de la deduplicación."""
    normalized = " ".join(unicodedata.normalize("NFC", text).split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class Document(BaseModel):
    """Documento fuente del corpus (un archivo markdown con front matter)."""

    doc_id: str = Field(pattern=DOC_ID_PATTERN, examples=["05_pinecone_serverless"])
    title: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=500_000)
    source: str = Field(min_length=1, max_length=300)
    category: str = Field(default="general", pattern=CATEGORY_PATTERN)
    tags: list[str] = Field(default_factory=list, max_length=20)

    @property
    def content_hash(self) -> str:
        return content_hash(self.text)


class Chunk(BaseModel):
    """Fragmento de un documento listo para indexar."""

    chunk_id: str = Field(max_length=512)
    doc_id: str = Field(pattern=DOC_ID_PATTERN)
    chunk_index: int = Field(ge=0)
    text: str = Field(min_length=1)
    title: str
    section: str
    source: str
    category: str
    tags: list[str] = Field(default_factory=list)
    content_hash: str = Field(min_length=64, max_length=64)

    def metadata(self) -> dict[str, Any]:
        """Metadatos aptos para Pinecone (str, números, bool o lista de str)."""
        return {
            "doc_id": self.doc_id,
            "chunk_index": self.chunk_index,
            "title": self.title,
            "section": self.section,
            "source": self.source,
            "category": self.category,
            "tags": self.tags,
            "content_hash": self.content_hash,
            "text": self.text,
        }


class SparseVector(BaseModel):
    """Vector disperso (BM25): índices de términos hasheados + pesos."""

    indices: list[int] = Field(default_factory=list)
    values: list[float] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> SparseVector:
        if len(self.indices) != len(self.values):
            raise ValueError("indices y values deben tener el mismo largo")
        if len(set(self.indices)) != len(self.indices):
            raise ValueError("indices repetidos en el vector disperso")
        if any(i < 0 or i >= MAX_SPARSE_INDEX for i in self.indices):
            raise ValueError("los índices dispersos deben estar en [0, 2^32)")
        return self

    @property
    def is_empty(self) -> bool:
        return not self.indices

    def scaled(self, factor: float) -> SparseVector:
        return SparseVector(indices=list(self.indices), values=[v * factor for v in self.values])

    def as_dict(self) -> dict[str, list[Any]]:
        return {"indices": list(self.indices), "values": list(self.values)}


class MetadataFilter(BaseModel):
    """Filtros de metadatos permitidos (lista blanca: no se aceptan operadores libres)."""

    model_config = ConfigDict(extra="forbid")

    doc_ids: list[str] | None = Field(default=None, max_length=50, examples=[["05_pinecone_serverless"]])
    categories: list[str] | None = Field(default=None, max_length=20, examples=[["vector_db"]])

    @field_validator("doc_ids")
    @classmethod
    def _valid_doc_ids(cls, value: list[str] | None) -> list[str] | None:
        if value and any(not re.match(DOC_ID_PATTERN, v) for v in value):
            raise ValueError("doc_ids inválidos")
        return value

    @field_validator("categories")
    @classmethod
    def _valid_categories(cls, value: list[str] | None) -> list[str] | None:
        if value and any(not re.match(CATEGORY_PATTERN, v) for v in value):
            raise ValueError("categories inválidas")
        return value

    def to_pinecone(self) -> dict[str, Any] | None:
        clauses: list[dict[str, Any]] = []
        if self.doc_ids:
            clauses.append({"doc_id": {"$in": self.doc_ids}})
        if self.categories:
            clauses.append({"category": {"$in": self.categories}})
        if not clauses:
            return None
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}


class RetrievalResult(BaseModel):
    """Un fragmento recuperado, con su score híbrido y su procedencia."""

    chunk_id: str
    doc_id: str
    title: str
    section: str
    source: str
    category: str
    text: str
    content_hash: str
    score: float
    rerank_score: float | None = None

    @classmethod
    def from_metadata(cls, chunk_id: str, score: float, metadata: dict[str, Any]) -> RetrievalResult:
        return cls(
            chunk_id=chunk_id,
            doc_id=str(metadata.get("doc_id", "")),
            title=str(metadata.get("title", "")),
            section=str(metadata.get("section", "")),
            source=str(metadata.get("source", "")),
            category=str(metadata.get("category", "general")),
            text=str(metadata.get("text", "")),
            content_hash=str(metadata.get("content_hash", "")),
            score=float(score),
        )


class IngestReport(BaseModel):
    """Resultado de una ingesta idempotente."""

    documents: int = Field(ge=0)
    chunks_total: int = Field(ge=0)
    chunks_upserted: int = Field(ge=0)
    chunks_unchanged: int = Field(ge=0)
    chunks_deleted: int = Field(ge=0)
    duration_ms: float = Field(ge=0)
