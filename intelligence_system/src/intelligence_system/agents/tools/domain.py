"""Tools de dominio del analyst: cálculos de AI Engineering, deterministas y validados.

Patrón de la pre-entrega 5: cada tool documenta cuándo usarla y cuándo no, y
los errores de validación vuelven al LLM como datos para que corrija los
argumentos (en lugar de tumbar el agente). La lógica de métricas y RRF se
movió desde `evaluate.py` / `retriever.py` de la pre-entrega 4.
"""

from __future__ import annotations

import asyncio
import math
from typing import Literal

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field, model_validator

from intelligence_system.rag.embeddings import EmbeddingsProvider
from intelligence_system.rag.evaluation import (
    hit_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    rrf_fuse,
)

# ---------------------------------------------------------------------------
# estimar_chunks
# ---------------------------------------------------------------------------


class EstimarChunksInput(BaseModel):
    total_tokens: int = Field(gt=0, le=50_000_000, description="Tokens totales del documento.")
    chunk_size: int = Field(gt=0, le=100_000, description="Tamaño de cada chunk, en tokens.")
    chunk_overlap: int = Field(ge=0, le=100_000, description="Solapamiento entre chunks, en tokens.")

    @model_validator(mode="after")
    def _overlap_lt_size(self) -> EstimarChunksInput:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap debe ser menor que chunk_size")
        return self


class EstimarChunksOutput(BaseModel):
    num_chunks: int
    paso_efectivo: int
    formula: str


async def _estimar_chunks(total_tokens: int, chunk_size: int, chunk_overlap: int) -> EstimarChunksOutput:
    step = chunk_size - chunk_overlap
    extra = max(0, total_tokens - chunk_size)
    num = 1 + math.ceil(extra / step)
    return EstimarChunksOutput(
        num_chunks=num,
        paso_efectivo=step,
        formula=f"1 + ceil(max(0, {total_tokens} - {chunk_size}) / ({chunk_size} - {chunk_overlap})) = {num}",
    )


# ---------------------------------------------------------------------------
# calcular_metricas_retrieval
# ---------------------------------------------------------------------------


class MetricasRetrievalInput(BaseModel):
    recuperados: list[str] = Field(min_length=1, max_length=200, description="IDs recuperados, en orden.")
    relevantes: list[str] = Field(min_length=1, max_length=200, description="IDs realmente relevantes.")
    k: int = Field(ge=1, le=200, description="Corte k para las métricas @k.")


class MetricasRetrievalOutput(BaseModel):
    k: int
    precision_at_k: float
    recall_at_k: float
    hit_at_k: int
    reciprocal_rank: float
    relevantes_en_top_k: list[str]


async def _calcular_metricas(
    recuperados: list[str], relevantes: list[str], k: int
) -> MetricasRetrievalOutput:
    relevant_set = set(relevantes)
    return MetricasRetrievalOutput(
        k=k,
        precision_at_k=round(precision_at_k(recuperados, relevantes, k), 4),
        recall_at_k=round(recall_at_k(recuperados, relevantes, k), 4),
        hit_at_k=hit_at_k(recuperados, relevantes, k),
        reciprocal_rank=round(reciprocal_rank(recuperados, relevantes), 4),
        relevantes_en_top_k=[r for r in recuperados[:k] if r in relevant_set],
    )


# ---------------------------------------------------------------------------
# fusionar_rankings_rrf
# ---------------------------------------------------------------------------


class FusionRRFInput(BaseModel):
    rankings: list[list[str]] = Field(
        min_length=2, max_length=6, description="Dos o más rankings de IDs, cada uno en orden."
    )
    k: int = Field(default=60, ge=1, le=1000, description="Hiperparámetro k de RRF (típico 60).")
    top_n: int = Field(default=10, ge=1, le=100)

    @model_validator(mode="after")
    def _non_empty(self) -> FusionRRFInput:
        if any(not ranking for ranking in self.rankings):
            raise ValueError("ningún ranking puede estar vacío")
        return self


class RankedItem(BaseModel):
    posicion: int
    id: str
    score: float


class FusionRRFOutput(BaseModel):
    ranking: list[RankedItem]


async def _fusionar_rrf(rankings: list[list[str]], k: int = 60, top_n: int = 10) -> FusionRRFOutput:
    fused = rrf_fuse(rankings, k=k)[:top_n]
    return FusionRRFOutput(
        ranking=[
            RankedItem(posicion=i, id=item, score=round(score, 6))
            for i, (item, score) in enumerate(fused, start=1)
        ]
    )


# ---------------------------------------------------------------------------
# estimar_memoria_vectores
# ---------------------------------------------------------------------------


class MemoriaVectoresInput(BaseModel):
    num_vectores: int = Field(gt=0, le=10**12, description="Cantidad de vectores del índice.")
    dimension: int = Field(gt=0, le=65_536, description="Dimensión de cada vector.")
    bytes_por_componente: Literal[1, 2, 4, 8] = Field(default=4, description="4 = float32.")
    metadata_bytes_por_vector: int = Field(default=0, ge=0, le=40_960)


class MemoriaVectoresOutput(BaseModel):
    bytes_totales: int
    megabytes: float
    gigabytes: float
    gibibytes: float
    formula: str


async def _estimar_memoria(
    num_vectores: int,
    dimension: int,
    bytes_por_componente: int = 4,
    metadata_bytes_por_vector: int = 0,
) -> MemoriaVectoresOutput:
    per_vector = dimension * bytes_por_componente + metadata_bytes_por_vector
    total = num_vectores * per_vector
    return MemoriaVectoresOutput(
        bytes_totales=total,
        megabytes=round(total / 1e6, 3),
        gigabytes=round(total / 1e9, 3),
        gibibytes=round(total / 2**30, 3),
        formula=(
            f"{num_vectores} × ({dimension} × {bytes_por_componente} B + "
            f"{metadata_bytes_por_vector} B) = {total} B"
        ),
    )


# ---------------------------------------------------------------------------
# similitud_coseno (usa el proveedor de embeddings: I/O async)
# ---------------------------------------------------------------------------


class SimilitudCosenoInput(BaseModel):
    texto_a: str = Field(min_length=1, max_length=2000)
    texto_b: str = Field(min_length=1, max_length=2000)


class SimilitudCosenoOutput(BaseModel):
    similitud: float
    modelo: str
    interpretacion: str


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def make_domain_tools(embeddings: EmbeddingsProvider) -> list[BaseTool]:
    async def _similitud(texto_a: str, texto_b: str) -> SimilitudCosenoOutput:
        vec_a, vec_b = await asyncio.gather(embeddings.embed_query(texto_a), embeddings.embed_query(texto_b))
        value = round(cosine(vec_a, vec_b), 4)
        level = "alta" if value >= 0.85 else "media" if value >= 0.7 else "baja"
        return SimilitudCosenoOutput(
            similitud=value,
            modelo=embeddings.model,
            interpretacion=f"similitud {level} para el modelo {embeddings.model}",
        )

    return [
        StructuredTool.from_function(
            coroutine=_estimar_chunks,
            name="estimar_chunks",
            description=(
                "Calcula cuántos chunks genera un documento dado su total de tokens, el "
                "chunk_size y el chunk_overlap. Usala cuando pregunten cuántos fragmentos "
                "salen de un documento. No la uses para tamaños en caracteres sin convertir."
            ),
            args_schema=EstimarChunksInput,
        ),
        StructuredTool.from_function(
            coroutine=_calcular_metricas,
            name="calcular_metricas_retrieval",
            description=(
                "Calcula precision@k, recall@k, hit@k y reciprocal rank a partir de la lista "
                "ordenada de IDs recuperados y la lista de IDs relevantes. Usala para evaluar "
                "un resultado de retrieval. No la uses sin la lista de relevantes."
            ),
            args_schema=MetricasRetrievalInput,
        ),
        StructuredTool.from_function(
            coroutine=_fusionar_rrf,
            name="fusionar_rankings_rrf",
            description=(
                "Fusiona dos o más rankings con Reciprocal Rank Fusion (1/(k+rank)). Usala "
                "cuando pidan combinar resultados de varios retrievers."
            ),
            args_schema=FusionRRFInput,
        ),
        StructuredTool.from_function(
            coroutine=_estimar_memoria,
            name="estimar_memoria_vectores",
            description=(
                "Estima la memoria que ocupan N vectores de una dimensión dada (float32 por "
                "defecto). Usala para dimensionar un índice. No incluye overhead del motor."
            ),
            args_schema=MemoriaVectoresInput,
        ),
        StructuredTool.from_function(
            coroutine=_similitud,
            name="similitud_coseno",
            description=(
                "Calcula la similitud coseno entre dos textos con el modelo de embeddings del "
                "sistema. Usala para comparar qué tan parecidos son dos textos. No la uses "
                "para buscar en la documentación."
            ),
            args_schema=SimilitudCosenoInput,
        ),
    ]
