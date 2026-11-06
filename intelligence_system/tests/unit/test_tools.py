from __future__ import annotations

import pytest
from pydantic import ValidationError

from helpers import DIM
from intelligence_system.agents.tools import make_domain_tools, make_search_tool
from intelligence_system.agents.tools.domain import (
    EstimarChunksOutput,
    FusionRRFOutput,
    MemoriaVectoresOutput,
    MetricasRetrievalOutput,
    SimilitudCosenoOutput,
)
from intelligence_system.agents.tools.knowledge import SearchKnowledgeBaseOutput
from intelligence_system.container import AppContainer
from intelligence_system.rag.embeddings import HashEmbeddings
from intelligence_system.rag.evaluation import precision_at_k, recall_at_k, reciprocal_rank, rrf_fuse


@pytest.fixture
def tools() -> dict[str, object]:
    return {t.name: t for t in make_domain_tools(HashEmbeddings(dimension=DIM))}


def test_every_tool_has_args_schema_and_guidance(tools: dict[str, object]) -> None:
    for tool in tools.values():
        assert tool.args_schema is not None  # type: ignore[attr-defined]
        assert "Usala" in tool.description or "Usá" in tool.description  # type: ignore[attr-defined]


async def test_estimar_chunks(tools: dict[str, object]) -> None:
    out = await tools["estimar_chunks"].ainvoke(  # type: ignore[attr-defined]
        {"total_tokens": 12000, "chunk_size": 500, "chunk_overlap": 50}
    )
    assert isinstance(out, EstimarChunksOutput)
    assert out.num_chunks == 27 and out.paso_efectivo == 450
    small = await tools["estimar_chunks"].ainvoke(  # type: ignore[attr-defined]
        {"total_tokens": 100, "chunk_size": 500, "chunk_overlap": 50}
    )
    assert small.num_chunks == 1


async def test_estimar_chunks_rejects_invalid_args(tools: dict[str, object]) -> None:
    with pytest.raises(ValidationError, match="chunk_overlap"):
        await tools["estimar_chunks"].ainvoke(  # type: ignore[attr-defined]
            {"total_tokens": 1000, "chunk_size": 100, "chunk_overlap": 100}
        )
    with pytest.raises(ValidationError):
        await tools["estimar_chunks"].ainvoke({"total_tokens": -5, "chunk_size": 100, "chunk_overlap": 0})  # type: ignore[attr-defined]


async def test_metricas_retrieval(tools: dict[str, object]) -> None:
    out = await tools["calcular_metricas_retrieval"].ainvoke(  # type: ignore[attr-defined]
        {"recuperados": ["a", "b", "c", "d", "e"], "relevantes": ["b", "f"], "k": 5}
    )
    assert isinstance(out, MetricasRetrievalOutput)
    assert (out.precision_at_k, out.recall_at_k, out.hit_at_k, out.reciprocal_rank) == (0.2, 0.5, 1, 0.5)
    assert out.relevantes_en_top_k == ["b"]


async def test_rrf_and_memory_tools(tools: dict[str, object]) -> None:
    fused = await tools["fusionar_rankings_rrf"].ainvoke(  # type: ignore[attr-defined]
        {"rankings": [["a", "b", "c"], ["b", "c", "a"]], "top_n": 2}
    )
    assert isinstance(fused, FusionRRFOutput)
    assert [item.id for item in fused.ranking] == ["b", "a"]
    memory = await tools["estimar_memoria_vectores"].ainvoke(  # type: ignore[attr-defined]
        {"num_vectores": 1_000_000, "dimension": 1536}
    )
    assert isinstance(memory, MemoriaVectoresOutput)
    assert memory.bytes_totales == 6_144_000_000 and memory.gigabytes == 6.144


async def test_similitud_coseno_uses_embeddings(tools: dict[str, object]) -> None:
    out = await tools["similitud_coseno"].ainvoke(  # type: ignore[attr-defined]
        {"texto_a": "bm25 búsqueda léxica", "texto_b": "bm25 búsqueda léxica"}
    )
    assert isinstance(out, SimilitudCosenoOutput)
    assert out.similitud == pytest.approx(1.0)


def test_evaluation_functions() -> None:
    assert precision_at_k(["a", "b"], ["a"], 2) == 0.5
    assert recall_at_k(["a", "b"], ["a", "z"], 2) == 0.5
    assert reciprocal_rank(["x", "a"], ["a"]) == 0.5
    assert rrf_fuse([["a"], ["a", "b"]], k=60)[0][0] == "a"
    with pytest.raises(ValueError, match="positivo"):
        precision_at_k(["a"], ["a"], 0)


async def test_search_tool_validates_and_returns_typed_results(ingested: AppContainer) -> None:
    assert ingested.retriever is not None
    tool = make_search_tool(ingested.retriever)
    out = await tool.ainvoke({"query": "namespaces en Pinecone", "top_k": 2, "categories": ["vector_db"]})
    assert isinstance(out, SearchKnowledgeBaseOutput)
    assert out.total == 2 and all(r.category == "vector_db" for r in out.results)
    with pytest.raises(ValidationError):
        await tool.ainvoke({"query": "x"})  # menos de 3 caracteres
    with pytest.raises(ValidationError):
        await tool.ainvoke({"query": "consulta válida", "doc_ids": ["NO VALIDO"]})
