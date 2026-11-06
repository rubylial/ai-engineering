from __future__ import annotations

from pathlib import Path

import pytest

from helpers import CORPUS_DIR
from intelligence_system.rag.chunking import MarkdownChunker, chunk_id_for
from intelligence_system.rag.loaders import CorpusError, load_corpus, parse_markdown
from intelligence_system.rag.schemas import Document, MetadataFilter, SparseVector, content_hash

DOC = """---
doc_id: guia-01
title: Guía
category: retrieval
tags: [bm25, rrf]
---

# Guía

Intro breve que explica el documento completo con suficiente texto.

## Sección A

Contenido de la sección A con detalle suficiente para ser un chunk útil.

## Sección B

Contenido de la sección B, distinto del anterior, también suficientemente largo.
"""


def test_parse_markdown_reads_front_matter() -> None:
    doc = parse_markdown(DOC, fallback_id="x", source="guia.md")
    assert (doc.doc_id, doc.title, doc.category, doc.tags) == (
        "guia-01",
        "Guía",
        "retrieval",
        ["bm25", "rrf"],
    )
    assert not doc.text.startswith("---")


def test_parse_markdown_without_front_matter_uses_fallbacks() -> None:
    doc = parse_markdown("# Título H1\n\ncuerpo", fallback_id="Mi Archivo", source="a.md")
    assert doc.doc_id == "mi-archivo"
    assert doc.title == "Título H1"
    assert doc.category == "general"


async def test_load_corpus_reads_all_documents() -> None:
    docs = await load_corpus(CORPUS_DIR)
    assert len(docs) == 7
    assert {d.category for d in docs} >= {"python", "vector_db", "retrieval", "evaluation"}


async def test_load_corpus_errors(tmp_path: Path) -> None:
    with pytest.raises(CorpusError, match="no existe"):
        await load_corpus(tmp_path / "nada")
    with pytest.raises(CorpusError, match="vacío"):
        await load_corpus(tmp_path)


def test_chunker_respects_sections_and_ids_are_deterministic() -> None:
    doc = parse_markdown(DOC, fallback_id="x", source="guia.md")
    chunker = MarkdownChunker(chunk_size=200, chunk_overlap=20, min_chunk_chars=30)
    chunks = chunker.split(doc)
    sections = [c.section for c in chunks]
    assert "Guía › Sección A" in sections and "Guía › Sección B" in sections
    assert all(c.chunk_id == chunk_id_for(doc.doc_id, c.content_hash) for c in chunks)
    assert [c.chunk_id for c in chunker.split(doc)] == [c.chunk_id for c in chunks]  # idempotente
    assert all(len(c.text) <= 200 for c in chunks)


def test_chunker_skips_tiny_and_duplicated_chunks() -> None:
    text = (
        "# T\n\n## A\n\nTexto repetido con suficiente largo.\n\n## B\n\nTexto repetido con suficiente largo."
    )
    doc = Document(doc_id="dup", title="T", text=text, source="d.md")
    chunks = MarkdownChunker(chunk_size=500, chunk_overlap=0, min_chunk_chars=10).split(doc)
    bodies = [c.text.split("\n", 1)[-1].strip() for c in chunks]
    assert len(bodies) == len(set(c.content_hash for c in chunks))


def test_chunk_metadata_is_pinecone_compatible() -> None:
    doc = parse_markdown(DOC, fallback_id="x", source="guia.md")
    meta = MarkdownChunker(chunk_size=300, chunk_overlap=0, min_chunk_chars=10).split(doc)[0].metadata()
    for value in meta.values():
        assert isinstance(value, (str, int, float, bool)) or (
            isinstance(value, list) and all(isinstance(v, str) for v in value)
        )


def test_content_hash_normalizes_whitespace() -> None:
    assert content_hash("hola   mundo\n") == content_hash("hola mundo")


def test_sparse_vector_validation() -> None:
    with pytest.raises(ValueError, match="mismo largo"):
        SparseVector(indices=[1, 2], values=[0.1])
    with pytest.raises(ValueError, match="repetidos"):
        SparseVector(indices=[1, 1], values=[0.1, 0.2])
    with pytest.raises(ValueError, match="2\\^32"):
        SparseVector(indices=[2**32], values=[0.1])


def test_metadata_filter_translation() -> None:
    assert MetadataFilter().to_pinecone() is None
    assert MetadataFilter(categories=["retrieval"]).to_pinecone() == {"category": {"$in": ["retrieval"]}}
    both = MetadataFilter(doc_ids=["a"], categories=["b"]).to_pinecone()
    assert both == {"$and": [{"doc_id": {"$in": ["a"]}}, {"category": {"$in": ["b"]}}]}
    with pytest.raises(ValueError, match="doc_ids"):
        MetadataFilter(doc_ids=["NO VALIDO"])
    with pytest.raises(ValueError, match="extra"):
        MetadataFilter.model_validate({"$where": "x"})
