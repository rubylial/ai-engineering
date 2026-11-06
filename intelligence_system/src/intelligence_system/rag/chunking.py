"""Chunking consciente de markdown, con IDs deterministas por contenido.

1. `MarkdownHeaderTextSplitter` corta por secciones (#, ##, ###) y conserva la
   ruta de encabezados como metadato `section` (útil para citar).
2. `RecursiveCharacterTextSplitter` sub-divide secciones largas con overlap.

El tamaño se mide en caracteres (≈4 caracteres por token en español). La
pre-entrega 4 medía en tokens con tiktoken, pero tiktoken descarga sus tablas
en el primer uso con una llamada HTTP síncrona; en un servicio async eso
bloquearía el loop (ver ADR-0004).

Cada chunk se identifica como `<doc_id>#<sha256(texto)[:16]>`: el mismo texto
produce siempre el mismo id, lo que hace la ingesta idempotente y permite
deduplicar sin consultar el contenido.
"""

from __future__ import annotations

from collections.abc import Sequence

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from intelligence_system.rag.schemas import Chunk, Document, content_hash

_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]


def chunk_id_for(doc_id: str, text_hash: str) -> str:
    return f"{doc_id}#{text_hash[:16]}"


def chunk_prefix(doc_id: str) -> str:
    return f"{doc_id}#"


class MarkdownChunker:
    """Convierte `Document` en `Chunk` (CPU puro: se ejecuta en `asyncio.to_thread`)."""

    def __init__(self, *, chunk_size: int, chunk_overlap: int, min_chunk_chars: int) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap debe ser menor que chunk_size")
        self._min_chars = min_chunk_chars
        self._headers = MarkdownHeaderTextSplitter(headers_to_split_on=_HEADERS, strip_headers=False)
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def split(self, document: Document) -> list[Chunk]:
        chunks: list[Chunk] = []
        seen: set[str] = set()
        for section in self._headers.split_text(document.text):
            path = " › ".join(section.metadata[k] for k, _ in _iter_levels() if k in section.metadata)
            for piece in self._splitter.split_text(section.page_content):
                text = piece.strip()
                if len(text) < self._min_chars:
                    continue  # solo un encabezado suelto: no aporta evidencia
                text_hash = content_hash(text)
                if text_hash in seen:
                    continue  # dedup dentro del documento
                seen.add(text_hash)
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id_for(document.doc_id, text_hash),
                        doc_id=document.doc_id,
                        chunk_index=len(chunks),
                        text=text,
                        title=document.title,
                        section=path or document.title,
                        source=document.source,
                        category=document.category,
                        tags=document.tags,
                        content_hash=text_hash,
                    )
                )
        return chunks

    def split_many(self, documents: Sequence[Document]) -> list[Chunk]:
        return [chunk for doc in documents for chunk in self.split(doc)]


def _iter_levels() -> list[tuple[str, str]]:
    return [(key, marker) for marker, key in _HEADERS]
