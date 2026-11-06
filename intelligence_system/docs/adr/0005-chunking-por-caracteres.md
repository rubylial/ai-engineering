# ADR-0005 · Chunking markdown-aware medido en caracteres

- Estado: aceptada · Fecha: 2026-11-05

## Contexto
La pre-entrega 4 medía chunks en tokens con `tiktoken`, que descarga sus tablas BPE con una
llamada HTTP síncrona en el primer uso: dentro de un endpoint async bloquearía el event loop y
fallaría sin red.

## Decisión
`MarkdownHeaderTextSplitter` (conserva la sección como metadato para citar) + recursivo con
overlap, medido en caracteres (`RAG_CHUNK_SIZE=1500`, `RAG_CHUNK_OVERLAP=150`, ≈375/37 tokens).
El chunking corre en `asyncio.to_thread` porque es CPU puro. Se descartan fragmentos de menos
de `RAG_MIN_CHUNK_CHARS` (encabezados sueltos) y duplicados dentro del documento.

## Consecuencias
Tamaños aproximados en tokens (≈4 caracteres por token en español). Suficiente para un corpus
de documentación; un modelo de tokenización local podría reemplazarlo si hiciera falta.
