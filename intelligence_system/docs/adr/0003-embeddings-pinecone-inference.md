# ADR-0003 · Embeddings con Pinecone Inference (multilingual-e5-large)

- Estado: aceptada · Fecha: 2026-11-05

## Contexto
Solo hay claves de Anthropic y Pinecone (no OpenAI). La pre-entrega 4 usaba
`sentence-transformers` local (384 dims): síncrono, exige torch (+GB en la imagen) y bloquea el
event loop si no se envuelve.

## Decisión
Usar Pinecone Inference `multilingual-e5-large` (1024 dims) con `PineconeAsyncio.inference.embed`:
async nativo, multilingüe (corpus en español), misma clave que el vector store. Lotes de
`EMBEDDINGS_BATCH_SIZE` en paralelo acotado por semáforo; `input_type=passage|query`.

## Alternativas descartadas
- sentence-transformers local con `asyncio.to_thread`: válido, pero agrega torch a la imagen.
- OpenAI embeddings: no hay clave.

## Consecuencias
El índice debe tener dimensión 1024; un índice de la pre-entrega 4 (384) no sirve: se usa otro
nombre (`PINECONE_INDEX_NAME=intelligence-system`). Consume cuota de inference solo al embeber
chunks nuevos.
