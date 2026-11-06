# ADR-0004 · Retrieval híbrido nativo en Pinecone con BM25 propio

- Estado: aceptada · Fecha: 2026-11-05

## Contexto
La pre-entrega 4 combinaba Pinecone denso con un BM25 en memoria (pickle local) vía
`EnsembleRetriever`/RRF. Eso no escala, deja estado local en el contenedor y el
`EnsembleRetriever` quedó roto en LangChain 1.x. Con `pinecone==10.3.0` (API 2026-07) un índice
denso con `dotproduct` NO acepta vectores dispersos salvo que el campo sparse se declare al
crearlo (documentado en el propio SDK).

## Decisión
- Crear el índice con `indexes.create(schema=...)` declarando exactamente los campos reservados
  `_values` (dense_vector, dotproduct) y `_sparse_values` (sparse_vector). Según el SDK, un
  schema con solo esos campos es servido de forma nativa por la API de vectores
  (`upsert`/`query` con `values` + `sparse_values`).
- Combinación convexa en una sola consulta: denso × `alpha` y disperso × `(1 − alpha)`.
- BM25 propio sin dependencias: TF normalizado en el documento, IDF normalizado en la consulta;
  índices por hash estable (uint32).
- Estadísticas del corpus (N, longitud total, document frequency) en Redis, idempotentes por
  `chunk_id` (SET + MULTI/EXEC), compartidas entre API e ingesta.
- `ensure_ready` valida el schema de un índice existente y explica cómo resolver un conflicto.

## Alternativas descartadas
- Dos índices (denso + disperso) y fusión en cliente: dos consultas por búsqueda.
- Documents API con full-text search de Pinecone: no permite combinar en una consulta un
  clause vectorial con uno de texto, y es una API nueva sin forma de probarla acá.
- `pinecone-text`: dependencia sin mantenimiento activo.

## Consecuencias
Sin estado local: cualquier réplica de la API consulta igual. Si Redis se vacía (`make clean`)
y los vectores siguen en Pinecone, la reingesta re-registra los chunks existentes (operación
idempotente por `chunk_id`) y reconstruye el IDF sin volver a embeber nada.
