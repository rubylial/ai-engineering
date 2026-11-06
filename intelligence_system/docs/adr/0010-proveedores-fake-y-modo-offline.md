# ADR-0010 · Proveedores falsos para tests y modo offline

- Estado: aceptada · Fecha: 2026-11-05

## Decisión
El paquete incluye `LLM_PROVIDER=fake` (responder heurístico), `EMBEDDINGS_PROVIDER=fake`
(hashing de términos) y `VECTOR_STORE=memory` (misma semántica de score que Pinecone). Sirven
como dobles de los tests unitarios (sin red, sin costo) y para validar el cableado completo
(API, grafo, Redis, Phoenix) sin claves.

## Consecuencias
Las respuestas del modo offline son extractivas y no representan la calidad del sistema real;
la evidencia de la entrega se genera con Anthropic + Pinecone.
