# Smoke de concurrencia (`make load`)

Corrido el 2026-11-06 07:22 UTC: **5 peticiones simultáneas** a `POST /v1/chat`, cada una en un `thread_id` distinto (el lock por thread no serializa entre ellas).

| Métrica | Valor |
| --- | --- |
| Completadas (200) | 5/5 |
| Latencia p50 | 7.80 s |
| Latencia p95 | 9.07 s |
| Latencia máx. | 9.07 s |
| Suma de latencias | 39.70 s |
| Wall time (todas en paralelo) | 9.08 s |
| Paralelismo efectivo (suma / wall) | 4.37× |

Lectura: si el servidor serializara las peticiones, el wall time se acercaría a la suma (paralelismo ≈ 1×); con I/O async se acerca a la latencia máxima individual.

| # | thread_id | Pregunta | Status | grounded | citas | Latencia | Traza |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `load-307a0c06` | ¿Para qué sirve asyncio.Semaphore cuando llamo a un LLM? | 200 | True | 1 | 7.68 s | [`f091c68517fe…`](http://localhost:6006/redirects/traces/f091c68517fe469b576be9651b83151b) |
| 2 | `load-27fb86b7` | ¿Qué es BM25 y qué limitación tiene? | 200 | True | 3 | 7.80 s | [`63440965c448…`](http://localhost:6006/redirects/traces/63440965c4482d0720e4fbea6031eddc) |
| 3 | `load-4800f3a0` | ¿Qué diferencia hay entre get y query en una vector database? | 200 | True | 2 | 9.07 s | [`19e3b3299426…`](http://localhost:6006/redirects/traces/19e3b32994261e0a5340f1072bcfc291) |
| 4 | `load-415a94da` | ¿Por qué la similitud coseno es la métrica estándar en RAG? | 200 | True | 4 | 8.25 s | [`6b1a19e3fadb…`](http://localhost:6006/redirects/traces/6b1a19e3fadb8b480cbc636044ad4d84) |
| 5 | `load-8c0fc49c` | ¿Para qué sirven los namespaces en Pinecone? | 200 | True | 2 | 6.89 s | [`27f142ebcf5e…`](http://localhost:6006/redirects/traces/27f142ebcf5e7a4e98662366612cb982) |
