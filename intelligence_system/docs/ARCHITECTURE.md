# Arquitectura, inventario y gap analysis

Documento de la Fase 0 (discovery) y de las decisiones que salieron de ella. Las decisiones no
triviales tienen su ADR en [adr/](adr/).

## 1. Inventario de las pre-entregas

Se revisaron las pre-entregas presentes en el repo (copias de trabajo del 2026-11-05). El
enunciado habla de "Módulo 6 (RAG)" y "Módulo 7 (agentes)"; en el repo esa lógica vive en las
pre-entregas 3–4 (RAG) y 5 (agentes), más los ejercicios de supervisor y estado compartido.

| Origen | Contenido | Estado encontrado | Qué se reutiliza | Qué se refactoriza |
| --- | --- | --- | --- | --- |
| Pre-entrega 1 | Cliente LLM async unificado: ABC `BaseLLMClient`, Factory/manager, OpenAI + Anthropic, streaming, excepciones | Funciona; envía `temperature`/`top_p` (rechazados por anthropic 2.5.0) | ABC, jerarquía de excepciones, `ChatMessage`/`ModelConfig`/`ModelResponse`, streaming | + tool calling neutral, + salida estructurada con bucle de reparación, + reintentos tenacity, + semáforo; sin `temperature` en Anthropic |
| Pre-entrega 2 | LCEL async (`prompt | model | parser`), `RunnableParallel` | Funciona | Conceptos | Sin código: la orquestación pasa a LangGraph |
| Pre-entrega 3 | RAG con ChromaDB + `RAGResponse {answer, references, has_answer}` | Funciona (embeddings HF locales) | Contrato anti-alucinación | Pasa a citas validadas en código (`grounded`) |
| Pre-entrega 4 | RAG Pinecone: corpus de 7 docs, `ensure_index`, chunking por tokens, BM25 en memoria + RRF, `evaluate.py` + golden set | Funciona con `pinecone` 5–9; `EnsembleRetriever` roto en LangChain 1.x | Corpus, golden set, métricas P@k/R@k, RRF, idea de IDs deterministas | Async (`PineconeAsyncio`), índice híbrido nativo, BM25 sin pickle local, ingesta idempotente por hash |
| Pre-entrega 5 | Agente ReAct (LangGraph, `MessagesState`, `ToolNode`, `SqliteSaver`), tools con docstrings, traza JSON | Funciona; `SqliteSaver` sync no sirve en async | Patrón ReAct, docstrings "cuándo usar", errores como datos, `recursion_limit`, checkpointer inyectado | `AsyncRedisSaver`; ReAct dentro del worker analyst sobre el cliente unificado; trazas a Phoenix |
| Ejercicio supervisor | Supervisor con `with_structured_output(SupervisorDecision)` | Funciona | Ruteo con salida estructurada | `Command(goto=...)` + reglas deterministas |
| Ejercicio estado compartido | Artefactos tipados con aislamiento de escritura por nodo | Funciona | Aislamiento de escritura | `NODE_OWNERSHIP` verificado en runtime |

## 2. Deuda detectada y cómo se resolvió

| Deuda | Dónde | Resolución |
| --- | --- | --- |
| `temperature`/`top_p` enviados a Anthropic | pre-entrega 1 | El SDK 2.5.0 ya no tiene esos parámetros (verificado con `inspect.signature`): no se envían |
| Clientes sync en rutas async (`Pinecone`, `SqliteSaver`, `HuggingFaceEmbeddings`) | pre-entregas 4 y 5 | `PineconeAsyncio`/`IndexAsyncio`, `AsyncRedisSaver`, Pinecone Inference async |
| `tiktoken` descarga sus tablas con HTTP síncrono en el primer uso | pre-entrega 4 | Chunking por caracteres (ADR-0005) |
| Hard-coding (`CATEGORY_MAP`, modelos, índices) | pre-entregas 3–5 | `Settings` + front matter en cada documento |
| BM25 con pickle local + `EnsembleRetriever` roto | pre-entrega 4 | Híbrido nativo en Pinecone + estadísticas BM25 en Redis (ADR-0004) |
| Sin trazas ni ids de correlación | todas | OpenInference + Phoenix + logs JSON con `request_id`/`thread_id`/`trace_id` |
| Sin tests automatizados | todas | 131 unitarios + 4 de integración |

## 3. Versiones verificadas (PyPI, 2026-11-05)

Instaladas y fijadas en `uv.lock`; las APIs se verificaron contra el código instalado, no de memoria.

| Paquete | Versión | Verificación clave |
| --- | --- | --- |
| langgraph | 1.2.20 | `Command(goto)`, `add_node(destinations=...)`, `get_stream_writer`, `astream(stream_mode=[...])` |
| langgraph-checkpoint-redis | 0.6.0 | `AsyncRedisSaver.from_conn_string` (async CM que corre `asetup`), TTL; requiere Redis 8+ |
| langgraph-checkpoint | 4.3.1 | Serializador permisivo por defecto; el estado usa solo tipos JSON |
| langchain-core / text-splitters | 1.7.6 / 1.1.3 | `StructuredTool(args_schema=...)`, splitters markdown + recursivo |
| pinecone | 10.3.0 | `PineconeAsyncio` (= `AsyncPinecone`), `indexes.create(schema=...)`, campos reservados `_values`/`_sparse_values` para híbrido |
| anthropic | 2.5.0 | `messages.create` sin `temperature`/`top_p`; `tools` + `tool_choice`; `models.retrieve` |
| openai | 3.46.0 | `chat.completions.create` con `tools`, `max_completion_tokens` |
| arize-phoenix-otel | 0.17.2 | `register(endpoint, project_name, batch, protocol)` |
| openinference-instrumentation-langchain / anthropic / openai | 0.1.82 / 3.3.1 / 0.1.69 | `get_current_span()` para anidar spans bajo el nodo |
| fastapi / starlette | 0.162.0 / 1.10.0 | Telemetría nativa configurable (`telemetry={...}`) |
| pydantic / pydantic-settings | 2.14.3 / 2.16.2 | `env_ignore_empty` |
| Imágenes Docker | `redis:8.10.3`, `arizephoenix/phoenix:version-21.0.0`, `python:3.12.16-slim-trixie`, `ghcr.io/astral-sh/uv:0.11.32` | Tags verificados en official-images / workflow de release de Phoenix / tags de uv. El tag de Phoenix sin sufijo es la variante root (distroless `python3-debian13`, amd64/arm64): tiene `/usr/bin/python3` para el healthcheck y escribe en el volumen |
| Modelos Anthropic | `claude-haiku-5-5` (default), `claude-sonnet-5-5` (opcional para el supervisor) | Activos en la página de deprecaciones de Anthropic (2026-11-05); `claude-haiku-4-5` sigue activo pero su fecha tentativa de retiro ya pasó |

## 4. Gap analysis contra los 7 puntos

| Punto | Antes (pre-entregas) | Ahora | Estado |
| --- | --- | --- | --- |
| 1. Integración | Proyectos sueltos | Un paquete `intelligence_system` con LLM, RAG, agentes y API | ✅ |
| 2. Refactor async | Mezcla sync/async | Todo async; `asyncio.to_thread` solo para chunking (CPU) | ✅ |
| 3. Persistencia | `SqliteSaver` sync | `AsyncRedisSaver` + AOF; probado con reinicio contra Redis 8.10 real | ✅ |
| 4. Pydantic v2 | Parcial | API, tools (`args_schema` y salidas), supervisor, RAG, Settings | ✅ |
| 5. Observabilidad | Traza JSON local | Phoenix + OpenInference; 6 escenarios con link a cada traza | ✅ (faltan las capturas, se toman al correr) |
| 6. README + Mermaid | No | Grafo generado con `draw_mermaid()` + documentación técnica | ✅ |
| 7. Un comando | No | `./run.sh` | ✅ |

## 5. Caso de uso y corpus

Asistente técnico de AI Engineering sobre el corpus de la pre-entrega 4 (7 documentos en
español). El dominio de las tools del analyst se eligió para que combine naturalmente con ese
corpus (cálculos de chunking, métricas de retrieval, RRF, memoria de índices, similitud), en
lugar de las tools de clientes y pedidos de la pre-entrega 5, que pertenecen a otro dominio
(ADR-0012).

## 6. Riesgos

| Riesgo | Mitigación |
| --- | --- |
| Cuotas del plan gratuito de Pinecone (inference / rerank) | Reingesta sin costo (dedup por hash); rerank apagado por defecto |
| Índice viejo con el mismo nombre (cosine, 384 dims) | `ensure_ready` valida campos, dimensión y métrica y explica qué hacer |
| Cambios de API en SDKs que se mueven rápido | Versiones fijadas en `uv.lock`; tests contra dobles con las firmas reales |
| Ruteo del LLM incorrecto | Reglas deterministas (presupuesto, anti-bucle, fallback) + `AGENT_RECURSION_LIMIT` |
| Latencia y costo por pregunta (3–7 llamadas al LLM) | Haiku por defecto, contexto mínimo por rol, grader desactivable |
| Varias réplicas de la API | Lock por thread en proceso: documentado como limitación |
