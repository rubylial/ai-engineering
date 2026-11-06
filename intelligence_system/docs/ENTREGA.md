# Reporte de entrega — The Intelligence System

Fecha: 2026-11-06 · Carpeta `intelligence_system/` del repo del curso · Verificado de punta a
punta con Anthropic, Pinecone y Redis reales en WSL2 + Docker Desktop.

## 1. Resumen ejecutivo

Sistema único que integra las pre-entregas: cliente LLM async unificado (pre-entrega 1) extendido
con tool calling y salida estructurada; RAG híbrido en Pinecone (denso e5 + BM25 en una sola
consulta) con ingesta idempotente; Supervisor LangGraph con researcher, analyst y writer
coordinados con `Command(goto=...)`; persistencia con `AsyncRedisSaver`; API FastAPI documentada
(6 endpoints, `X-API-Key`, problem+json, SSE) y una traza por request en Arize Phoenix con
`request_id` y `thread_id`. Se levanta con `./run.sh`.

### Qué se verificó, y dónde

| Verificación | Resultado | Evidencia |
| --- | --- | --- |
| Build de la imagen (`python:3.12.16-slim-trixie`, multi-stage con uv 0.11.32) | OK | Salida de `./run.sh` |
| `./run.sh`: 3 contenedores healthy, `GET /ready` 200 con `redis`, `vector_store` (55 vectores), `llm` (`claude-haiku-5-5`) y `agent` OK | OK | `curl /ready` |
| Ingesta idempotente | 7 documentos → 55 chunks en 4.3 s; segunda corrida: `chunks_upserted=0`, `chunks_unchanged=55` | `POST /v1/ingest` ×2 |
| Consultas manuales | RAG simple (4 citas); multi-paso researcher → analyst (`estimar_chunks` → 27 chunks, cita de documento y de cálculo); continuación en el mismo `thread_id` | Respuestas JSON con `steps`, `citations`, `trace_id` |
| `make scenarios` | 6/6 en dos corridas independientes | `docs/traces/traces.md`, `scenario_results.json` |
| Persistencia ante reinicio | `docker compose restart api` → `GET /v1/threads/persist-test` devolvió los 2 mensajes, `last_citations` y `checkpoint_id` | Sección 8 del runbook |
| Calidad | `make test`: 131 passed, 4 skipped · `make lint` limpio · `make graph` sin diferencias | — |
| Evaluación del retriever (golden v1, 5 preguntas) | híbrido P@5 0.680 · denso 0.640 · BM25 0.680; R@5 y MRR 1.000 en los tres | `docs/traces/evaluation.txt` (corrida inicial) |

### Un bug real encontrado por la evaluación, y corregido

La variante "solo BM25" (`alpha=0.0`) fallaba contra Pinecone con
`400 Cannot query index with dense 'vector_type' with only sparse vector`. Causa: `hybrid_scale`
omitía el vector denso cuando alpha era 0, y el test unitario, que corre contra el store en
memoria (que sí acepta `dense=None`), fijaba exactamente ese comportamiento. El índice híbrido de
Pinecone exige un vector denso en toda consulta. Corrección: en alpha=0.0 se manda el denso
escalado a cero (aporta 0 al dotproduct → el score queda puro BM25); test actualizado. No
afectaba a la API (usa alpha=0.7), pero sí a la evaluación, y es un ejemplo de por qué los dobles
de prueba no reemplazan una corrida contra el backend real.

### Incidencias operativas durante el despliegue (ya documentadas en el runbook §9)

- Puertos 6006 y 8000 ocupados por el compose de otra pre-entrega → `docker compose down` del
  otro proyecto.
- Un primer `up` fallido por el puerto dejó el contenedor `api` creado sin red ni puertos; el
  siguiente `up` lo reutilizó ("healthy" pero inalcanzable y sin DNS a `redis`) →
  `docker compose down && ./run.sh`.
- Docker Hub devolvió `520` al resolver `docker/dockerfile:1.7` → transitorio;
  `docker compose up -d --wait` sin `--build` usa la imagen ya construida.
- El descompresor de WSL omitió `.env.example` → recuperado; es parte del entregable y lo exige
  un test.

### Lo que queda sin verificar

| Punto | Estado | Mitigación |
| --- | --- | --- |
| Que Phoenix muestre **costo** para `claude-haiku-5-5` (depende de que el precio esté en su tabla) | Incierto (~40 % de que aparezca solo) | Cargar el precio en *Settings → Models*, o correr los escenarios con `LLM_MODEL_WORKER=claude-haiku-4-5`, o calcular desde los tokens de la traza |
| Capturas de Phoenix (11 archivos) | Pendiente, manual | Instrucciones exactas en `docs/traces/traces.md` |
| `make eval` con el golden set v2 (16 preguntas tipificadas) y `make load` (concurrencia) | Pendiente de correr con el stack arriba | `make evidence` los genera juntos |

## 2. Punto → estado → evidencia

| Punto | Estado | Evidencia |
| --- | --- | --- |
| 1. Integración RAG + agentes + cliente unificado | ✅ | `src/intelligence_system/{llm,rag,agents}`; mapa de reutilización en `docs/ARCHITECTURE.md` §1 |
| 2. Todo el I/O async | ✅ | `AsyncAnthropic`, `AsyncOpenAI`, `PineconeAsyncio`, `redis.asyncio`, `anyio.Path`; `asyncio.to_thread` solo para chunking (CPU); regla `ASYNC` de ruff; sin `requests` ni `time.sleep`; paralelismo medido con `make load` (`docs/traces/load.md`) |
| 3. Checkpointer async que sobrevive reinicios | ✅ verificado con reinicio real | `agents/checkpointer.py`; `tests/integration/test_redis_persistence.py`; `docker compose restart api` + `GET /v1/threads/{id}` |
| 4. Pydantic v2 en API y tools | ✅ | `api/schemas`, `agents/schemas.py`, `agents/tools/*` (`args_schema` y salidas), `rag/schemas.py`, `config.py` |
| 5. ≥5 escenarios con trazas en Phoenix | ✅ 6/6 con el LLM real; capturas pendientes | `scripts/run_scenarios.py`, `docs/traces/traces.md`, `scenario_results.json` (con `trace_id` reales) |
| 6. README con Mermaid + docs técnicas | ✅ | `README.md` (grafo generado con `draw_mermaid()`, sección *Resultados medidos*), `docs/ARCHITECTURE.md`, 12 ADRs, `docs/RUNBOOK.md` |
| 7. Despliegue con un comando | ✅ build y arranque reales | `./run.sh`; `docker-compose.yml`; imagen `intelligence-system-api:1.0.0` construida en WSL |

## 3. Rúbrica → ubicación → nota estimada

La rúbrica oficial no venía en el enunciado. Uso como rúbrica los 7 puntos y las reglas duras; las
notas son mi estimación, no una calificación.

| Criterio | Dónde | Nota estimada |
| --- | --- | --- |
| Integración de pre-entregas | `llm/`, `rag/`, `agents/`, `ARCHITECTURE.md` | 9/10 |
| Async de punta a punta | `llm/providers`, `rag/vector_store.py`, `agents/checkpointer.py`, `docs/traces/load.md` | 9.5/10 |
| Persistencia ante reinicios | `agents/checkpointer.py`, test de integración, reinicio real verificado | 9.5/10 |
| Pydantic v2 en cada borde | `api/schemas`, `agents/tools`, `config.py` | 9.5/10 |
| Observabilidad y escenarios | `observability/`, `scripts/run_scenarios.py`, `docs/traces/` | 8.5/10 hoy (6/6 real, `trace_id` reales); 9.5/10 con las capturas |
| Evaluación del RAG | `data/golden_set.json` (16 tipificadas), `scripts/evaluate_retrieval.py`, `docs/traces/evaluation.md` | 9/10 (métricas a nivel documento sobre 7 docs: comparativas, no absolutas) |
| Documentación | `README.md`, `docs/` | 9.5/10 |
| Despliegue en un comando | `run.sh`, `docker-compose.yml`, `Dockerfile` | 9.5/10 |
| Calidad (tests, lint, tipos) | `tests/` (131 + 4), `pyproject.toml` (ruff con `ANN`), bug real encontrado y cubierto por test | 9.5/10 |
| Reglas duras (zero hard-coding, sin clientes sync, APIs verificadas) | `config.py`, `.env.example`, ADRs | 9/10 |

## 4. Comandos de verificación

```bash
make install && make test && make lint     # 131 passed, 4 skipped · ruff limpio
cp .env.example .env                       # completar ANTHROPIC_API_KEY, PINECONE_API_KEY, API_KEY
./run.sh                                   # build + salud + /ready + ingesta (55 chunks)
curl -s localhost:8000/ready | python3 -m json.tool
make test-integration                      # 3 passed contra el Redis del compose (+1 con PINECONE_API_KEY)
make evidence                              # scenarios 6/6 + eval (3 variantes, por tipo) + load (p50/p95)
make logs-trace REQ=req_…                  # correlación request_id ↔ trace_id en los logs JSON
```

Detalle de cada endpoint con `curl` en `docs/RUNBOOK.md` §5.

## 5. Commits atómicos propuestos

Los comandos `git add` exactos (con la ruta `intelligence_system/`) están en `docs/RUNBOOK.md` §10.

1. `chore(intelligence-system): scaffold con uv, pyproject y lockfile`
2. `feat(config): settings centralizados con pydantic-settings y errores tipados`
3. `feat(observability): logs JSON y trazas en Phoenix con request_id y thread_id`
4. `feat(llm): cliente async unificado con tool calling, reintentos y salida estructurada`
5. `feat(rag): ingesta idempotente y retriever híbrido denso + BM25 en Pinecone`
6. `feat(agents): supervisor LangGraph con researcher, analyst, writer y checkpointer Redis`
7. `feat(api): endpoints FastAPI con X-API-Key, problem+json y streaming SSE`
8. `feat(scripts): escenarios de evidencia con latencia, export del grafo e ingesta`
9. `feat(eval): golden set tipificado, informe P@k/R@k/MRR por tipo y smoke de concurrencia`
10. `test: unitarios sin red, informe de evaluación e integración contra Redis y Pinecone`
11. `build(docker): imagen multi-stage, compose, run.sh y targets evidence/load/logs-trace`
12. `docs: README con grafo Mermaid y resultados medidos, arquitectura, ADRs, runbook y reporte`
13. `ci(intelligence-system): lint y tests unitarios en cada push` (opcional, en la raíz del repo)
14. `docs(traces): escenarios, evaluación, concurrencia y capturas de Phoenix` (después de `make evidence` y las capturas)

## 6. Riesgos, deuda y pendientes

| Tipo | Detalle | Mitigación o siguiente paso |
| --- | --- | --- |
| Riesgo | Costo por ejecución en Phoenix para `claude-haiku-5-5` | Tres alternativas en el README (*Qué capturar*) |
| Riesgo | Ruteo del LLM real distinto al esperado en algún escenario | No ocurrió en dos corridas; reglas deterministas de respaldo; repetir cuesta centavos |
| Decisión | Modelo por defecto `claude-haiku-5-5` | Cambiable con `LLM_MODEL_*` sin tocar código |
| Deuda | Lock por thread dentro del proceso | Con varias réplicas haría falta un lock distribuido (Redis) |
| Deuda | BM25 sin stemming | El componente denso cubre la morfología |
| Deuda | Golden set de 16 preguntas sobre 7 documentos | Métricas comparativas entre variantes, no absolutas; ampliar el corpus las volvería más informativas |
| Deuda | Phoenix sin autenticación, una sola `API_KEY` estática, sin rate limiting | Aceptable en local; fuera de alcance |
| Fuera de alcance | CD, despliegue en nube, HITL | Documentado en el README; la CI del repo cubre lint + tests |
| Pendiente (vos) | Capturas (11, en `docs/traces/`), `make evidence`, commits | `docs/traces/traces.md` y `docs/RUNBOOK.md` §10 |

Costos de las operaciones que faltan (todas en centavos de dólar):

| Comando | Costo aproximado |
| --- | --- |
| `make scenarios` | ~25 llamadas a `claude-haiku-5-5` + menos de 10 embeddings |
| `make eval` | 48 embeddings de consulta (3 variantes × 16 preguntas); sin LLM |
| `make load` | 5 turnos completos (~15–25 llamadas a Haiku + 5 embeddings) |
| `PINECONE_API_KEY=… make test-integration` | ~3 embeddings; usa el índice existente |

## 7. Capturas de Phoenix

Las instrucciones exactas (11 archivos, dónde hacer click y qué tiene que verse) están en
`docs/traces/traces.md` y en `docs/RUNBOOK.md` §6. Se guardan en `docs/traces/` con los nombres
`01-lista-de-trazas.png` … `11-logs-correlacion.png`. Mínimo recomendable si el tiempo aprieta:
`01` (lista), `04` (multi-paso con los dos workers y las tools), `07` (abstención sin evidencia),
`08` (sesión con dos turnos), `09` (error con `trace_id`) y `11` (logs correlacionados, que
ahora salen con `make logs-trace REQ=<request_id del escenario 2>`).
