# Evidencia de observabilidad (Arize Phoenix)

Seis escenarios ejecutados contra el sistema real (Anthropic + Pinecone + Redis) con
`make scenarios`. La tabla de abajo la reescribe el script en cada corrida, con el `trace_id` y
el link directo a cada traza (`http://localhost:6006/redirects/traces/<trace_id>`). El detalle
completo (respuesta, citas, cuerpos de error) queda en `scenario_results.json`.

## Resultados

<!-- SCENARIOS:START -->
_Última corrida: 2026-11-06 07:21 UTC_

| # | Escenario | Input | Esperado | Obtenido | OK | Latencia | trace_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Pregunta RAG simple | ¿Para qué sirve asyncio.Semaphore cuando llamo a un LLM? | grounded=true, ≥1 cita de documento, pasa por researcher | grounded=True, citas=2, pasos=supervisor→researcher→supervisor→writer | ✅ | 7.8 s | [`df0b74ae59c0…`](http://localhost:6006/redirects/traces/df0b74ae59c07c6040cb1fb979127ae1) |
| 2 | Tarea multi-paso (≥2 workers) | Según la documentación, ¿con qué tamaño de chunk y overlap conviene empezar? Calculá cuántos chunks salen de un documento de 12000 tokens con esa configuración. | researcher + analyst, citas de documento y de cálculo | pasos=supervisor→researcher→supervisor→analyst→supervisor→writer, tipos de cita=['document', 'tool'] | ✅ | 13.6 s | [`4ee3c12a8d99…`](http://localhost:6006/redirects/traces/4ee3c12a8d9966daf9fab0429755271e) |
| 3 | Pregunta sin evidencia | ¿Cuál es la capital de Australia? | grounded=false, 0 citas, responde que no sabe | grounded=False, citas=0 | ✅ | 1.4 s | [`2eff937629e5…`](http://localhost:6006/redirects/traces/2eff937629e5635900d91ebb2f5be62a) |
| 4 | Continuación por thread_id | ¿Qué es BM25? → ¿Y qué limitación tiene frente a la búsqueda semántica? | 2º turno resuelto con el contexto; 4 mensajes | mensajes en el thread=4, grounded 2º turno=True | ✅ | 10.4 s | [`f7fae6e6f115…`](http://localhost:6006/redirects/traces/f7fae6e6f115f049646441e4190fbb71) |
| 5 | Error controlado / timeout | Explicá RRF (deadline_s=0.001) + mensaje vacío | 504 problem+json con request_id y trace_id; 422 para el mensaje vacío | 504 /errors/graph-timeout; mensaje vacío → 422 | ✅ | 0.0 s | [`39e7fad29cbc…`](http://localhost:6006/redirects/traces/39e7fad29cbc1f50438e779b29893c1b) |
| 6 | Streaming SSE (extra) | ¿Qué es Reciprocal Rank Fusion? | eventos step → token → citation → done | eventos=43 (step=4, token=36, citation=2), último=done | ✅ | 6.7 s | [`3a94d1cd8d7b…`](http://localhost:6006/redirects/traces/3a94d1cd8d7bbc705dcd2c5ade07120c) |
<!-- SCENARIOS:END -->

## Qué atributos tiene cada traza

| Span | Kind | Atributos clave |
| --- | --- | --- |
| `chat.turn` / `chat.stream` (raíz) | CHAIN | `request_id`, `thread_id`, `session.id`, `input.value`, `output.value`, `grounded`, `citations.count` |
| `intelligence_system` → `supervisor`/`researcher`/`analyst`/`writer` | CHAIN | Spans de LangGraph (OpenInference) con `session.id` y `metadata` |
| `agent.<worker>` | AGENT | `worker`, `agent.step`, `agent.status`, `agent.goto`, `request_id`, `thread_id` |
| `rag.hybrid_search` | RETRIEVER | `retrieval.documents.*` (id, contenido, score), `rag.alpha`, `rag.results` |
| `search_knowledge_base`, `estimar_chunks`, … | TOOL | Argumentos y salida de la tool |
| `messages.create` / `messages.stream` (Anthropic) | LLM | Prompt, tool forzada, tokens de entrada/salida |
| `rag.embed_query` | EMBEDDING | `embedding.model_name` |

## Instrucciones de captura

Prerequisito: `./run.sh` terminó bien y `make scenarios` mostró 6/6. Abrí <http://localhost:6006>,
en el menú lateral **Tracing** → proyecto **intelligence-system**. Guardá cada captura en esta
carpeta (`docs/traces/`) con el nombre indicado. Los links de la tabla de arriba abren cada
traza directamente.

| Archivo | Dónde | Qué tiene que verse |
| --- | --- | --- |
| `01-lista-de-trazas.png` | Pestaña **Traces** del proyecto | Las trazas de los 6 escenarios con latencia y estado; la del escenario 5 con el ícono de error rojo |
| `02-esc1-arbol.png` | Link del escenario 1 | Árbol completo `chat.turn → intelligence_system → supervisor → researcher → writer`, y a la derecha (pestaña **Info**) **Input**, **Output** con las citas `[n]` y **Metadata** con `request_id` y `thread_id` |
| `03-esc1-retriever.png` | Mismo trace, click en `rag.hybrid_search` | Los documentos recuperados con id, contenido y score |
| `04-esc2-multipaso.png` | Link del escenario 2 | Árbol con **researcher y analyst**: `search_knowledge_base` y `estimar_chunks` como spans TOOL; Output con citas de documento y de cálculo |
| `05-esc2-llm.png` | Escenario 2, expandí `agent.supervisor` y click en `messages.create` | Prompt del supervisor, la tool `supervisor_decision` y el conteo de tokens |
| `06-esc2-atributos-worker.png` | Escenario 2, click en `agent.analyst` → pestaña **Attributes** | `worker=analyst`, `agent.step`, `agent.status`, `request_id`, `thread_id` |
| `07-esc3-sin-evidencia.png` | Link del escenario 3 | Recorrido corto `supervisor → writer` y Output con la abstención ("No encontré evidencia suficiente…") |
| `08-esc4-sesion.png` | Link del escenario 4 → botón **View Session** | La sesión `esc4-…` con los **2 turnos** (input y output de cada uno) |
| `09-esc5-error.png` | Link del escenario 5 | `chat.turn` en estado **ERROR**; pestaña **Events** con la excepción `GraphTimeoutError` |
| `10-esc6-stream.png` | Link del escenario 6 | Raíz `chat.stream` con Input/Output |
| `11-logs-correlacion.png` | Terminal: `docker compose logs api \| grep <request_id del escenario 2>` | Líneas JSON con el mismo `request_id`, `thread_id` y `trace_id` que la traza |

Consejo: usá la pestaña **Sessions** del proyecto para mostrar que cada `thread_id` agrupa sus
turnos, y el campo de búsqueda de la traza para encontrar un span por nombre.
