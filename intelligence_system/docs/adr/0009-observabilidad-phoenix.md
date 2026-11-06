# ADR-0009 · Caja de cristal: Phoenix + OpenInference + logs JSON

- Estado: aceptada · Fecha: 2026-11-05

## Decisión
- `phoenix.otel.register` (OTLP/HTTP a `PHOENIX_COLLECTOR_ENDPOINT/v1/traces`, exportación en
  lote desde un hilo aparte) + instrumentadores OpenInference de LangChain, Anthropic y OpenAI.
- Span raíz `chat.turn`/`chat.stream` con input, output, `request_id`, `thread_id` y
  `session.id = thread_id` (vista Sessions de Phoenix por conversación).
- `using_attributes` propaga session y metadata a los spans de los instrumentadores.
- Span `agent.<worker>` (kind AGENT, atributo `worker`) colgado del span del nodo de LangGraph
  vía `openinference.instrumentation.langchain.get_current_span()`; las llamadas al LLM quedan
  debajo de su worker.
- Telemetría HTTP nativa de FastAPI 0.162 apagada por defecto (`PHOENIX_HTTP_SPANS=false`):
  si no, la raíz sería `POST /v1/chat` sin input/output.
- Spans nativos del SDK de Anthropic 2.x apagados (`AsyncAnthropic(open_telemetry=False)`):
  OpenInference ya los reemplaza en `messages.*`, y los únicos que sobrevivían eran trazas
  sueltas `anthropic.models.retrieve` del probe de `/ready`, sin `request_id` ni `thread_id`.
- Los errores llevan su `trace_id` en el problem+json.
- structlog JSON con `request_id`, `thread_id`, `trace_id` y `span_id` en cada línea.

## Verificación
Con Phoenix 21.0.0 local: 6 escenarios, trazas recibidas con los atributos esperados, el
timeout como `ERROR`, y el link `/redirects/traces/<trace_id>` abre la traza correcta.
