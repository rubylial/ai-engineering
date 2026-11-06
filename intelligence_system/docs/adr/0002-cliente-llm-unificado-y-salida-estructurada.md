# ADR-0002 · Cliente LLM unificado propio con tool calling y salida estructurada

- Estado: aceptada · Fecha: 2026-11-05

## Contexto
El entregable exige reutilizar el cliente async de la pre-entrega 1 detrás de una Factory. Los
agentes necesitan además tool calling (analyst) y salida estructurada (supervisor y grader).
El SDK `anthropic==2.5.0` ya no acepta `temperature` ni `top_p` (verificado en la firma de
`messages.create`).

## Decisión
- Extender `BaseLLMClient` con `generate(tools, tool_choice)`, `stream` y `generate_structured`.
- Salida estructurada forzando una tool cuyo `input_schema` es el JSON Schema del modelo
  Pydantic (`tool_choice={"type": "tool", ...}`), funciona igual en OpenAI con `tools`.
- Bucle de reparación: si la salida no valida, se reenvía la llamada fallida + el error exacto
  del validador (no se reintenta a ciegas). Tope `LLM_STRUCTURED_MAX_ATTEMPTS`.
- Reintentos con tenacity (backoff exponencial + jitter) solo para errores transitorios
  (429, red, 5xx); `max_retries=0` en los SDK para no duplicar.
- `LLM_TEMPERATURE` queda opcional y solo se envía a OpenAI.

## Alternativas descartadas
- `ChatAnthropic.with_structured_output`: no reutiliza la pre-entrega 1.
- Structured outputs nativos (`output_config.format`): soporte desigual entre modelos; el
  forzado por tool funciona con todos.

## Consecuencias
Los nodos no conocen ningún SDK. Las llamadas al LLM se trazan con el instrumentador de
OpenInference para Anthropic/OpenAI y quedan anidadas bajo el span del worker.
