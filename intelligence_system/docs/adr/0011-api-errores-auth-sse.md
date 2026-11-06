# ADR-0011 · Contrato HTTP: X-API-Key, problem+json y SSE

- Estado: aceptada · Fecha: 2026-11-05

## Decisión
- Autenticación por header `X-API-Key` (comparación en tiempo constante). `/health` y `/ready`
  sin auth para orquestadores.
- Errores RFC 7807 simplificado: `{type, title, status, detail, request_id, trace_id}` también
  para validación (422), auth (401), dependencias no listas (503), proveedor externo (502) y
  deadline (504). Un middleware ASGI captura lo no manejado y responde 500 en el mismo formato.
- `X-Request-ID` del cliente se respeta si es seguro; si no, se genera.
- SSE con `sse-starlette`: eventos `step`, `token`, `citation`, `done`, `error`. El grafo corre
  en una tarea productora con su contexto de traza; si el cliente corta, se cancela.
- La app arranca aunque falten claves: `/ready` explica qué falta y los endpoints afectados
  devuelven 503 con el detalle.
