# ADR-0007 · Supervisor jerárquico con handoffs `Command` y límites duros

- Estado: aceptada · Fecha: 2026-11-05

## Decisión
- Topología estrella: supervisor → researcher | analyst | writer; los workers vuelven con
  `Command(goto="supervisor")`; solo el writer termina el turno.
- Ruteo con `SupervisorDecision {next, reason, task}`; `task` es una instrucción autocontenida
  (el supervisor reescribe preguntas de seguimiento usando el historial).
- Reglas deterministas encima del LLM: presupuesto `AGENT_MAX_STEPS`, anti-bucle (mismo worker
  + misma tarea → writer) y ruteo por reglas si el LLM falla.
- Runtime común por nodo: `asyncio.timeout(AGENT_NODE_TIMEOUT_S)`, fallback por nodo,
  bitácora `steps`, verificación de la parcela del estado que escribe cada nodo.
- `AGENT_RECURSION_LIMIT` en el config de cada ejecución como red de seguridad.
- Estado mínimo por rol: el supervisor ve historial + progreso; el analyst, su tarea + evidencia
  documental; el writer, la pregunta + evidencia numerada.

## Consecuencias
Un fallo del researcher o analyst degrada la respuesta; un fallo del writer o del deadline
devuelve 502/504 (nunca un 200 con una respuesta inventada).
