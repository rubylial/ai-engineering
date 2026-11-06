# ADR-0008 · "Sin cita no hay afirmación"

- Estado: aceptada · Fecha: 2026-11-05

## Decisión
1. Grader del researcher (salida estructurada) que descarta fragmentos irrelevantes; los
   índices inventados se filtran en código.
2. Sin evidencia, el writer responde la abstención sin llamar al LLM.
3. El writer cita con `[n]`; se validan en código contra la evidencia; los inventados se borran.
4. Sin ninguna cita válida (o si el modelo responde `SIN_EVIDENCIA`), la respuesta se reemplaza
   por la abstención y `grounded=false`.
5. En streaming, el arranque se retiene hasta descartar el centinela; el evento `done` trae la
   respuesta final validada.

## Consecuencias
Puede abstenerse de más si el modelo olvida citar; es preferible a una respuesta sin respaldo.
