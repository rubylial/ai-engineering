# ADR-0001 · El sistema vive en `intelligence-system/` dentro del repo del curso

- Estado: aceptada · Fecha: 2026-11-05

## Contexto
El repo del curso contiene las pre-entregas como carpetas independientes. La regla "no destruir"
prohíbe borrar su código; la entrega final debe reutilizarlo.

## Decisión
Crear la carpeta `intelligence-system/` al lado de las pre-entregas, con su propio
`pyproject.toml`, `uv.lock`, Docker y docs. El código reutilizado se **refactoriza** dentro del
paquete (`llm/`, `rag/`, `agents/`) y el corpus y el golden set se **copian** a `data/`, de modo
que cada pre-entrega sigue ejecutándose como el día que se entregó.

## Alternativas descartadas
- Repo nuevo: pierde la trazabilidad con las pre-entregas que pide la integración.
- Mover archivos de las pre-entregas: rompe la regla de no destruir.

## Consecuencias
Hay duplicación controlada (corpus). `docs/ARCHITECTURE.md` mapea qué vino de cada pre-entrega.
