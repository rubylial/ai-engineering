# ADR-0012 · Tools del analyst: cálculos de AI Engineering

- Estado: aceptada · Fecha: 2026-11-05

## Contexto
La pre-entrega 5 usaba tools de clientes y pedidos (otro dominio). El corpus del RAG es
documentación de AI Engineering.

## Decisión
El analyst usa cinco tools deterministas del mismo dominio que el corpus, todas con
`args_schema` y salida Pydantic: `estimar_chunks`, `calcular_metricas_retrieval` (lógica de
`evaluate.py` de la pre-entrega 4), `fusionar_rankings_rrf` (RRF de la pre-entrega 4),
`estimar_memoria_vectores` y `similitud_coseno` (usa el proveedor de embeddings).
Se mantienen los patrones de la pre-entrega 5: docstrings "cuándo usar / cuándo no", errores
devueltos como datos para que el LLM corrija y tope de llamadas.

## Consecuencias
Las preguntas multi-paso son naturales ("según la documentación... calculá..."): el researcher
trae el parámetro y el analyst calcula con él.
