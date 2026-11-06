---
doc_id: 07_rag_metricas
title: "Evaluación de sistemas RAG: Precision@k y Recall@k"
category: evaluation
tags: [precision, recall, golden-set]
source: 07_rag_metricas.md
---

# Evaluación de sistemas RAG: Precision@k y Recall@k

## Por qué evaluar RAG

Un sistema RAG puede alucinar, puede recuperar documentos irrelevantes, puede
perder documentos relevantes. Sin métricas, no sabés si un cambio en el
chunking, el embedder o el prompt mejoró o empeoró el sistema. La evaluación
es la diferencia entre "parece que anda bien" y "mejoró 15% el recall sobre
el baseline".

## El Golden Set

Un golden set es un conjunto curado de preguntas con su respuesta esperada
o con los documentos que deberían recuperarse para responderlas. Se construye
a mano, típicamente a partir de preguntas reales de usuarios o preguntas
diseñadas para cubrir casos borde.

Para evaluar retrieval (no la generación), cada entrada del golden set es:

```json
{
  "pregunta": "¿Qué es asyncio.Semaphore?",
  "documento_id_esperado": "01_asyncio_python.md"
}
```

## Precision@k

De los top-k documentos recuperados, qué fracción es relevante.

Precision@5 = relevantes_en_top_5 / 5

Si recuperás 5 documentos y 3 son relevantes, Precision@5 = 0.6. Es una
métrica orientada a "precisión": no quiero inundar al LLM con ruido.

## Recall@k

De todos los documentos relevantes que existen para la query, qué fracción
aparece en los top-k.

Recall@5 = relevantes_recuperados / total_relevantes

Si existen 2 documentos relevantes para una query y uno aparece en el top-5,
Recall@5 = 0.5. Es una métrica orientada a "cobertura": no quiero perderme
documentos importantes.

## El tradeoff

Aumentar k mejora recall pero empeora precision. Un sistema con k=100 va a
encontrar casi todo lo relevante (alto recall) pero el 95% de lo recuperado
será basura (bajo precision). Un sistema con k=1 va a tener alta precision
pero puede perderse documentos importantes.

El valor de k se elige según el caso de uso:

- Para RAG con LLM: k=3-5 porque más contexto diluye la atención.
- Para búsqueda facetada (Google-like): k=10-20 porque el usuario ve la lista.
- Para re-ranking: k=20-50 para dar material al re-ranker.

## Métricas adicionales

Más allá de P@k y R@k, hay otras métricas útiles:

- **MRR (Mean Reciprocal Rank)**: 1/rank_del_primer_relevante, promediado.
- **NDCG (Normalized DCG)**: pondera por posición y relevancia graduada.
- **Hit@k**: binaria, 1 si al menos un relevante está en top-k.

Para evaluación de generación (no de retrieval) se usan otras: faithfulness,
answer relevancy, context precision — típicamente con RAGAS u otra librería.
