# Evaluación del retriever (golden set)

Corrido con `make eval` el 2026-11-06 07:21 UTC. 16 preguntas, k=5. Métricas a nivel documento: Precision@k, Recall@k y MRR.

## Global

| variante | P@5 | R@5 | MRR |
| --- | --- | --- | --- |
| híbrido (alpha=0.7) | 0.613 | 1.000 | 0.969 |
| solo denso (alpha=1.0) | 0.588 | 1.000 | 0.950 |
| solo BM25 (alpha=0.0) | 0.550 | 1.000 | 0.969 |

## P@5 por tipo de pregunta

| variante | termino_exacto (n=6) | concepto (n=6) | parafrasis (n=4) |
| --- | --- | --- | --- |
| híbrido (alpha=0.7) | 0.600 | 0.700 | 0.500 |
| solo denso (alpha=1.0) | 0.600 | 0.633 | 0.500 |
| solo BM25 (alpha=0.0) | 0.467 | 0.667 | 0.500 |

Tipos: `concepto` (pregunta semántica), `termino_exacto` (contiene el nombre técnico tal cual aparece en el documento), `parafrasis` (misma idea sin compartir vocabulario con el documento).
