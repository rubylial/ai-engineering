"""
Script de evaluación del retriever híbrido.

Carga el golden_set.json, corre cada query contra el RAGSystem y calcula:
  - Precision@5 por query: qué fracción de los 5 chunks recuperados pertenece
    al documento esperado.
  - Recall@5 por query:    1 si al menos un chunk del documento esperado aparece
    en el top-5, 0 si no (binario — hay una sola fuente relevante por query).

Compara además el ensemble vs las ramas aisladas (semántico-only y BM25-only)
para mostrar si el híbrido realmente aporta.
"""

import json
import sys
from pathlib import Path
from statistics import mean

from rag import RAGSystem


GOLDEN_PATH = Path("./golden_set.json")
TOP_K = 5


def load_golden_set() -> list[dict]:
    if not GOLDEN_PATH.exists():
        raise FileNotFoundError(f"No se encontró {GOLDEN_PATH}")
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        data = json.load(f)
    if not data:
        raise ValueError("golden_set.json está vacío")
    return data


def evaluate_single(
    retrieved_docs: list,
    expected_source: str,
    k: int = TOP_K,
) -> tuple[float, int]:
    """
    Retorna (precision@k, hit) para una query:
      - precision@k: cuántos de los top-k son del source esperado / k
      - hit: 1 si al menos uno es del source esperado, 0 si no
             (equivalente a Recall@k cuando hay 1 doc relevante por query)
    """
    top_k = retrieved_docs[:k]
    hits_in_topk = sum(1 for d in top_k if d.metadata.get("source") == expected_source)
    precision = hits_in_topk / k
    recall = 1 if hits_in_topk > 0 else 0
    return precision, recall


def evaluate_retriever(
    name: str,
    retrieve_fn,
    golden: list[dict],
) -> dict:
    """Corre todas las queries por un retriever y agrega métricas."""
    print(f"\n{'─' * 72}")
    print(f"Evaluando: {name}")
    print(f"{'─' * 72}")

    precisions = []
    recalls = []
    for i, item in enumerate(golden, 1):
        query = item["pregunta"]
        expected = item["documento_id_esperado"]

        docs = retrieve_fn(query)
        p, r = evaluate_single(docs, expected, k=TOP_K)
        precisions.append(p)
        recalls.append(r)

        top_sources = [d.metadata.get("source", "?") for d in docs[:TOP_K]]
        status = "✓" if r == 1 else "✗"
        print(f"{status} Q{i}: P@{TOP_K}={p:.2f} R@{TOP_K}={r}")
        print(f"   query:    {query}")
        print(f"   esperado: {expected}")
        print(f"   top-{TOP_K}:   {top_sources}")

    return {
        "name": name,
        "precision_at_k_mean": mean(precisions),
        "recall_at_k_mean": mean(recalls),
        "n_queries": len(golden),
    }


def print_summary(results: list[dict]) -> None:
    """Imprime una tabla comparativa de los 3 retrievers."""
    print("\n" + "=" * 72)
    print(f"RESUMEN COMPARATIVO (top_k={TOP_K}, n_queries={results[0]['n_queries']})")
    print("=" * 72)
    print(f"{'Retriever':<28} {'Precision@5':>14} {'Recall@5':>12}")
    print("-" * 72)
    for r in results:
        print(
            f"{r['name']:<28} {r['precision_at_k_mean']:>14.3f} "
            f"{r['recall_at_k_mean']:>12.3f}"
        )
    print("=" * 72)

    # Diagnóstico corto
    best_recall = max(results, key=lambda r: r["recall_at_k_mean"])
    best_prec = max(results, key=lambda r: r["precision_at_k_mean"])
    print(f"→ Mejor Recall@5:    {best_recall['name']} ({best_recall['recall_at_k_mean']:.3f})")
    print(f"→ Mejor Precision@5: {best_prec['name']} ({best_prec['precision_at_k_mean']:.3f})")


def main() -> int:
    print("=" * 72)
    print("EVALUACIÓN DEL RETRIEVER HÍBRIDO")
    print("=" * 72)
    print(f"Golden set: {GOLDEN_PATH}")

    golden = load_golden_set()
    print(f"Queries:    {len(golden)}")
    print(f"Top-k:      {TOP_K}")

    rag = RAGSystem(top_k=TOP_K)

    # Corremos los 3 retrievers para comparar
    results = [
        evaluate_retriever("Ensemble (semántico + BM25)", rag.retrieve, golden),
        evaluate_retriever("Semántico puro (Pinecone)", rag.retrieve_semantic_only, golden),
        evaluate_retriever("BM25 puro (léxico)",         rag.retrieve_bm25_only, golden),
    ]

    print_summary(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
