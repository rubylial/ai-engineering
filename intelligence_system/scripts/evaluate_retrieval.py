"""Evalúa el retrieval con el golden set (lógica de evaluate.py de la pre-entrega 4).

Compara el híbrido (RAG_HYBRID_ALPHA) contra solo-denso (alpha=1) y solo-BM25 (alpha=0)
con Precision@k, Recall@k y MRR a nivel documento, global y por tipo de pregunta
(concepto / termino_exacto / parafrasis). Escribe el informe en docs/traces/evaluation.md.

Uso: uv run --env-file .env python scripts/evaluate_retrieval.py --k 5   (o: make eval)
Requiere el corpus ingestado y Redis arriba (estadísticas BM25).
COSTO: 1 embedding de consulta por pregunta y por variante (3 × tamaño del golden set);
sin llamadas al LLM.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from intelligence_system.config import get_settings
from intelligence_system.container import build_container
from intelligence_system.observability.logging import configure_logging
from intelligence_system.rag.evaluation import (
    EvalRow,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    render_markdown_report,
)

ROOT = Path(__file__).resolve().parents[1]


async def evaluate(golden: list[dict[str, str]], k: int, *, ingest_corpus: bool) -> list[EvalRow]:
    settings = get_settings()
    configure_logging("WARNING", settings.log_json)
    rows: list[EvalRow] = []
    async with build_container(settings) as container:
        if container.retriever is None:
            raise SystemExit(f"Retriever no disponible: {container.problems}")
        if ingest_corpus:
            # Modo offline (VECTOR_STORE=memory): el store nace vacío, hay que poblarlo.
            if container.ingest is None:
                raise SystemExit(f"Ingesta no disponible: {container.problems}")
            await container.ingest.ingest_corpus(settings.rag_corpus_dir)
        variants = {
            f"híbrido (alpha={settings.rag_hybrid_alpha})": container.retriever,
            "solo denso (alpha=1.0)": container.retriever.with_alpha(1.0),
            "solo BM25 (alpha=0.0)": container.retriever.with_alpha(0.0),
        }
        for name, retriever in variants.items():
            for item in golden:
                expected = [Path(item["documento_id_esperado"]).stem]
                results = await retriever.retrieve(item["pregunta"], top_k=k)
                docs = [r.doc_id for r in results]
                rows.append(
                    EvalRow(
                        variant=name,
                        tipo=item.get("tipo", "sin_tipo"),
                        precision=precision_at_k(docs, expected, k),
                        recall=recall_at_k(docs, expected, k),
                        reciprocal_rank=reciprocal_rank(docs, expected),
                    )
                )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--golden", type=Path, default=ROOT / "data" / "golden_set.json")
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "traces" / "evaluation.md")
    parser.add_argument(
        "--ingest-corpus",
        action="store_true",
        help="Ingesta el corpus antes de evaluar (para VECTOR_STORE=memory; con Pinecone ya está ingestado)",
    )
    args = parser.parse_args()
    golden = json.loads(args.golden.read_text(encoding="utf-8"))
    rows = asyncio.run(evaluate(golden, args.k, ingest_corpus=args.ingest_corpus))
    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    report = render_markdown_report(rows, k=args.k, stamp=stamp)
    print(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(f"Informe guardado en {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
