"""Informe de evaluación del retriever: agregados y Markdown (funciones puras, sin red)."""

from __future__ import annotations

import json
from pathlib import Path

from intelligence_system.rag.evaluation import (
    EvalRow,
    aggregate,
    aggregate_by_type,
    render_markdown_report,
)

ROOT = Path(__file__).resolve().parents[2]
CORPUS_DOCS = {p.name for p in (ROOT / "data" / "corpus").glob("*.md")}

ROWS = [
    EvalRow("híbrido", "concepto", 1.0, 1.0, 1.0),
    EvalRow("híbrido", "parafrasis", 0.5, 1.0, 0.5),
    EvalRow("denso", "concepto", 1.0, 1.0, 1.0),
    EvalRow("denso", "parafrasis", 0.0, 0.0, 0.0),
]


def test_aggregate_averages_per_variant_in_order() -> None:
    out = aggregate(ROWS)
    assert list(out) == ["híbrido", "denso"]
    assert out["híbrido"] == {"P@k": 0.75, "R@k": 1.0, "MRR": 0.75, "n": 2.0}
    assert out["denso"]["P@k"] == 0.5 and out["denso"]["R@k"] == 0.5


def test_aggregate_by_type_keeps_counts() -> None:
    out = aggregate_by_type(ROWS)
    assert out["híbrido"]["parafrasis"] == (0.5, 1)
    assert out["denso"]["concepto"] == (1.0, 1)


def test_markdown_report_has_both_tables_and_values() -> None:
    md = render_markdown_report(ROWS, k=5, stamp="2026-11-06 00:00 UTC")
    assert "## Global" in md and "## P@5 por tipo de pregunta" in md
    assert "| híbrido | 0.750 | 1.000 | 0.750 |" in md
    assert "concepto (n=1)" in md and "parafrasis (n=1)" in md
    assert "| denso | 1.000 | 0.000 |" in md
    assert "2 preguntas, k=5" in md


def test_golden_set_points_to_existing_documents_and_is_typed() -> None:
    golden = json.loads((ROOT / "data" / "golden_set.json").read_text(encoding="utf-8"))
    assert len(golden) >= 15
    tipos = {item["tipo"] for item in golden}
    assert tipos == {"concepto", "termino_exacto", "parafrasis"}
    missing = {item["documento_id_esperado"] for item in golden} - CORPUS_DOCS
    assert not missing, f"documentos esperados que no existen en el corpus: {missing}"
    assert len({item["pregunta"] for item in golden}) == len(golden), "preguntas duplicadas"
