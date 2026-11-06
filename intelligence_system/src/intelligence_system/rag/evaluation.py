"""Métricas de retrieval (lógica de `evaluate.py` de la pre-entrega 4 + RRF).

Funciones puras: las usan las tools del analyst y `scripts/evaluate_retrieval.py`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import mean


@dataclass(frozen=True)
class EvalRow:
    """Métricas de UNA pregunta del golden set bajo UNA variante del retriever."""

    variant: str
    tipo: str
    precision: float
    recall: float
    reciprocal_rank: float


def aggregate(rows: Sequence[EvalRow]) -> dict[str, dict[str, float]]:
    """Promedios por variante (en orden de aparición): {variante: {P@k, R@k, MRR, n}}."""
    out: dict[str, dict[str, float]] = {}
    for variant in dict.fromkeys(r.variant for r in rows):
        sub = [r for r in rows if r.variant == variant]
        out[variant] = {
            "P@k": mean(r.precision for r in sub),
            "R@k": mean(r.recall for r in sub),
            "MRR": mean(r.reciprocal_rank for r in sub),
            "n": float(len(sub)),
        }
    return out


def aggregate_by_type(rows: Sequence[EvalRow]) -> dict[str, dict[str, tuple[float, int]]]:
    """P@k promedio por tipo de pregunta: {variante: {tipo: (P@k, n)}}."""
    out: dict[str, dict[str, tuple[float, int]]] = {}
    for variant in dict.fromkeys(r.variant for r in rows):
        per_type: dict[str, tuple[float, int]] = {}
        for tipo in dict.fromkeys(r.tipo for r in rows if r.variant == variant):
            sub = [r for r in rows if r.variant == variant and r.tipo == tipo]
            per_type[tipo] = (mean(r.precision for r in sub), len(sub))
        out[variant] = per_type
    return out


def render_markdown_report(rows: Sequence[EvalRow], *, k: int, stamp: str) -> str:
    """Informe en Markdown: tabla global + tabla de P@k por tipo de pregunta."""
    overall = aggregate(rows)
    by_type = aggregate_by_type(rows)
    tipos = list(dict.fromkeys(r.tipo for r in rows))
    n_questions = int(next(iter(overall.values()))["n"]) if overall else 0
    lines = [
        "# Evaluación del retriever (golden set)",
        "",
        f"Corrido con `make eval` el {stamp}. {n_questions} preguntas, k={k}. "
        "Métricas a nivel documento: Precision@k, Recall@k y MRR.",
        "",
        "## Global",
        "",
        f"| variante | P@{k} | R@{k} | MRR |",
        "| --- | --- | --- | --- |",
    ]
    for variant, m in overall.items():
        lines.append(f"| {variant} | {m['P@k']:.3f} | {m['R@k']:.3f} | {m['MRR']:.3f} |")
    lines += [
        "",
        f"## P@{k} por tipo de pregunta",
        "",
        "| variante | " + " | ".join(f"{t} (n={by_type[next(iter(by_type))][t][1]})" for t in tipos) + " |",
        "| --- | " + " | ".join("---" for _ in tipos) + " |",
    ]
    for variant, per_type in by_type.items():
        cells = " | ".join(f"{per_type[t][0]:.3f}" if t in per_type else "—" for t in tipos)
        lines.append(f"| {variant} | {cells} |")
    lines += [
        "",
        "Tipos: `concepto` (pregunta semántica), `termino_exacto` (contiene el nombre técnico tal cual "
        "aparece en el documento), `parafrasis` (misma idea sin compartir vocabulario con el documento).",
        "",
    ]
    return "\n".join(lines)


def precision_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k debe ser positivo")
    relevant_set = set(relevant)
    return sum(1 for item in retrieved[:k] if item in relevant_set) / k


def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    relevant_set = set(relevant)
    if not relevant_set:
        raise ValueError("relevant no puede estar vacío")
    hits = {item for item in retrieved[:k] if item in relevant_set}
    return len(hits) / len(relevant_set)


def hit_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> int:
    relevant_set = set(relevant)
    return int(any(item in relevant_set for item in retrieved[:k]))


def reciprocal_rank(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    relevant_set = set(relevant)
    for position, item in enumerate(retrieved, start=1):
        if item in relevant_set:
            return 1.0 / position
    return 0.0


def rrf_fuse(rankings: Sequence[Sequence[str]], *, k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: score(d) = Σ 1/(k + rank_i(d))."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
