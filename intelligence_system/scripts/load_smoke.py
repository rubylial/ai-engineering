"""Smoke de concurrencia: N peticiones simultáneas a POST /v1/chat con threads distintos.

Respalda la afirmación "asyncio de punta a punta": si algo bloqueara el event loop, el tiempo
total (wall) tendería a la SUMA de las latencias; si no, al MÁXIMO. Imprime p50/p95/max/wall,
el paralelismo efectivo (suma/wall) y escribe el informe en docs/traces/load.md.

Uso (con el stack arriba): uv run python scripts/load_smoke.py --n 5     (o: make load)
Variables: API_BASE_URL (http://localhost:8000), API_KEY, PHOENIX_UI_URL (http://localhost:6006)
COSTO: N turnos completos (cada uno 3–5 llamadas a claude-haiku-5-5 + 1 embedding) → centavos.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import os
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = [
    "¿Para qué sirve asyncio.Semaphore cuando llamo a un LLM?",
    "¿Qué es BM25 y qué limitación tiene?",
    "¿Qué diferencia hay entre get y query en una vector database?",
    "¿Por qué la similitud coseno es la métrica estándar en RAG?",
    "¿Para qué sirven los namespaces en Pinecone?",
    "¿Qué es Reciprocal Rank Fusion?",
    "¿Qué es un golden set?",
    "¿Cómo se usa asyncio.timeout con un LLM?",
]


@dataclass
class Sample:
    thread_id: str
    question: str
    status_code: int
    latency_ms: float
    grounded: bool | None = None
    citations: int = 0
    trace_id: str | None = None
    error: str | None = None


def percentile(values: list[float], pct: float) -> float:
    """Percentil por rango más cercano (nearest-rank), sin interpolar."""
    if not values:
        raise ValueError("sin muestras")
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[rank - 1]


async def one_request(client: httpx.AsyncClient, question: str) -> Sample:
    thread_id = f"load-{uuid.uuid4().hex[:8]}"
    started = time.perf_counter()
    try:
        resp = await client.post("/v1/chat", json={"message": question, "thread_id": thread_id})
        latency = (time.perf_counter() - started) * 1000
        body: dict[str, Any] = resp.json() if resp.content else {}
        return Sample(
            thread_id=thread_id,
            question=question,
            status_code=resp.status_code,
            latency_ms=round(latency, 1),
            grounded=body.get("grounded"),
            citations=len(body.get("citations") or []),
            trace_id=body.get("trace_id"),
            error=None if resp.status_code == 200 else body.get("detail") or resp.text[:120],
        )
    except httpx.HTTPError as exc:
        latency = (time.perf_counter() - started) * 1000
        return Sample(thread_id, question, 0, round(latency, 1), error=f"{type(exc).__name__}: {exc}")


async def run(base_url: str, api_key: str, n: int) -> tuple[list[Sample], float]:
    async with httpx.AsyncClient(base_url=base_url, headers={"X-API-Key": api_key}, timeout=240) as client:
        ready = await client.get("/ready")
        if ready.status_code != 200:
            print(f"⚠ /ready respondió {ready.status_code}: {ready.text}", file=sys.stderr)
        questions = [QUESTIONS[i % len(QUESTIONS)] for i in range(n)]
        started = time.perf_counter()
        samples = await asyncio.gather(*(one_request(client, q) for q in questions))
        wall_ms = (time.perf_counter() - started) * 1000
    return list(samples), round(wall_ms, 1)


def render_markdown(samples: list[Sample], wall_ms: float, *, stamp: str, phoenix_url: str) -> str:
    latencies = [s.latency_ms for s in samples]
    ok = [s for s in samples if s.status_code == 200]
    total = sum(latencies)
    lines = [
        "# Smoke de concurrencia (`make load`)",
        "",
        f"Corrido el {stamp}: **{len(samples)} peticiones simultáneas** a `POST /v1/chat`, cada una en un "
        "`thread_id` distinto (el lock por thread no serializa entre ellas).",
        "",
        "| Métrica | Valor |",
        "| --- | --- |",
        f"| Completadas (200) | {len(ok)}/{len(samples)} |",
        f"| Latencia p50 | {percentile(latencies, 50) / 1000:.2f} s |",
        f"| Latencia p95 | {percentile(latencies, 95) / 1000:.2f} s |",
        f"| Latencia máx. | {max(latencies) / 1000:.2f} s |",
        f"| Suma de latencias | {total / 1000:.2f} s |",
        f"| Wall time (todas en paralelo) | {wall_ms / 1000:.2f} s |",
        f"| Paralelismo efectivo (suma / wall) | {total / wall_ms:.2f}× |",
        "",
        "Lectura: si el servidor serializara las peticiones, el wall time se acercaría a la suma "
        "(paralelismo ≈ 1×); con I/O async se acerca a la latencia máxima individual.",
        "",
        "| # | thread_id | Pregunta | Status | grounded | citas | Latencia | Traza |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for i, s in enumerate(samples, start=1):
        trace = f"[`{s.trace_id[:12]}…`]({phoenix_url}/redirects/traces/{s.trace_id})" if s.trace_id else "—"
        status = str(s.status_code) if s.status_code else f"error: {s.error}"
        lines.append(
            f"| {i} | `{s.thread_id}` | {s.question} | {status} | {s.grounded} | {s.citations} | "
            f"{s.latency_ms / 1000:.2f} s | {trace} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--base-url", default=os.getenv("API_BASE_URL", "http://localhost:8000"))
    parser.add_argument("--api-key", default=os.getenv("API_KEY"))
    parser.add_argument("--phoenix-url", default=os.getenv("PHOENIX_UI_URL", "http://localhost:6006"))
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "traces" / "load.md")
    args = parser.parse_args()
    if not args.api_key:
        print("Falta API_KEY (variable de entorno o --api-key).", file=sys.stderr)
        return 2

    print(f"Lanzando {args.n} peticiones concurrentes contra {args.base_url} ...")
    samples, wall_ms = asyncio.run(run(args.base_url, args.api_key, args.n))
    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    report = render_markdown(samples, wall_ms, stamp=stamp, phoenix_url=args.phoenix_url.rstrip("/"))
    print(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(f"Informe guardado en {args.out}", file=sys.stderr)
    return 0 if all(s.status_code == 200 for s in samples) else 1


if __name__ == "__main__":
    sys.exit(main())
