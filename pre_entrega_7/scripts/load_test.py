"""Prueba de carga: 5 peticiones concurrentes contra la API.

Lanza 5 POST /tasks en paralelo, hace polling de cada GET /tasks/{id} hasta DONE o
FAILED, y reporta latencias (incluido p95 local como sanity check). Las métricas que
valen para la entrega (costo por ejecución y p95) salen del dashboard de Phoenix;
este script solo dispara la carga y confirma que las 5 completaron.

Uso (con la API y Redis levantados):
    uv run python scripts/load_test.py --base-url http://localhost:8000

Las queries son NO críticas a propósito: tienen que completar solas, sin pausa HITL.
Costo: ~5 corridas del grafo (cada una 3-7 llamadas a Haiku).
"""

from __future__ import annotations

import argparse
import asyncio
import time

import httpx

QUERIES = [
    "¿Qué dice el knowledge base sobre el patrón supervisor en LangGraph?",
    "Resumí qué es RAG híbrido denso + BM25 según la documentación.",
    "Calculá la media y el desvío estándar de 120, 180, 200, 140 y 160.",
    "¿Qué sabés de Redis 8 y sus módulos?",
    "Contá cuántas veces aparece 'async' en: 'async FastAPI usa async def y await async'.",
]


async def fire_one(client: httpx.AsyncClient, query: str) -> dict[str, object]:
    start = time.perf_counter()
    resp = await client.post("/tasks", json={"query": query})
    resp.raise_for_status()
    job_id = resp.json()["job_id"]

    status = "PENDING"
    while status not in {"DONE", "FAILED"}:
        await asyncio.sleep(0.5)
        r = await client.get(f"/tasks/{job_id}")
        r.raise_for_status()
        status = r.json()["status"]
        if status == "WAITING_APPROVAL":  # no debería pasar con estas queries
            await client.post(f"/tasks/{job_id}/approve", json={"approved": True})

    elapsed = time.perf_counter() - start
    return {"job_id": job_id, "status": status, "latency_s": round(elapsed, 2), "query": query[:48]}


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--n", type=int, default=5, help="peticiones concurrentes")
    args = parser.parse_args()

    async with httpx.AsyncClient(base_url=args.base_url, timeout=120) as client:
        queries = [QUERIES[i % len(QUERIES)] for i in range(args.n)]
        print(f"Lanzando {args.n} peticiones concurrentes contra {args.base_url} ...")
        t0 = time.perf_counter()
        results = await asyncio.gather(*(fire_one(client, q) for q in queries))
        wall = time.perf_counter() - t0

    latencies = sorted(r["latency_s"] for r in results)  # type: ignore[type-var]
    p95 = latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))]
    ok = sum(1 for r in results if r["status"] == "DONE")

    print("\n--- Resultados ---")
    for r in results:
        print(f"  {r['status']:<7} {r['latency_s']:>6}s  {r['job_id'][:8]}  {r['query']}")
    print(
        f"\n  completadas : {ok}/{len(results)}\n"
        f"  wall time   : {wall:.2f}s\n"
        f"  latencia    : min={latencies[0]}s  p95(local)={p95}s  max={latencies[-1]}s\n"
        f"\nLas métricas de la entrega (costo/ejecución y p95) salen de Phoenix: "
        f"abrí el proyecto 'api-agente' y mirá las 5 trazas de esta corrida."
    )
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
