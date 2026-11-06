"""Mini script asíncrono: corre el pipeline sobre tres textos de ejemplo.

Uso:
    cp .env.example .env            # completá ANTHROPIC_API_KEY
    uv run python scripts/demo.py

Costo: ~3 llamadas chicas a `claude-haiku-5-5` (unos pocos centavos).
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

from langchain_anthropic import ChatAnthropic

from entity_pipeline import build_chain, process_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

SAMPLES = [
    "API en FastAPI con caché en Redis y persistencia en PostgreSQL. El endpoint "
    "/checkout está cayendo bajo carga: p95 de 4s y 503 recurrentes. La cola de "
    "conexiones del pool a Postgres se satura y los workers de Uvicorn se "
    "quedan esperando.",
    "Pipeline nocturno en AWS Glue que lee Parquet de S3, aplica joins en Spark "
    "y escribe a Redshift. Falla intermitentemente por 'OutOfMemoryError' en "
    "executors de 4 GB; el DAG queda en estado FAILED y se reintenta a mano.",
    "Se agregó un índice BTREE sobre `orders.created_at` en PostgreSQL 16. La "
    "query del dashboard bajó de 1.2s a 40 ms. Sin impacto en escrituras.",
]


async def main() -> int:
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        print(
            "Falta ANTHROPIC_API_KEY. Copiá .env.example a .env y completalo, "
            "o exportá la variable antes de correr el script.",
            file=sys.stderr,
        )
        return 1

    model = ChatAnthropic(
        model=os.getenv("LLM_MODEL", "claude-haiku-5-5"),
        max_retries=0,  # la cadena ya tiene .with_retry(); evitamos retries dobles
    )
    chain = build_chain(model, max_attempts=3)

    for i, text in enumerate(SAMPLES, start=1):
        print(f"\n=== Texto {i} ===\n{text}\n--- Salida ---")
        result = await process_text(chain, text)
        print(result.model_dump_json(indent=2, by_alias=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
