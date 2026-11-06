"""Ingesta idempotente del corpus a Pinecone (sin pasar por la API).

Uso:
    uv run python scripts/ingest.py                    # corpus de RAG_CORPUS_DIR
    uv run python scripts/ingest.py --dir otra/carpeta
    docker compose exec api python scripts/ingest.py   # dentro del contenedor
Correrlo dos veces seguidas sube 0 vectores la segunda vez (dedup por hash).
COSTO: solo embebe chunks nuevos o modificados (Pinecone Inference).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from intelligence_system.config import get_settings
from intelligence_system.container import build_container
from intelligence_system.observability.logging import configure_logging


async def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dir", type=Path, default=None, help="carpeta con .md (default: RAG_CORPUS_DIR)")
    args = parser.parse_args()
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    async with build_container(settings) as container:
        if container.ingest is None:
            print(f"La ingesta no está disponible: {container.problems}", file=sys.stderr)
            return 1
        report = await container.ingest.ingest_corpus(args.dir or settings.rag_corpus_dir)
    print(json.dumps(report.model_dump(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
