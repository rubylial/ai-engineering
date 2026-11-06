"""Demo del orquestador: 3 queries que ejercitan rutas distintas del supervisor.

Query 1 → researcher + analyst (dos agentes)
Query 2 → solo researcher (una búsqueda alcanza)
Query 3 → solo analyst (cálculo puro)

Uso:
    cp .env.example .env            # completá ANTHROPIC_API_KEY
    uv run python scripts/demo.py

Costo: ~15-25 llamadas a claude-haiku-5-5 (prompts cortos). Centavos de dólar.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

from langchain_core.messages import HumanMessage

from orquestador import build_graph_from_llm
from orquestador.llm import build_chat_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

QUERIES: list[tuple[str, str]] = [
    (
        "doble dominio",
        "Buscá qué dice el knowledge base sobre el patrón supervisor en LangGraph "
        "y después contá cuántas veces aparece la palabra 'supervisor' en lo que "
        "haya devuelto la búsqueda.",
    ),
    (
        "solo búsqueda",
        "¿Qué sabés de Redis 8 y sus módulos?",
    ),
    (
        "solo cálculo",
        "Calculá la media y el desvío estándar de los números 120, 180, 200, 140 y 160.",
    ),
]


def _print_step(idx: int, update: dict[str, object]) -> None:
    for node, payload in update.items():
        if not isinstance(payload, dict):
            continue
        msgs = payload.get("messages") or []
        label = msgs[-1].content if msgs else "(sin mensaje)"
        print(f"  [{idx}] {node:<12} → {str(label)[:220]}")


async def main() -> int:
    if not os.getenv("ANTHROPIC_API_KEY"):
        print(
            "Falta ANTHROPIC_API_KEY. Copiá .env.example a .env y completalo, "
            "o exportá la variable antes de correr el script.",
            file=sys.stderr,
        )
        return 1

    graph = build_graph_from_llm(build_chat_model())

    for name, query in QUERIES:
        print(f"\n╔══ Query: {name} ══╗\n{query}\n--- Traza ---")
        idx = 0
        final_state: dict[str, object] = {}
        async for update in graph.astream(
            {"messages": [HumanMessage(content=query)]},
            config={"recursion_limit": 25},
        ):
            idx += 1
            _print_step(idx, update)
            # Guardamos el último update para resumir
            for payload in update.values():
                if isinstance(payload, dict):
                    final_state.update(payload)

        contributions = final_state.get("contributions", [])
        contribs_list = contributions if isinstance(contributions, list) else []
        print(
            f"--- Resumen ---\n"
            f"  pasos          : {final_state.get('step', '?')}\n"
            f"  task_completed : {final_state.get('task_completed')}\n"
            f"  contribuciones : {[c['agent'] for c in contribs_list]}\n"
            f"  respuesta      : {final_state.get('task', '(sin task)')[:220]}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
