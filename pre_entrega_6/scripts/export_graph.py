"""Genera `docs/graph.mmd` (y, si hay red, `docs/graph.png`) desde el grafo compilado.

Uso:
    uv run python scripts/export_graph.py

No consume tokens ni requiere claves: usa `build_graph(...)` directamente con nodos
stub, salteando `create_react_agent` (que valida que el LLM sea un `BaseChatModel`
real). El PNG usa el servicio público mermaid.ink, así que requiere red saliente.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableLambda

from orquestador.graph import build_graph
from orquestador.state import SupervisorDecision, SupervisorState
from orquestador.supervisor import supervisor_node_factory


async def _never(_: object) -> SupervisorDecision:
    raise RuntimeError("stub invocado al exportar el grafo (no debería pasar)")


async def _stub_node(_: SupervisorState) -> dict[str, Any]:
    return {}


def main() -> int:
    supervisor = supervisor_node_factory(RunnableLambda(_never), max_steps=1)
    graph = build_graph(supervisor, _stub_node, _stub_node)

    out_dir = Path(__file__).resolve().parents[1] / "docs"
    out_dir.mkdir(exist_ok=True)

    mermaid = graph.get_graph().draw_mermaid()
    (out_dir / "graph.mmd").write_text(mermaid, encoding="utf-8")
    print(f"ok {out_dir / 'graph.mmd'}")

    try:
        png = graph.get_graph().draw_mermaid_png()
        (out_dir / "graph.png").write_bytes(png)
        print(f"ok {out_dir / 'graph.png'}")
    except Exception as exc:
        print(f"skip PNG ({exc.__class__.__name__}: {exc}); el .mmd igual quedó.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
