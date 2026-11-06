"""Genera el diagrama Mermaid del grafo REAL (graph.get_graph().draw_mermaid()).

No dibuja a mano: compila el mismo StateGraph que usa la API (con dependencias
en memoria, sin red) y escribe:
  - docs/graph.mmd
  - el bloque entre <!-- GRAPH:START --> y <!-- GRAPH:END --> del README.md
Uso: uv run python scripts/export_graph.py   (o: make graph)
"""

from __future__ import annotations

import sys
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver

from intelligence_system.config import Settings
from intelligence_system.container import AppContainer, assemble
from intelligence_system.llm.providers.fake_client import FakeLLMClient
from intelligence_system.rag.embeddings import HashEmbeddings
from intelligence_system.rag.retriever import NoopReranker
from intelligence_system.rag.sparse import InMemoryBM25StatsStore
from intelligence_system.rag.vector_store import InMemoryVectorStore

ROOT = Path(__file__).resolve().parents[1]
START, END = "<!-- GRAPH:START -->", "<!-- GRAPH:END -->"


def render() -> str:
    settings = Settings(
        _env_file=None,
        llm_provider="fake",
        embeddings_provider="fake",
        vector_store="memory",
        checkpointer_backend="memory",
        phoenix_enabled=False,
        embeddings_dimension=64,
    )
    container = assemble(
        AppContainer(settings=settings),
        llm=FakeLLMClient(),
        embeddings=HashEmbeddings(dimension=64),
        store=InMemoryVectorStore(dimension=64),
        stats=InMemoryBM25StatsStore(),
        reranker=NoopReranker(),
        checkpointer=InMemorySaver(),
    )
    assert container.graph is not None
    return container.graph.get_graph().draw_mermaid()


def main() -> int:
    mermaid = render()
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "graph.mmd").write_text(mermaid, encoding="utf-8")
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    start, end = text.find(START), text.find(END)
    if start == -1 or end == -1:
        print("README.md no tiene los marcadores GRAPH:START/END", file=sys.stderr)
        return 1
    block = f"{START}\n```mermaid\n{mermaid.strip()}\n```\n{END}"
    readme.write_text(text[:start] + block + text[end + len(END) :], encoding="utf-8")
    print(mermaid)
    print("✓ docs/graph.mmd y README.md actualizados")
    return 0


if __name__ == "__main__":
    sys.exit(main())
