"""
Demo rápido de consulta contra el retriever híbrido.
Permite probar queries sin correr la batería entera de evaluación.

Uso:
    python query.py "¿Qué es un semáforo en asyncio?"
"""

import sys
from rag import RAGSystem


def main() -> int:
    if len(sys.argv) < 2:
        print("Uso: python query.py '<tu pregunta>'")
        return 1

    query = " ".join(sys.argv[1:])
    rag = RAGSystem()
    print(f"\n❯ Query: {query}\n")

    docs = rag.retrieve(query)
    for i, d in enumerate(docs, 1):
        source = d.metadata.get("source", "?")
        category = d.metadata.get("category", "?")
        snippet = d.page_content[:160].replace("\n", " ")
        print(f"[{i}] {source}  (category={category})")
        print(f"    {snippet}...\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
