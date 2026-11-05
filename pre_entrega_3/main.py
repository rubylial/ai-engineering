"""
Pre-entrega 3 — Sistema RAG end-to-end.

Flujo que ejecuta este script:
  1. Ingesta: indexa los .md de data/ en Chroma (o salta si ya está hecho).
  2. Prueba 1: pregunta cuya respuesta SÍ está en el corpus.
  3. Prueba 2 (trampa): pregunta cuya respuesta NO está en el corpus.
     Verifica que el modelo responda "no tengo esa información" en vez de alucinar.
"""

import asyncio
import sys

from rag import get_rag_response, ingest_data_directory
from rag.schemas import RAGResponse


def _print_response(query: str, response: RAGResponse) -> None:
    """Formatea una respuesta de forma legible para la consola."""
    print(f"\n❯ Pregunta: {query}")
    print(f"  has_answer: {response.has_answer}")
    print(f"  answer:     {response.answer}")
    if response.references:
        print(f"  references: ({len(response.references)})")
        for i, ref in enumerate(response.references, 1):
            excerpt = ref.excerpt[:140] + "..." if len(ref.excerpt) > 140 else ref.excerpt
            print(f"    {i}. [{ref.source}] {excerpt}")
    else:
        print("  references: (sin referencias — respuesta no encontrada en el contexto)")


async def main() -> int:
    # --- Paso 1: asegurar que la vectorstore esté lista ---
    print("=" * 72)
    print("INGESTA")
    print("=" * 72)
    ingest_data_directory()

    # --- Paso 2: prueba grounded — respuesta SÍ está en el corpus ---
    print("\n" + "=" * 72)
    print("PRUEBA 1 — Pregunta cuya respuesta ESTÁ en el corpus")
    print("=" * 72)
    q1 = "¿Para qué sirve asyncio.Semaphore y cuándo lo usarías?"
    r1 = await get_rag_response(q1)
    _print_response(q1, r1)

    assert r1.has_answer, "La prueba 1 debería tener respuesta en el contexto."
    assert len(r1.references) > 0, "La prueba 1 debería incluir referencias."

    # --- Paso 3: prueba trampa — respuesta NO está en el corpus ---
    print("\n" + "=" * 72)
    print("PRUEBA 2 (trampa) — Pregunta cuya respuesta NO está en el corpus")
    print("=" * 72)
    q2 = "¿Cuál es el precio actual de Bitcoin en pesos colombianos?"
    r2 = await get_rag_response(q2)
    _print_response(q2, r2)

    # Validación anti-alucinación: el modelo no debe inventar un precio.
    assert not r2.has_answer, (
        "ALUCINACIÓN detectada: el modelo respondió algo que no está en el contexto."
    )
    assert len(r2.references) == 0, (
        "El modelo citó referencias para algo que no estaba en el contexto."
    )

    print("\n" + "=" * 72)
    print("✓ Las dos pruebas pasaron. RAG grounded funcionando correctamente.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
