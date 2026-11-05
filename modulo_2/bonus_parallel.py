"""
Bonus — Pre-entrega 2 extendida con RunnableParallel.

Caso real del material: pipeline de análisis de reseñas.
En paralelo clasifica el sentimiento y extrae el tema,
después sintetiza una línea de ticket para soporte.

Este archivo NO es obligatorio para la pre-entrega; es la pieza
que demuestra por qué LCEL es más que sintaxis elegante: te da
paralelismo declarativo sin escribir asyncio.gather manualmente.
"""

import asyncio
import os
import sys
import time

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableParallel


load_dotenv()
PROVIDER = os.getenv("LLM_PROVIDER", "anthropic").lower().strip()


def build_model() -> Runnable:
    if PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model="claude-haiku-4-5", temperature=0, max_tokens=256)
    if PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=256)
    raise ValueError(f"LLM_PROVIDER inválido: {PROVIDER}")


model = build_model()
parser = StrOutputParser()


# ============================================================
# Sub-cadena 1: sentimiento
# ============================================================
sentiment_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "Clasificás sentimientos en una sola palabra."),
        ("human", "Reseña: {review}\n\nRespondé solo: positivo, negativo o neutro."),
    ])
    | model
    | parser
)


# ============================================================
# Sub-cadena 2: tema principal
# ============================================================
topic_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "Extraés el tema principal de reseñas de clientes."),
        ("human", "Reseña: {review}\n\nRespondé en máximo 3 palabras."),
    ])
    | model
    | parser
)


# ============================================================
# Análisis paralelo + síntesis
# ============================================================
# RunnableParallel dispara ambas sub-cadenas a la vez sobre la
# misma entrada. El tiempo total es el de la rama más lenta,
# no la suma. Produce {"sentiment": "...", "topic": "..."}.
analysis = RunnableParallel(sentiment=sentiment_chain, topic=topic_chain)

# Las claves del diccionario de salida de `analysis` matchean las
# variables del siguiente prompt. Eso NO es casualidad: es el
# contrato de LCEL para encadenar paralelo con secuencial.
ticket_prompt = ChatPromptTemplate.from_messages([
    ("system", "Escribís tickets de soporte concisos y accionables."),
    (
        "human",
        "Sentimiento: {sentiment}\nTema: {topic}\n\n"
        "Escribí una sola línea de ticket (máx. 20 palabras) para el equipo."
    ),
])

full_chain = analysis | ticket_prompt | model | parser


async def main() -> int:
    reseña = (
        "Llevo tres días esperando que me respondan el reclamo por un cobro "
        "duplicado y nadie me contesta por ningún canal. Esto es inaceptable."
    )

    print("=" * 70)
    print(f"Proveedor: {PROVIDER}")
    print("Reseña:")
    print(f"  {reseña}")
    print("=" * 70)

    t0 = time.perf_counter()
    ticket = await full_chain.ainvoke({"review": reseña})
    elapsed = time.perf_counter() - t0

    print(f"\nTicket generado ({elapsed:.2f}s):\n{ticket}\n")

    # Para ver las piezas intermedias, también podemos invocar `analysis`
    # directamente — útil para debuggear qué le llega al prompt final.
    print("--- Debug: salida del RunnableParallel ---")
    partes = await analysis.ainvoke({"review": reseña})
    for k, v in partes.items():
        print(f"  {k}: {v}")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
