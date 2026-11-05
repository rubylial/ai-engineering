"""
Pre-entrega 2 — Refactorización a LCEL Asíncrono
Curso: AI Engineering

Reemplaza la llamada manual al SDK de la pre-entrega 1 por una cadena
LCEL declarativa: Prompt | Model | Parser, ejecutada asíncronamente.

Puntos clave que demuestra:
- Composición con el operador | (pipe) — no hay lógica imperativa.
- ChatPromptTemplate con roles System/Human (no un string plano).
- Ejecución asíncrona con await chain.ainvoke(...).
- Salida en texto plano gracias a StrOutputParser.
- Factory por proveedor (OpenAI / Anthropic) sin cambiar la cadena.
"""

import asyncio
import os
import sys
import time

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable


# ============================================================
# 1. Setup
# ============================================================
load_dotenv()

PROVIDER = os.getenv("LLM_PROVIDER", "anthropic").lower().strip()


def build_model() -> Runnable:
    """
    Factory del modelo según la variable LLM_PROVIDER.

    La belleza de LCEL: el resto de la cadena (prompt | model | parser)
    no cambia una sola línea al alternar entre OpenAI y Anthropic, porque
    ambos clientes implementan el mismo protocolo Runnable.
    """
    if PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model="claude-haiku-4-5",
            max_tokens=512,
        )
    if PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            max_tokens=512,
        )
    raise ValueError(
        f"LLM_PROVIDER inválido: '{PROVIDER}'. Usá 'openai' o 'anthropic'."
    )


# ============================================================
# 2. Prompt con roles (System + Human)
# ============================================================
# Un ChatPromptTemplate con roles es muy superior a un string plano:
# el 'system' es un espacio protegido donde el modelo toma instrucciones
# de comportamiento con más peso que las del turno del usuario.

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Sos un profesor de física que explica conceptos complejos "
        "con claridad y brevedad. Respondé siempre en español neutro, "
        "en un máximo de 4 oraciones. No uses fórmulas matemáticas."
    ),
    (
        "human",
        "{pregunta}"
    ),
])


# ============================================================
# 3. Composición de la cadena LCEL
# ============================================================
# Tres Runnables conectados con `|`. La salida de uno se convierte
# en la entrada del siguiente:
#   dict -> prompt -> ChatPromptValue -> model -> AIMessage -> parser -> str

model = build_model()
parser = StrOutputParser()

chain: Runnable = prompt | model | parser


# ============================================================
# 4. Ejecución asíncrona
# ============================================================

async def main() -> int:
    pregunta = "¿Qué es la entropía y por qué es importante en termodinámica?"

    print("=" * 70)
    print(f"Proveedor: {PROVIDER}")
    print(f"Pregunta:  {pregunta}")
    print("=" * 70)

    t0 = time.perf_counter()

    # ainvoke = la versión async de invoke.
    # NUNCA olvides el `await`: sin él obtenés una corrutina en lugar
    # del resultado, y es el bug #1 al empezar con LCEL.
    respuesta = await chain.ainvoke({"pregunta": pregunta})

    elapsed = time.perf_counter() - t0

    print(f"\nRespuesta ({elapsed:.2f}s):\n{respuesta}\n")

    # Validación rápida: tiene que ser un str plano, NO un AIMessage.
    assert isinstance(respuesta, str), (
        f"Se esperaba str (gracias a StrOutputParser), llegó {type(respuesta)}"
    )
    print("✓ Salida validada como str plano (no AIMessage).")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
