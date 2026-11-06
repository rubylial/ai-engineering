"""Analyst agent: procesa los datos recolectados con 3 tools de cómputo determinista.

Las tools devuelven números exactos (statistics stdlib). El prompt le dice al agente
que NO los redondee ni invente: así validamos que el modelo use las tools en vez de
"inventar" la cuenta en el prompt.
"""

from __future__ import annotations

import statistics
from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from orquestador.state import Contribution, SupervisorState

ANALYST_PROMPT = (
    "Sos un analista. Tenés tres herramientas: `contar_menciones`, "
    "`calcular_estadisticas` y `analizar_sentimiento`. Usá al menos una tool "
    "para responder la tarea del supervisor. Reportá los resultados con los "
    "números EXACTOS que devolvió la tool (no redondees, no inventes). "
    "Si la tarea no se puede resolver con estas herramientas, decilo."
)


@tool
def contar_menciones(texto: str, terminos: list[str]) -> dict[str, int]:
    """Cuenta cuántas veces aparece cada término en el texto (case-insensitive).

    Args:
        texto: texto plano sobre el que buscar.
        terminos: lista de términos a contar.
    """
    lowered = texto.lower()
    return {term: lowered.count(term.lower()) for term in terminos}


@tool
def calcular_estadisticas(numeros: list[float]) -> dict[str, float | int]:
    """Devuelve n, media, mediana, min, max y desvío estándar de una lista de números.

    Args:
        numeros: lista no vacía de floats o ints.
    """
    if not numeros:
        return {"error": -1, "n": 0}  # señal explícita de lista vacía
    return {
        "n": len(numeros),
        "media": float(statistics.fmean(numeros)),
        "mediana": float(statistics.median(numeros)),
        "min": float(min(numeros)),
        "max": float(max(numeros)),
        "desvio": float(statistics.pstdev(numeros)) if len(numeros) > 1 else 0.0,
    }


_POS = {
    "bueno",
    "excelente",
    "exitoso",
    "mejora",
    "rápido",
    "rapido",
    "estable",
    "robusto",
    "eficiente",
    "funciona",
    "resuelve",
}
_NEG = {
    "malo",
    "falla",
    "lento",
    "error",
    "crítico",
    "critico",
    "cae",
    "bloqueado",
    "roto",
    "problema",
    "satura",
    "saturación",
    "saturacion",
}


@tool
def analizar_sentimiento(texto: str) -> dict[str, object]:
    """Clasifica el sentimiento del texto en 'positivo', 'negativo' o 'neutro'.

    Heurística léxica en español sobre palabras clave técnicas (saturación, estable…).
    No usa LLM: es determinista y rápida.
    """
    lowered = texto.lower()
    pos = sum(1 for word in _POS if word in lowered)
    neg = sum(1 for word in _NEG if word in lowered)
    if pos == neg:
        sentimiento = "neutro"
    elif pos > neg:
        sentimiento = "positivo"
    else:
        sentimiento = "negativo"
    return {"sentimiento": sentimiento, "positivas": pos, "negativas": neg}


ANALYST_TOOLS = [contar_menciones, calcular_estadisticas, analizar_sentimiento]


def build_analyst_agent(llm: BaseChatModel) -> Any:
    """Arma el ciclo ReAct del analista con las tres tools de cómputo."""
    return create_react_agent(llm, tools=ANALYST_TOOLS, prompt=ANALYST_PROMPT)


def build_analyst_node(
    agent: Any,
) -> Callable[[SupervisorState], Awaitable[dict[str, Any]]]:
    """Envuelve el agente ReAct como nodo del supervisor-graph."""

    async def _node(state: SupervisorState) -> dict[str, Any]:
        task = state.get("task") or "Procesá los datos disponibles con las tools."
        result = await agent.ainvoke({"messages": [HumanMessage(content=task)]})
        reply = result["messages"][-1].content
        text = reply if isinstance(reply, str) else str(reply)
        contribution: Contribution = {
            "agent": "analyst",
            "step": state.get("step", 0),
            "summary": text[:280],
        }
        return {
            "messages": [AIMessage(content=f"[analyst] {text}", name="analyst")],
            "contributions": [contribution],
        }

    return _node
