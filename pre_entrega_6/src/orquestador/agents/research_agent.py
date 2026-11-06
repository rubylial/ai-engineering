"""Research agent: busca en el knowledge base local y devuelve un resumen con citas.

El nodo del grafo (`build_research_node`) hace dos cosas clave para la topología:
1. Toma del state solo `task` (la instrucción del supervisor), NO todo el historial.
   Esto es lo que el enunciado llama "evitar la contaminación de contexto".
2. Vuelve al supervisor registrando una `Contribution` con lo que aportó, para que
   el supervisor pueda decidir con base en el historial estructurado.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from orquestador.agents.knowledge_base import search
from orquestador.state import Contribution, SupervisorState

RESEARCH_PROMPT = (
    "Sos un investigador. Tu única herramienta es `search_knowledge_base`. "
    "Usala al menos una vez para contestar la tarea del supervisor. "
    "Resumí los hallazgos en 2-4 oraciones citando los ids de los docs usados "
    "(formato: `[doc:xxx]`). Si no encontrás nada, decilo explícitamente."
)


@tool
def search_knowledge_base(query: str, k: int = 3) -> str:
    """Busca en la base de conocimiento del curso AI Engineering.

    Útil para preguntas sobre LangGraph, LangChain, FastAPI, Redis, RAG, Pydantic, etc.
    Devuelve hasta `k` documentos (default 3) con id, título y contenido. Si no hay
    hits, devuelve una cadena fija ("No hay documentos relevantes").

    Args:
        query: términos de búsqueda, en español o inglés.
        k: cuántos documentos devolver como máximo.
    """
    hits = search(query, k=k)
    if not hits:
        return "No hay documentos relevantes."
    return "\n\n".join(f"[{doc['id']}] {doc['title']}\n{doc['content']}" for doc in hits)


RESEARCH_TOOLS = [search_knowledge_base]


def build_research_agent(llm: BaseChatModel) -> Any:
    """Arma el ciclo ReAct con la tool de búsqueda."""
    return create_react_agent(llm, tools=RESEARCH_TOOLS, prompt=RESEARCH_PROMPT)


def build_research_node(
    agent: Any,
) -> Callable[[SupervisorState], Awaitable[dict[str, Any]]]:
    """Envuelve el agente ReAct como nodo del supervisor-graph.

    Le pasamos solo `task` (la instrucción del supervisor), no todo el historial.
    """

    async def _node(state: SupervisorState) -> dict[str, Any]:
        task = state.get("task") or _fallback_task(state)
        result = await agent.ainvoke({"messages": [HumanMessage(content=task)]})
        reply = result["messages"][-1].content
        text = reply if isinstance(reply, str) else str(reply)
        contribution: Contribution = {
            "agent": "researcher",
            "step": state.get("step", 0),
            "summary": text[:280],
        }
        return {
            "messages": [AIMessage(content=f"[researcher] {text}", name="researcher")],
            "contributions": [contribution],
        }

    return _node


def _fallback_task(state: SupervisorState) -> str:
    """Si el supervisor no puso una tarea (ej. primer turno simulado), usa el último msg."""
    messages = state.get("messages", [])
    if not messages:
        return "Resumí los documentos disponibles del knowledge base."
    last = messages[-1]
    return last.content if isinstance(last.content, str) else str(last.content)
