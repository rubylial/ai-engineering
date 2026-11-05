"""
Construcción del StateGraph con:
  - MessagesState (hereda el esquema estándar con add_messages como reducer)
  - Ciclo ReAct (agent ↔ tools)
  - Checkpointer async (se inyecta desde afuera)

NOTA: En LangGraph 1.x, el SqliteSaver síncrono NO soporta `ainvoke`
(lanza NotImplementedError). Para flujos async hay que usar AsyncSqliteSaver,
que requiere `async with ...` como context manager. Por eso build_agent
recibe el checkpointer ya construido desde afuera (desde main.py).
"""

from langchain_anthropic import ChatAnthropic
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from .config import get_settings
from .tools import TOOLS


def build_agent(checkpointer):
    """
    Construye y compila el grafo del agente.

    Args:
        checkpointer: instancia de AsyncSqliteSaver (u otro saver async)
                      ya abierta como context manager por el caller.
    Returns:
        La aplicación compilada (CompiledGraph), lista para `ainvoke`.
    """
    settings = get_settings()

    # --- Modelo con tools bindeadas ---
    # bind_tools() le da al LLM el esquema de cada tool (nombre, args,
    # docstring). El modelo decide por sí mismo cuándo invocar cada una.
    # NOTA: temperature omitido porque anthropic SDK >= 2.5.0 lo rechaza.
    model = ChatAnthropic(
        model=settings.llm_model,
        max_tokens=settings.llm_max_tokens,
    ).bind_tools(TOOLS)

    # --- Nodo 'agent': el cerebro del loop ReAct ---
    async def call_model(state: MessagesState) -> dict:
        """Lee TODO el historial y responde (con texto o con tool_calls)."""
        response = await model.ainvoke(state["messages"])
        return {"messages": [response]}

    # --- Nodo 'tools': ejecuta las tools que el agent pidió ---
    tool_node = ToolNode(TOOLS)

    # --- Grafo ---
    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "agent")

    # Arista condicional:
    #   - Si el agent emitió tool_calls → va a "tools"
    #   - Si no → END (el agent decidió que ya terminó)
    workflow.add_conditional_edges("agent", tools_condition)

    # Volvemos al agent después de ejecutar tools. Esto crea el CICLO ReAct.
    workflow.add_edge("tools", "agent")

    return workflow.compile(checkpointer=checkpointer)