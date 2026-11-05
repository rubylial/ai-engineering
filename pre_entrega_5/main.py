"""
Entry point: ejecuta dos consultas que demuestran los 4 criterios de aceptación.

  Prueba 1: autonomía + razonamiento multi-paso
    "¿Cuántos pedidos tuvo Juan Pérez y cuál fue el total?"
    → El agente debe llamar PRIMERO a buscar_cliente, DESPUÉS a buscar_pedidos.

  Prueba 2: resiliencia de estado (misma thread_id)
    "¿Y cuál fue el último pedido?"
    → El agente debe recordar que seguimos hablando de Juan Pérez.

  Control: cambiamos el thread_id → el agente queda "ciego".

AsyncSqliteSaver se abre como context manager para todo el run.
"""

import asyncio
import sys

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agente import build_agent, dump_trace, get_settings


def _short(m) -> str:
    """Formato corto de un mensaje para log en consola."""
    if hasattr(m, "tool_calls") and m.tool_calls:
        calls = [f"{tc['name']}({tc['args']})" for tc in m.tool_calls]
        return f"→ tool_calls: {'; '.join(calls)}"
    content = m.content if isinstance(m.content, str) else str(m.content)
    return content[:200] + ("..." if len(content) > 200 else "")


def _print_run(label: str, query: str, result: dict, only_new: int | None = None) -> None:
    print("\n" + "=" * 72)
    print(label)
    print("=" * 72)
    print(f"\n❯ Usuario: {query}\n")
    msgs = result["messages"]
    if only_new is not None:
        msgs = msgs[-only_new:]
    for m in msgs:
        print(f"  [{m.__class__.__name__:<14}] {_short(m)}")


async def main() -> int:
    settings = get_settings()
    THREAD_ID = "cliente-juan-perez-001"

    # El recursion_limit SE PASA en el config, no en compile().
    # Es nuestro guardarraíl contra loops infinitos.
    thread_config = {
        "configurable": {"thread_id": THREAD_ID},
        "recursion_limit": settings.recursion_limit,
    }

    # AsyncSqliteSaver se usa como async context manager.
    # Abre la conexión aiosqlite al entrar y la cierra al salir.
    async with AsyncSqliteSaver.from_conn_string(str(settings.checkpoint_db)) as checkpointer:
        app = build_agent(checkpointer)

        # -----------------------------------------------------------
        # PRUEBA 1 — razonamiento multi-paso
        # -----------------------------------------------------------
        q1 = "¿Cuántos pedidos tuvo Juan Pérez y cuál fue el total?"
        result1 = await app.ainvoke(
            {"messages": [("user", q1)]},
            config=thread_config,
        )
        _print_run(
            "PRUEBA 1 — Autonomía + razonamiento multi-paso (≥2 tool_calls esperadas)",
            q1, result1,
        )
        print(f"\n✓ Respuesta final: {result1['messages'][-1].content}")

        # -----------------------------------------------------------
        # PRUEBA 2 — continuidad gracias al checkpointer
        # -----------------------------------------------------------
        q2 = "¿Y cuál fue el último pedido? Dame el detalle completo."
        result2 = await app.ainvoke(
            {"messages": [("user", q2)]},
            config=thread_config,
        )
        _print_run(
            "PRUEBA 2 — Continuidad: mismo thread_id, pregunta elíptica",
            q2, result2,
            only_new=6,
        )
        print(f"\n✓ Respuesta final: {result2['messages'][-1].content}")

        # -----------------------------------------------------------
        # CONTROL — thread nuevo → el agente no sabe de qué hablamos
        # -----------------------------------------------------------
        other_config = {
            "configurable": {"thread_id": "sesion-sin-memoria"},
            "recursion_limit": settings.recursion_limit,
        }
        r_control = await app.ainvoke(
            {"messages": [("user", q2)]},
            config=other_config,
        )
        _print_run(
            "CONTROL — mismo mensaje con thread_id NUEVO (memoria limpia)",
            q2, r_control,
        )
        print(f"\n✓ Respuesta final: {r_control['messages'][-1].content}")

        # -----------------------------------------------------------
        # Dump de la traza completa
        # -----------------------------------------------------------
        state = await app.aget_state(thread_config)
        trace_path = settings.trace_dir / f"trace_{THREAD_ID}.json"
        dump_trace(
            state.values["messages"],
            trace_path,
            metadata={
                "thread_id": THREAD_ID,
                "model": settings.llm_model,
                "recursion_limit": settings.recursion_limit,
                "queries": [q1, q2],
            },
        )
        print("\n" + "=" * 72)
        print(f"✓ Traza ReAct completa guardada en: {trace_path}")
        print(f"✓ Checkpoints persistentes en: {settings.checkpoint_db}")
        print("=" * 72)

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))