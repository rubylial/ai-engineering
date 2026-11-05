"""
Herramientas del agente: simulan operaciones sobre una "base de datos" de
clientes y pedidos.

Diseño intencional: tres tools que SE ENCADENAN. Para responder "¿cuántos
pedidos tuvo Juan Pérez?" el agente necesita:
  1. buscar_cliente("Juan Pérez")       → obtiene cliente_id
  2. buscar_pedidos(cliente_id)         → obtiene resumen

Esto fuerza el razonamiento multi-paso (≥2 llamadas) que pide el enunciado.
"""

from langchain_core.tools import tool
from pydantic import BaseModel, Field


# ============================================================
# Mock de la base de datos
# ============================================================

_CLIENTES: dict[int, dict] = {
    101: {"nombre": "María García", "email": "maria@acme.com"},
    102: {"nombre": "Juan Pérez",   "email": "juan@acme.com"},
    103: {"nombre": "Ana Ruiz",     "email": "ana@acme.com"},
}

_PEDIDOS: dict[int, list[dict]] = {
    101: [
        {"id": 1001, "fecha": "2026-10-15", "monto": 5200},
        {"id": 1015, "fecha": "2026-10-28", "monto": 3100},
    ],
    102: [
        {"id": 1002, "fecha": "2026-09-30", "monto": 4800},
        {"id": 1008, "fecha": "2026-10-20", "monto": 6200},
        {"id": 1020, "fecha": "2026-11-02", "monto": 3500},
    ],
    103: [],  # Cliente sin pedidos (caso borde)
}

# Índice inverso para búsqueda por nombre (case-insensitive)
_NOMBRE_A_ID: dict[str, int] = {c["nombre"].lower(): cid for cid, c in _CLIENTES.items()}


# ============================================================
# Tool 1: buscar cliente por nombre → cliente_id
# ============================================================

class BuscarClienteInput(BaseModel):
    nombre: str = Field(
        ...,
        description="Nombre completo del cliente, ej: 'Juan Pérez', 'María García'.",
    )


@tool(args_schema=BuscarClienteInput)
async def buscar_cliente(nombre: str) -> dict:
    """Busca un cliente por su nombre completo en la base de datos.

    USÁ ESTA TOOL cuando:
      - El usuario menciona un cliente por nombre (ej: "pedidos de Juan Pérez")
      - Necesitás el cliente_id para después llamar a otras tools como
        buscar_pedidos, que requieren el ID numérico.

    NO LA USES si el usuario ya te dio el ID numérico (ej: "cliente 102").

    Devuelve un dict con `cliente_id`, `nombre` y `email`.
    Si el cliente no existe, devuelve `{"error": "..."}`.
    """
    cid = _NOMBRE_A_ID.get(nombre.lower().strip())
    if cid is None:
        return {"error": f"No se encontró un cliente llamado '{nombre}'."}
    cliente = _CLIENTES[cid]
    return {
        "cliente_id": cid,
        "nombre": cliente["nombre"],
        "email": cliente["email"],
    }


# ============================================================
# Tool 2: buscar pedidos de un cliente → resumen
# ============================================================

class BuscarPedidosInput(BaseModel):
    cliente_id: int = Field(
        ...,
        description="ID numérico del cliente (ej: 101, 102, 103).",
    )


@tool(args_schema=BuscarPedidosInput)
async def buscar_pedidos(cliente_id: int) -> dict:
    """Devuelve el resumen de pedidos de un cliente específico.

    USÁ ESTA TOOL cuando necesites saber cuántos pedidos tiene un cliente
    o cuál fue el total facturado.

    REQUIERE el `cliente_id` (int). Si solo tenés el NOMBRE del cliente,
    llamá PRIMERO a `buscar_cliente` para obtener su ID.

    Devuelve `{"pedidos": int, "total": float, "ultimo_pedido_id": int | None}`.
    Si el cliente no tiene pedidos, devuelve ceros y ultimo_pedido_id=None.
    """
    if cliente_id not in _PEDIDOS:
        return {"error": f"Cliente ID {cliente_id} no existe."}
    pedidos = _PEDIDOS[cliente_id]
    if not pedidos:
        return {"pedidos": 0, "total": 0, "ultimo_pedido_id": None}
    return {
        "pedidos": len(pedidos),
        "total": sum(p["monto"] for p in pedidos),
        "ultimo_pedido_id": pedidos[-1]["id"],
    }


# ============================================================
# Tool 3: detalle de un pedido específico
# ============================================================

class DetallePedidoInput(BaseModel):
    pedido_id: int = Field(..., description="ID del pedido, ej: 1008, 1020.")


@tool(args_schema=DetallePedidoInput)
async def obtener_detalle_pedido(pedido_id: int) -> dict:
    """Obtiene el detalle completo de un pedido específico.

    USÁ ESTA TOOL cuando el usuario pregunte por un pedido en particular
    (ej: "¿cuánto fue el último pedido?", "detalles del pedido 1020").

    REQUIERE el `pedido_id`. Si solo tenés el cliente, llamá primero a
    `buscar_pedidos` para obtener el `ultimo_pedido_id`.

    Devuelve `{"id": int, "fecha": str, "monto": float}` o `{"error": "..."}`.
    """
    for pedidos in _PEDIDOS.values():
        for p in pedidos:
            if p["id"] == pedido_id:
                return p
    return {"error": f"Pedido {pedido_id} no encontrado."}


# ============================================================
# Export
# ============================================================
TOOLS = [buscar_cliente, buscar_pedidos, obtener_detalle_pedido]
