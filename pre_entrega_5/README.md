# Pre-entrega 5 — Agente ReAct con memoria persistente

Agente autónomo implementado con **LangGraph** que razona en un ciclo **ReAct** (Reason → Act → Observe), invoca herramientas por su cuenta sin rutas `if/else` manuales, y persiste su estado en SQLite para poder retomar conversaciones vía `thread_id`.

## Checklist de entrega

| Item del enunciado | Dónde |
|---|---|
| Repo público con README y variables de entorno | este repo + `.env.example` |
| `StateGraph` que hereda de `MessagesState` | `agente/graph.py` (`StateGraph(MessagesState)`) |
| Nodo de modelo + nodo de herramientas + `tools_condition` | `agente/graph.py` (nodos `agent` y `tools` + `add_conditional_edges("agent", tools_condition)`) |
| ≥ 1 herramienta propia con `@tool` + docstring descriptivo | `agente/tools.py` (3 tools: `buscar_cliente`, `buscar_pedidos`, `obtener_detalle_pedido`) |
| Persistencia con `SqliteSaver` + `thread_id` | `agente/graph.py` (`SqliteSaver(sqlite3.connect(...))`) |
| Razonamiento multi-paso (herramienta se invoca ≥2 veces) | `main.py` (Prueba 1 diseñada para forzar 2 tool-calls encadenados) |
| `recursion_limit` definido | `agente/config.py` → pasado en `thread_config` |
| Traza de ejecución como JSON/log | `traces/example_trace.json` (incluida) + generación automática en `main.py` |

## Stack

| Componente | Tecnología |
|---|---|
| Framework agente | LangGraph |
| LLM | `claude-haiku-4-5` (Anthropic) |
| Checkpointer | `SqliteSaver` (persistencia a disco, compatible con asyncio) |
| Validación de tools | Pydantic |

> El enunciado permite OpenAI o Anthropic. Usamos **Anthropic** porque es el único proveedor disponible en el entorno. El patrón ReAct + `MessagesState` + `SqliteSaver` es idéntico para cualquier LLM.

## Estructura

```
pre-entrega-5/
├── agente/
│   ├── __init__.py
│   ├── config.py                ← Settings (Pydantic, lee .env)
│   ├── tools.py                 ← 3 tools encadenables con @tool + docstrings
│   ├── graph.py                 ← StateGraph(MessagesState) + SqliteSaver
│   └── tracer.py                ← Serialización de la traza ReAct a JSON
├── traces/
│   └── example_trace.json       ← Ejemplo de traza (comiteada)
├── main.py                      ← Demo con 2 pruebas + 1 control
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Setup

Requiere **Python 3.12+**.

```bash
python3.12 -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Editar .env: cargar ANTHROPIC_API_KEY
```

## Cómo correrlo

```bash
python main.py
```

Esto ejecuta tres queries en el mismo proceso:

1. **Prueba 1** (razonamiento multi-paso):
   `"¿Cuántos pedidos tuvo Juan Pérez y cuál fue el total?"`
   El agente debe llamar **primero** a `buscar_cliente("Juan Pérez")` para obtener el ID, **después** a `buscar_pedidos(cliente_id=102)` para el resumen. Son **≥ 2 llamadas** sin que ningún `if` las coordine — es el LLM decidiendo.

2. **Prueba 2** (resiliencia de estado):
   `"¿Y cuál fue el último pedido? Dame el detalle completo."`
   Misma `thread_id`. El agente **recuerda** que seguimos hablando de Juan Pérez (su cliente_id está en el historial) y llama a `obtener_detalle_pedido(pedido_id=1020)`.

3. **Control** (prueba negativa):
   La misma pregunta pero con un `thread_id` nuevo → el agente responde que no sabe de qué le estamos hablando. Demuestra que la memoria vive en el thread, no en una variable global.

Al final se escribe:

- `checkpoints.sqlite` — todos los checkpoints del proceso (persistentes).
- `traces/trace_<thread_id>.json` — la traza ReAct completa serializada.

## Las 3 tools diseñadas para forzar razonamiento multi-paso

| Tool | Entrada | Salida | Rol en el ciclo |
|---|---|---|---|
| `buscar_cliente(nombre)` | `str` | `{cliente_id, nombre, email}` | "Entry point" cuando el usuario menciona un cliente por nombre |
| `buscar_pedidos(cliente_id)` | `int` | `{pedidos, total, ultimo_pedido_id}` | Requiere `cliente_id` — fuerza llamar antes a `buscar_cliente` |
| `obtener_detalle_pedido(pedido_id)` | `int` | `{id, fecha, monto}` | Requiere `pedido_id` — fuerza llamar antes a `buscar_pedidos` |

**La dependencia entre tools es intencional**: fuerza al agente a planear 2+ pasos en lugar de responder con una sola invocación.

## Decisiones de diseño defendibles

### 1. `MessagesState` en vez de TypedDict propio

`MessagesState` es el esquema estándar de LangGraph: ya trae `messages: Annotated[list, add_messages]` configurado. Heredar de él en lugar de redefinirlo evita bugs sutiles (el error más común: olvidar el reducer y que cada turno borre el historial).

### 2. `SqliteSaver(sqlite3.connect(..., check_same_thread=False))`

El flag `check_same_thread=False` es necesario porque con `asyncio` la conexión SQLite es usada desde distintos "threads lógicos" del event loop. Sin ese flag, SQLite tira error en producción. Para una app realmente multi-worker, se cambiaría a `PostgresSaver` o `RedisSaver`.

### 3. `recursion_limit` en el config, no hardcodeado

El enunciado lo pide explícitamente: sin tope, si el modelo entra en bucle (ej: llama a la misma tool indefinidamente) el gasto en tokens puede ser enorme. 15 iteraciones es suficiente para cualquier razonamiento de ≤ 7 tool-calls encadenados + los pares AIMessage/ToolMessage.

### 4. Tool errors devueltos como dict `{"error": "..."}`, NO como excepción

Si una tool levantara una excepción, el agente entero crashearía. Al devolver `{"error": "..."}`, el LLM lee el error en el próximo paso y puede **reintentar con otros parámetros** o pedirle aclaraciones al usuario — exactamente el "ciclo de retorno" que pide el criterio #2 del enunciado.

### 5. Docstrings con secciones "USÁ ESTA TOOL cuando:" / "NO LA USES si:"

El LLM elige si llama a una tool basándose solo en el docstring. Una descripción vaga es el error #1 que menciona el enunciado. Los docstrings de las 3 tools explicitan cuándo corresponde usarlas y cuándo no.

### 6. Traza serializada a JSON

El `tracer.py` convierte `BaseMessage` → dicts legibles. Permite auditar el razonamiento del agente sin re-correrlo. En producción esta traza iría a CloudWatch / Datadog / LangSmith.

## Ejemplo de traza esperada (razonamiento ReAct completo)

Ver `traces/example_trace.json`. Resumen de lo que verás al correr `python main.py`:

```
Usuario: "¿Cuántos pedidos tuvo Juan Pérez y cuál fue el total?"
  ↓
Agent razona: "necesito el cliente_id, llamo a buscar_cliente"
Tool call: buscar_cliente(nombre="Juan Pérez")
Observation: {"cliente_id": 102, ...}
  ↓
Agent razona: "ahora tengo el ID, llamo a buscar_pedidos"
Tool call: buscar_pedidos(cliente_id=102)
Observation: {"pedidos": 3, "total": 14500, "ultimo_pedido_id": 1020}
  ↓
Agent razona: "ya tengo todo, puedo responder"
Respuesta: "Juan Pérez (cliente 102) tuvo 3 pedidos por un total de $14.500."
```

## Troubleshooting

**`sqlite3.OperationalError: database is locked`**
El checkpointer intenta escribir mientras algo más (otro proceso o el DB browser) tiene abierta la DB. Cerrá cualquier herramienta que tenga `checkpoints.sqlite` abierta, o borrala: `rm checkpoints.sqlite`.

**El agente responde directamente sin llamar a las tools**
Puede que su conocimiento interno responda a la query sin necesidad del RAG. Probá con una query que **requiera datos específicos del mock** (p. ej. un cliente que solo existe en nuestro DB).

**"Extra inputs are not permitted" al llamar a Claude**
El SDK de anthropic ≥ 2.5.0 removió `temperature` y `top_p` del endpoint. El código provisto ya los omite. Si aparece con otro parámetro, revisá `build_agent()` en `graph.py`.

**Quiero empezar de cero (borrar checkpoints)**
```bash
rm checkpoints.sqlite traces/trace_*.json
python main.py
```
