"""Mini knowledge base local, para que el researcher tenga algo que buscar sin red.

En un sistema real esto saldría de un vector store (Pinecone, Chroma) o de Tavily
(búsqueda web). Lo dejamos como un dict literal para que el prototipo corra offline
y sea 100% reproducible en los tests.
"""

from __future__ import annotations

KB_DOCS: list[dict[str, str]] = [
    {
        "id": "doc:langgraph-supervisor",
        "title": "Supervisor en LangGraph",
        "category": "agents",
        "content": (
            "El patrón supervisor en LangGraph consiste en un nodo central que decide, "
            "en cada paso, qué especialista debe trabajar. El supervisor usa salida "
            "estructurada (tool calling) para devolver el nombre del próximo nodo. Los "
            "workers vuelven al supervisor después de cada turno, formando un ciclo que "
            "termina cuando el supervisor devuelve FINISH o se agota un presupuesto de pasos."
        ),
    },
    {
        "id": "doc:langgraph-state",
        "title": "Estado en LangGraph",
        "category": "agents",
        "content": (
            "El estado de un grafo LangGraph es un TypedDict. Los campos con reducer "
            "(Annotated[list[X], operator.add] o add_messages) acumulan los deltas que "
            "escribe cada nodo. Los campos sin reducer se sobreescriben con el último "
            "valor. MessagesState provee el campo `messages` ya anotado con add_messages."
        ),
    },
    {
        "id": "doc:create-react-agent",
        "title": "create_react_agent",
        "category": "agents",
        "content": (
            "create_react_agent de langgraph.prebuilt construye un ciclo ReAct completo: "
            "el modelo decide si llamar a una tool o responder, y el bucle corre hasta "
            "que el modelo no pida más tools. Recibe el chat model, la lista de tools y "
            "un prompt (string o ChatPromptTemplate) con la descripción del rol."
        ),
    },
    {
        "id": "doc:fastapi",
        "title": "FastAPI y async",
        "category": "api",
        "content": (
            "FastAPI expone endpoints async sobre Starlette. Uvicorn con workers async "
            "no paraleliza por proceso: usa un solo event loop por worker. El cuello de "
            "botella típico es bloquear el loop con llamadas síncronas (requests, "
            "time.sleep) o agotar el pool de conexiones de la base de datos."
        ),
    },
    {
        "id": "doc:redis",
        "title": "Redis 8 y módulos",
        "category": "cache",
        "content": (
            "Redis 8 incluye RedisJSON y RediSearch de fábrica (antes eran módulos "
            "externos). AsyncRedisSaver de langgraph-checkpoint-redis necesita estos "
            "módulos para guardar checkpoints. El modo AOF (appendonly=yes) hace que la "
            "persistencia sobreviva reinicios del contenedor."
        ),
    },
    {
        "id": "doc:rag-hibrido",
        "title": "RAG híbrido denso + BM25",
        "category": "retrieval",
        "content": (
            "El retrieval híbrido combina un vector denso (semántico) con BM25 "
            "(léxico). Pinecone 10 permite hacerlo en una sola query definiendo un "
            "índice con campos reservados `_values` (dotproduct) y `_sparse_values`. "
            "La combinación convexa `alpha*denso + (1-alpha)*BM25` suele dar mejor "
            "P@k que cualquiera de los dos solo."
        ),
    },
    {
        "id": "doc:pydantic-v2",
        "title": "Pydantic v2 en los bordes",
        "category": "validation",
        "content": (
            "Pydantic v2 es rápido (validación en Rust) y estricto por defecto. El "
            "patrón recomendado es validar en los bordes del sistema: parámetros de "
            "API, argumentos de tools (args_schema) y salida estructurada del LLM "
            "(with_structured_output). Adentro, dataclasses o TypedDicts alcanzan."
        ),
    },
]


def search(query: str, *, k: int = 3) -> list[dict[str, str]]:
    """Búsqueda léxica muy simple sobre el KB: cuenta términos en title + content.

    Devuelve hasta `k` documentos ordenados por cantidad de términos de la consulta
    que aparecen en el documento (case-insensitive). Si ninguno matchea, lista vacía.
    """
    if not query.strip():
        return []
    terms = [t.lower() for t in query.split() if len(t) > 2]
    if not terms:
        return []

    def score(doc: dict[str, str]) -> int:
        blob = (doc["title"] + " " + doc["content"]).lower()
        return sum(blob.count(term) for term in terms)

    ranked = sorted(((score(d), d) for d in KB_DOCS), key=lambda p: p[0], reverse=True)
    return [doc for s, doc in ranked[:k] if s > 0]
