"""
Cadena RAG asíncrona usando LCEL.

Flujo:
    pregunta
        │
        ├──→ retriever (top-k por similitud)
        │        │
        │        ▼
        │   [docs]  ──→ format_docs ──→ {context: str}
        │
        └──→ {question: pregunta}

    {context, question, format_instructions} ──→ prompt ──→ Claude ──→ PydanticOutputParser ──→ RAGResponse
"""

from typing import List

from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

from .config import get_settings
from .ingest import get_vectorstore
from .schemas import RAGResponse


# ============================================================
# Prompt: filtro de veracidad explícito
# ============================================================
# El "contrato" grounded vive acá: el modelo NO puede inventar.
# El formato Pydantic se inyecta vía format_instructions.
SYSTEM_PROMPT = """\
Eres un asistente técnico especializado en ingeniería de IA. Respondes \
EXCLUSIVAMENTE basándote en el CONTEXTO que te doy a continuación. \
Reglas estrictas:

1. Si la respuesta a la pregunta NO está en el contexto, el campo `answer` \
   debe decir: "No tengo acceso a esa información en mi base de conocimiento.", \
   `has_answer` debe ser false y `references` debe ser una lista vacía.
2. Si la respuesta SÍ está en el contexto, respondé de forma clara y concisa, \
   llená `references` con los fragmentos usados y poné `has_answer` en true.
3. NUNCA inventes información ni completes con conocimiento propio: aunque \
   la pregunta sea sobre algo que sabés, si no está en el contexto, no respondés.
4. Las referencias deben citar el nombre del archivo fuente (source) \
   exactamente como aparece en el contexto.
5. Devolvé la respuesta EXCLUSIVAMENTE como un JSON válido que cumpla el \
   esquema que se te muestra abajo. No incluyas texto antes ni después del JSON.

{format_instructions}
"""

HUMAN_PROMPT = """\
CONTEXTO:
---
{context}
---

PREGUNTA:
{question}
"""


# ============================================================
# Helper: formatear los documentos recuperados como texto
# ============================================================

def _format_docs(docs: List[Document]) -> str:
    """
    Convierte los documentos del retriever en un string plano con separadores
    y marcas de fuente. El modelo usa estas marcas para llenar `references`.
    """
    if not docs:
        return "(No se encontraron documentos relevantes.)"
    parts = []
    for i, d in enumerate(docs, 1):
        source = d.metadata.get("source", "desconocido")
        parts.append(f"[Fragmento {i} | source: {source}]\n{d.page_content}")
    return "\n\n".join(parts)


# ============================================================
# Fábrica de la cadena (singleton diferido)
# ============================================================

_chain_cache = None


def _build_llm() -> ChatAnthropic:
    """
    Crea el cliente de Claude.

    NOTA: en el SDK de anthropic >= 2.5.0, los parámetros `temperature` y
    `top_p` fueron removidos de messages.create() y el SDK los rechaza como
    "Extra inputs are not permitted". Por eso no los pasamos acá. Para RAG
    esto no es un problema: Claude por defecto ya es bastante determinista.
    """
    settings = get_settings()
    return ChatAnthropic(
        model=settings.llm_model,
        max_tokens=settings.llm_max_tokens,
    )


def _build_chain():
    """
    Construye la cadena LCEL una sola vez.
    Usa PydanticOutputParser como pide el enunciado.
    """
    settings = get_settings()

    # --- Retriever: busca top-k por similitud ---
    retriever = get_vectorstore().as_retriever(
        search_kwargs={"k": settings.top_k}
    )

    # --- Parser Pydantic ---
    # format_instructions es un string que explica al modelo qué JSON producir.
    parser = PydanticOutputParser(pydantic_object=RAGResponse)
    format_instructions = parser.get_format_instructions()

    # --- Prompt con format_instructions ya resuelto ---
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_PROMPT),
    ]).partial(format_instructions=format_instructions)

    # --- LLM: Claude ---
    llm = _build_llm()

    # --- Cadena LCEL ---
    # RunnableParallel arma el dict {context, question} que el prompt necesita.
    # El retriever se compone con format_docs usando pipe también.
    chain = (
        RunnableParallel({
            "context": retriever | RunnableLambda(_format_docs),
            "question": RunnablePassthrough(),
        })
        | prompt
        | llm
        | parser
    ).with_retry(stop_after_attempt=3)  # resiliencia ante rate limits / 5xx

    return chain


def _get_chain():
    """Cachea la cadena para no reconstruirla en cada llamada."""
    global _chain_cache
    if _chain_cache is None:
        _chain_cache = _build_chain()
    return _chain_cache


# ============================================================
# API pública
# ============================================================

async def get_rag_response(query: str) -> RAGResponse:
    """
    Punto de entrada principal. Dada una pregunta del usuario, devuelve una
    RAGResponse validada (answer + references + has_answer).

    Args:
        query: pregunta del usuario en lenguaje natural.

    Returns:
        RAGResponse: objeto validado por Pydantic.
                     Si la pregunta no se puede responder con el contexto,
                     has_answer=False y references=[].

    Raises:
        anthropic.APIError u otros errores de la API tras agotar los 3
        reintentos de with_retry.
    """
    chain = _get_chain()
    return await chain.ainvoke(query)
