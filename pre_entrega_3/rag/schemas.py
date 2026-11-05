"""
Esquemas Pydantic para la salida del sistema RAG.
Al forzar esta estructura, el frontend (o cualquier consumidor) nunca recibe
un AIMessage crudo o un string sin estructura.
"""

from typing import List
from pydantic import BaseModel, Field


class RAGReference(BaseModel):
    """Una referencia a un documento usado como fuente."""

    source: str = Field(
        description=(
            "Nombre del archivo fuente del cual se extrajo la información, "
            "por ejemplo: '01_asyncio_python.md'."
        )
    )
    excerpt: str = Field(
        description=(
            "Fragmento literal (no parafraseado) del documento que respalda la "
            "respuesta. Máximo 240 caracteres."
        )
    )


class RAGResponse(BaseModel):
    """
    Respuesta final del sistema RAG.
    Incluye 3 campos: la respuesta en texto, las referencias usadas y un flag
    booleano que indica si hubo o no respuesta en el contexto (anti-alucinación).
    """

    answer: str = Field(
        description=(
            "La respuesta a la pregunta del usuario, basada EXCLUSIVAMENTE en "
            "el contexto provisto. Si la respuesta no está en el contexto, "
            "este campo debe decir: 'No tengo acceso a esa información en mi "
            "base de conocimiento.'"
        )
    )
    references: List[RAGReference] = Field(
        default_factory=list,
        description=(
            "Lista de fragmentos del contexto efectivamente usados para construir "
            "la respuesta. Vacía si la respuesta no estaba en el contexto."
        ),
    )
    has_answer: bool = Field(
        description=(
            "True si se pudo responder a partir del contexto, False si la "
            "información pedida no estaba disponible."
        )
    )
