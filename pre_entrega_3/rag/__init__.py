"""Paquete RAG: ingesta, retrieval y generación grounded."""

from .schemas import RAGResponse, RAGReference
from .config import Settings, get_settings
from .ingest import ingest_data_directory, get_vectorstore
from .chain import get_rag_response

__all__ = [
    "RAGResponse",
    "RAGReference",
    "Settings",
    "get_settings",
    "ingest_data_directory",
    "get_vectorstore",
    "get_rag_response",
]
