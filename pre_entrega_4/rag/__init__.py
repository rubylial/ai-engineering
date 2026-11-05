"""Paquete RAG escalable: Pinecone + Hybrid Retriever + evaluación."""

from .config import Settings, get_settings
from .pinecone_setup import ensure_index, get_pinecone_client, delete_index
from .ingest import ingest_data_directory
from .retriever import RAGSystem

__all__ = [
    "Settings",
    "get_settings",
    "ensure_index",
    "get_pinecone_client",
    "delete_index",
    "ingest_data_directory",
    "RAGSystem",
]
