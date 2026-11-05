"""
Infraestructura de Pinecone: crear el índice serverless si no existe,
esperar a que esté ready, y proveer el handle del cliente.
"""

import time

from pinecone import Pinecone, ServerlessSpec

from .config import get_settings


def get_pinecone_client() -> Pinecone:
    """Devuelve el cliente de Pinecone con la API key cargada del env."""
    settings = get_settings()
    return Pinecone(api_key=settings.pinecone_api_key)


def ensure_index(timeout: int = 90) -> Pinecone:
    """
    Crea el índice serverless si no existe y espera a que esté ready.
    Es idempotente: si el índice ya existe, no hace nada.

    Returns:
        El cliente de Pinecone ya inicializado.

    Raises:
        TimeoutError si el índice no queda ready dentro del timeout.
    """
    settings = get_settings()
    pc = get_pinecone_client()

    existing = pc.list_indexes().names()
    if settings.pinecone_index_name in existing:
        print(f"✓ Índice '{settings.pinecone_index_name}' ya existe.")
        return pc

    print(
        f"→ Creando índice serverless '{settings.pinecone_index_name}' "
        f"(dim={settings.embedding_dimension}, metric={settings.pinecone_metric}, "
        f"{settings.pinecone_cloud}/{settings.pinecone_region}) ..."
    )
    pc.create_index(
        name=settings.pinecone_index_name,
        dimension=settings.embedding_dimension,
        metric=settings.pinecone_metric,
        spec=ServerlessSpec(
            cloud=settings.pinecone_cloud,
            region=settings.pinecone_region,
        ),
    )

    # Polling hasta que el índice quede ready (serverless suele tardar 10-30s).
    elapsed = 0
    while elapsed < timeout:
        desc = pc.describe_index(settings.pinecone_index_name)
        if desc.status.get("ready", False):
            print(f"✓ Índice listo en {elapsed}s.")
            return pc
        time.sleep(2)
        elapsed += 2

    raise TimeoutError(
        f"Índice '{settings.pinecone_index_name}' no quedó ready en {timeout}s."
    )


def delete_index() -> None:
    """Borra el índice (útil para limpiar al terminar los experimentos)."""
    settings = get_settings()
    pc = get_pinecone_client()
    if settings.pinecone_index_name in pc.list_indexes().names():
        pc.delete_index(settings.pinecone_index_name)
        print(f"✓ Índice '{settings.pinecone_index_name}' eliminado.")
    else:
        print(f"ℹ Índice '{settings.pinecone_index_name}' no existe. Nada que borrar.")


if __name__ == "__main__":
    import sys
    if "--delete" in sys.argv:
        delete_index()
    else:
        ensure_index()
