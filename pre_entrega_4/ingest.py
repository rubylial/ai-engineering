"""
Script principal de ingesta.
Equivalente a `python -m rag.ingest` pero más corto de tipear.
"""

import sys
from rag.ingest import ingest_data_directory


if __name__ == "__main__":
    force = "--force" in sys.argv
    ingest_data_directory(force_reindex=force)
