"""Carga del corpus: archivos markdown con front matter YAML.

La lectura de disco es async (`anyio.Path`), así una ingesta disparada desde
la API no bloquea el event loop. El front matter reemplaza el `CATEGORY_MAP`
hard-codeado de la pre-entrega 4: cada documento declara su propia metadata.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

import anyio
import yaml

from intelligence_system.rag.schemas import Document

_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_H1 = re.compile(r"^#\s+(.+)$", re.MULTILINE)


class CorpusError(ValueError):
    """El corpus no existe o un documento es inválido."""


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9_-]+", "-", value.lower()).strip("-")
    return slug[:64] or "doc"


def parse_markdown(raw: str, *, fallback_id: str, source: str) -> Document:
    """Separa front matter y cuerpo; valida el resultado con `Document`."""
    meta: dict[str, Any] = {}
    body = raw
    match = _FRONT_MATTER.match(raw)
    if match:
        loaded = yaml.safe_load(match.group(1)) or {}
        if not isinstance(loaded, dict):
            raise CorpusError(f"{source}: el front matter debe ser un mapa YAML")
        meta = loaded
        body = raw[match.end() :]
    body = body.strip()
    heading = _H1.search(body)
    title = str(meta.get("title") or (heading.group(1).strip() if heading else fallback_id))
    return Document(
        doc_id=str(meta.get("doc_id") or _slug(fallback_id)),
        title=title,
        text=body,
        source=str(meta.get("source") or source),
        category=str(meta.get("category") or "general"),
        tags=[str(t) for t in meta.get("tags") or []],
    )


async def load_document(path: Path) -> Document:
    raw = await anyio.Path(path).read_text(encoding="utf-8")
    return parse_markdown(raw, fallback_id=path.stem, source=path.name)


async def load_corpus(directory: Path) -> list[Document]:
    """Lee todos los `.md` del directorio en paralelo, en orden determinista."""
    folder = anyio.Path(directory)
    if not await folder.is_dir():
        raise CorpusError(f"no existe el directorio del corpus: {directory}")
    paths = sorted([Path(str(p)) async for p in folder.glob("*.md")])
    if not paths:
        raise CorpusError(f"el corpus está vacío: {directory}")
    documents = await asyncio.gather(*(load_document(p) for p in paths))
    ids = [d.doc_id for d in documents]
    duplicated = {i for i in ids if ids.count(i) > 1}
    if duplicated:
        raise CorpusError(f"doc_id duplicados en el corpus: {sorted(duplicated)}")
    return list(documents)
