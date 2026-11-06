"""entity-pipeline: extrae entidades técnicas de texto con LCEL + Pydantic v2."""

from entity_pipeline.chain import build_chain, process_text
from entity_pipeline.schemas import NivelCriticidad, TechExtraction

__all__ = ["NivelCriticidad", "TechExtraction", "build_chain", "process_text"]
