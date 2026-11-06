"""Tools de los workers, todas con `args_schema` Pydantic."""

from intelligence_system.agents.tools.domain import make_domain_tools
from intelligence_system.agents.tools.knowledge import (
    SearchKnowledgeBaseInput,
    SearchKnowledgeBaseOutput,
    make_search_tool,
)

__all__ = [
    "SearchKnowledgeBaseInput",
    "SearchKnowledgeBaseOutput",
    "make_domain_tools",
    "make_search_tool",
]
