"""The public AgentRunner seam."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, Optional


def build_agent_query(agent_input: Mapping[str, Any]) -> str:
    """Build the canonical natural-language query sent to an Agent."""

    parts: list[str] = []
    if agent_input.get("data_path"):
        parts.append(f"Load the data file: {agent_input['data_path']}")
    if agent_input.get("label_path"):
        parts.append(f"Load the label file: {agent_input['label_path']}")
    parts.append(f"Instruction:\n{agent_input.get('instruction', '')}")
    return "\n\n".join(parts)



@dataclass(frozen=True)
class AgentRunResult:
    """Final natural-language response and target-model token usage."""

    response: str
    tokens: int = 0
    audit: Optional[Mapping[str, Any]] = None


class AgentRunner(ABC):
    """Interface implemented by an evaluated Agent or execution paradigm.

    ``query`` is the complete canonical text built from the case's structured
    ``agent_input``. Implementations can send it directly to their Agent.
    """

    evaluation_mode = "agent"

    def __call__(
        self,
        query: str,
        run_context: Optional[Any] = None,
    ) -> str:
        return self.run_with_usage(query, run_context).response

    @abstractmethod
    def run_with_usage(
        self,
        query: str,
        run_context: Optional[Any] = None,
    ) -> AgentRunResult:
        """Run one instance and return its final report."""
