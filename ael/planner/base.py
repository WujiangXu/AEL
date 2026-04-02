"""Planner interface for AEL framework."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ael.types import PlanStep


class BasePlanner(ABC):
    """Abstract base class for all AEL planners."""

    name: str = ""
    description: str = ""
    system_prompt_suffix: str = ""  # V2: evolving prompt from procedural memory

    def update_prompt_from_memory(self, memory_store) -> None:
        """Pull procedural memories relevant to this planner, append to prompt."""
        hits = memory_store.retrieve(
            query={"planner": self.name},
            tiers=["procedural", "semantic"],
            top_k=3,
        )
        if hits:
            self.system_prompt_suffix = "\nLearned strategies:\n" + "\n".join(
                f"- {h.content}" for h in hits
            )

    @abstractmethod
    def plan(
        self,
        task: dict,
        available_tools: list[str],
        memory_context: list[dict] | None = None,
    ) -> list[PlanStep]:
        """Generate an execution plan for the given task.

        Args:
            task: Task specification (ticker, date, features, etc.).
            available_tools: Names of tools available for this episode.
            memory_context: Retrieved memory entries relevant to this task.

        Returns:
            Ordered list of PlanStep describing actions to take.
        """
        ...

    @abstractmethod
    def replan(
        self,
        task: dict,
        completed_steps: list[PlanStep],
        remaining_steps: list[PlanStep],
        available_tools: list[str],
        memory_context: list[dict] | None = None,
    ) -> list[PlanStep]:
        """Adjust plan after observing partial execution results.

        Args:
            task: Original task specification.
            completed_steps: Steps already executed (with outcomes).
            remaining_steps: Steps not yet executed.
            available_tools: Currently available tools.
            memory_context: Retrieved memory entries.

        Returns:
            Updated list of remaining PlanStep.
        """
        ...

    def feature_vector(self, task: dict) -> list[float]:
        """Extract feature vector from task for bandit context.

        Default: empty. Subclasses can override for richer features.
        """
        return []
