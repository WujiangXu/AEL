"""Tool registry with reliability tracking and Thompson Sampling selection."""

from __future__ import annotations

import numpy as np

from ael.tools.base import BaseTool
from ael.types import ToolCall, ToolProfile


class ToolRegistry:
    """Manages available tools, tracks reliability, and selects tools via Thompson Sampling."""

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}
        self._profiles: dict[str, ToolProfile] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool
        if tool.name not in self._profiles:
            self._profiles[tool.name] = ToolProfile(tool_name=tool.name)

    def register_dynamic(self, tool: BaseTool) -> None:
        """Register a dynamically generated tool (from skills system)."""
        self.register(tool)

    def get(self, name: str) -> BaseTool:
        return self._tools[name]

    @property
    def all_names(self) -> list[str]:
        return list(self._tools.keys())

    def call(self, name: str, task_type: str = "", **kwargs) -> ToolCall:
        """Call a tool and update its reliability profile."""
        tool = self._tools[name]
        result = tool(**kwargs)
        self._profiles[name].update(result, task_type)
        return result

    def get_profile(self, name: str) -> ToolProfile:
        return self._profiles[name]

    def select_tools(
        self,
        candidates: list[str] | None = None,
        n: int | None = None,
        task_type: str = "",
    ) -> list[str]:
        """Select tools via Thompson Sampling based on reliability profiles.

        Args:
            candidates: Pool of tool names to select from (default: all).
            n: Number of tools to select (default: all candidates).
            task_type: Optional task type for task-specific selection.

        Returns:
            Ordered list of selected tool names (highest sampled value first).
        """
        if candidates is None:
            candidates = self.all_names
        if n is None:
            n = len(candidates)

        scores = {}
        for name in candidates:
            profile = self._profiles[name]
            # Thompson Sampling: sample from Beta(success+1, failure+1)
            alpha = profile.success_count + 1
            beta = profile.failure_count + 1
            scores[name] = np.random.beta(alpha, beta)

        ranked = sorted(scores, key=scores.get, reverse=True)
        return ranked[:n]

    def get_reliability_summary(self) -> dict[str, dict]:
        """Return summary stats for all tools."""
        return {
            name: {
                "success_rate": p.success_rate,
                "call_count": p.call_count,
                "avg_latency": p.avg_latency,
            }
            for name, p in self._profiles.items()
        }

    def state_dict(self) -> dict:
        """Serialize profiles for persistence (V2: includes directional accuracy)."""
        return {
            name: {
                "success_count": p.success_count,
                "failure_count": p.failure_count,
                "total_latency": p.total_latency,
                "total_cost": p.total_cost,
                "call_count": p.call_count,
                "task_type_stats": p.task_type_stats,
                "directional_hits": p.directional_hits,
                "directional_misses": p.directional_misses,
                "per_ticker_accuracy": p.per_ticker_accuracy,
            }
            for name, p in self._profiles.items()
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore profiles from serialized state (V2: includes directional accuracy)."""
        for name, data in state.items():
            if name in self._profiles:
                p = self._profiles[name]
                p.success_count = data["success_count"]
                p.failure_count = data["failure_count"]
                p.total_latency = data["total_latency"]
                p.total_cost = data["total_cost"]
                p.call_count = data["call_count"]
                p.task_type_stats = data.get("task_type_stats", {})
                p.directional_hits = data.get("directional_hits", 0)
                p.directional_misses = data.get("directional_misses", 0)
                p.per_ticker_accuracy = data.get("per_ticker_accuracy", {})
