"""Base tool interface for AEL framework."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from ael.types import ToolCall


class BaseTool(ABC):
    """Abstract base class for all AEL tools."""

    name: str = ""
    description: str = ""
    cost_per_call: float = 0.0  # estimated $ cost

    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """Execute the tool with given inputs. Returns output or raises."""
        ...

    def __call__(self, **kwargs) -> ToolCall:
        """Execute and wrap result in a ToolCall record."""
        start = time.time()
        try:
            output = self.execute(**kwargs)
            latency = time.time() - start
            return ToolCall(
                tool_name=self.name,
                inputs=kwargs,
                output=output,
                success=True,
                latency=latency,
                cost=self.cost_per_call,
            )
        except Exception as e:
            latency = time.time() - start
            return ToolCall(
                tool_name=self.name,
                inputs=kwargs,
                output=None,
                success=False,
                latency=latency,
                cost=self.cost_per_call,
                error=str(e),
            )
