"""Core data structures for AEL framework."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ToolCall:
    """Record of a single tool invocation."""
    tool_name: str
    inputs: dict
    output: Any = None
    success: bool = True
    latency: float = 0.0
    cost: float = 0.0
    error: str | None = None


@dataclass
class MemoryHit:
    """Record of a memory retrieval result."""
    memory_id: str
    tier: str              # "episodic", "semantic", "procedural"
    content: str
    relevance_score: float
    was_useful: bool | None = None  # filled post-hoc


@dataclass
class PlanStep:
    """A single step in a planner's execution plan."""
    step_id: int
    action: str            # tool call or sub-task
    rationale: str
    outcome: str | None = None
    success: bool | None = None


@dataclass
class TrajectoryRecord:
    """Full execution trace of one task episode."""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    task_type: str = ""                    # e.g., "price_prediction"
    task_features: dict = field(default_factory=dict)  # ticker, sector, volatility
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Configuration used
    planner_id: str = ""
    tool_config: list[str] = field(default_factory=list)
    memory_policy: dict = field(default_factory=dict)

    # Execution trace
    plan_steps: list[PlanStep] = field(default_factory=list)
    tools_invoked: list[ToolCall] = field(default_factory=list)
    memory_retrievals: list[MemoryHit] = field(default_factory=list)

    # Prediction output
    prediction: dict = field(default_factory=dict)  # direction, magnitude, confidence

    # Outcome (filled after realized returns available)
    outcome_score: float = 0.0
    external_signal: dict = field(default_factory=dict)  # realized returns

    # Cost
    cost: float = 0.0
    wall_time: float = 0.0

    # Credit assignment (filled post-hoc by FCC)
    module_credits: dict[str, float] = field(default_factory=dict)


@dataclass
class MemoryEntry:
    """A single entry in the memory store."""
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    tier: str = "episodic"  # "episodic", "semantic", "procedural"
    content: str = ""
    metadata: dict = field(default_factory=dict)  # task_type, ticker, sector, etc.
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    quality_score: float = 0.5
    retrieval_count: int = 0
    usefulness_sum: float = 0.0

    @property
    def avg_usefulness(self) -> float:
        if self.retrieval_count == 0:
            return 0.5
        return self.usefulness_sum / self.retrieval_count


@dataclass
class ToolProfile:
    """Reliability + directional accuracy profile for a registered tool."""
    tool_name: str
    success_count: int = 0
    failure_count: int = 0
    total_latency: float = 0.0
    total_cost: float = 0.0
    call_count: int = 0
    # Per-task-type success rates
    task_type_stats: dict[str, dict] = field(default_factory=dict)
    # V2: Ground-truth directional accuracy
    directional_hits: int = 0
    directional_misses: int = 0
    per_ticker_accuracy: dict[str, dict] = field(default_factory=dict)  # ticker → {hits, misses}

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.5
        return self.success_count / total

    @property
    def directional_accuracy(self) -> float:
        total = self.directional_hits + self.directional_misses
        if total == 0:
            return 0.5  # prior
        return self.directional_hits / total

    @property
    def avg_latency(self) -> float:
        if self.call_count == 0:
            return 0.0
        return self.total_latency / self.call_count

    def update(self, call: ToolCall, task_type: str = ""):
        self.call_count += 1
        self.total_latency += call.latency
        self.total_cost += call.cost
        if call.success:
            self.success_count += 1
        else:
            self.failure_count += 1
        if task_type:
            stats = self.task_type_stats.setdefault(task_type, {"success": 0, "fail": 0})
            stats["success" if call.success else "fail"] += 1

    def update_directional(self, correct: bool, ticker: str = ""):
        """Update directional accuracy after ground truth is known."""
        if correct:
            self.directional_hits += 1
        else:
            self.directional_misses += 1
        if ticker:
            t = self.per_ticker_accuracy.setdefault(ticker, {"hits": 0, "misses": 0})
            t["hits" if correct else "misses"] += 1


@dataclass
class Skill:
    """A learned skill: tool code, prompt template, or strategy playbook."""
    skill_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    skill_type: str = ""        # "tool_code", "prompt_template", "strategy"
    name: str = ""
    description: str = ""
    content: str = ""           # code for tools, template text for prompts, JSON for strategies
    preconditions: dict = field(default_factory=dict)  # when to apply
    source_code: str = ""       # for tool_code type
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    created_day: int = 0
    derived_from: str = ""      # reflection/trajectory that inspired it
    success_count: int = 0
    failure_count: int = 0
    episode_count: int = 0
    active: bool = True

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.5
        return self.success_count / total


@dataclass
class CrossModuleInsight:
    """A discovered cross-module configuration pattern."""
    insight_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    planner_id: str = ""
    tool_set: list[str] = field(default_factory=list)
    memory_policy: dict = field(default_factory=dict)
    task_type_scope: str = ""
    success_rate: float = 0.0
    sample_count: int = 0
    natural_language: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
