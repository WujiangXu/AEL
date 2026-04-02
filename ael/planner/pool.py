"""Planner pool with six strategies, LinUCB-based selection, and dynamic registration.

EAEL v2: supports register_dynamic() for LLM-generated planners,
validate_and_instantiate() for safety checking, and pool pruning.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
from typing import Any

from ael.planner.base import BasePlanner
from ael.types import PlanStep

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Planner implementations
# ---------------------------------------------------------------------------

class SequentialPlanner(BasePlanner):
    """Fetch all data → score all signals → synthesize.

    Mirrors the existing usstock_analysis pipeline: run every tool in
    sequence, gather all outputs, then produce a single prediction.
    """

    name = "sequential"
    description = "Run all tools in fixed order, then synthesize."

    def plan(self, task, available_tools, memory_context=None):
        steps = []
        # Phase 1: data collection — call every available tool
        for i, tool_name in enumerate(available_tools):
            steps.append(PlanStep(
                step_id=i,
                action=f"call:{tool_name}",
                rationale=f"Collect data from {tool_name}",
            ))
        # Phase 2: synthesize prediction
        steps.append(PlanStep(
            step_id=len(available_tools),
            action="synthesize",
            rationale="Combine all tool outputs into a price prediction",
        ))
        return steps

    def replan(self, task, completed_steps, remaining_steps, available_tools,
               memory_context=None):
        # Sequential planner does not adapt — just skip failed tools
        return [s for s in remaining_steps if s.success is not False]


class DecomposePlanner(BasePlanner):
    """Hierarchical decomposition into sub-tasks.

    Decomposes analysis into four sub-tasks: valuation, momentum,
    sentiment, and risk. Each sub-task uses a subset of tools. Results
    are synthesised into a single prediction.
    """

    name = "decompose"
    description = "Decompose into sub-tasks (valuation, momentum, sentiment, risk)."

    # Map sub-tasks to preferred tools
    SUBTASK_TOOLS = {
        "valuation": ["get_fundamentals", "run_dcf_model"],
        "momentum": ["get_price_history", "compute_technicals", "compute_momentum"],
        "sentiment": ["get_analyst_data", "get_earnings_data"],
        "risk": ["compute_quant_risk", "score_risk", "get_options_data"],
    }

    def plan(self, task, available_tools, memory_context=None):
        steps = []
        step_id = 0
        tool_set = set(available_tools)

        for subtask, preferred in self.SUBTASK_TOOLS.items():
            usable = [t for t in preferred if t in tool_set]
            if not usable:
                continue
            for tool_name in usable:
                steps.append(PlanStep(
                    step_id=step_id,
                    action=f"call:{tool_name}",
                    rationale=f"[{subtask}] Collect from {tool_name}",
                ))
                step_id += 1
            steps.append(PlanStep(
                step_id=step_id,
                action=f"sub_synthesize:{subtask}",
                rationale=f"Synthesize {subtask} sub-result",
            ))
            step_id += 1

        steps.append(PlanStep(
            step_id=step_id,
            action="synthesize",
            rationale="Combine sub-task results into final prediction",
        ))
        return steps

    def replan(self, task, completed_steps, remaining_steps, available_tools,
               memory_context=None):
        # Drop steps whose sub-task tools all failed
        return [s for s in remaining_steps if s.success is not False]


class AdaptivePlanner(BasePlanner):
    """Quick assessment first, deep dive only if ambiguous.

    Phase 1: fetch fundamentals + technicals (cheap).
    Phase 2: if the initial signal is ambiguous (composite near 0),
    run remaining tools for deeper analysis.
    """

    name = "adaptive"
    description = "Quick assessment first; deep analysis only if signal is ambiguous."

    QUICK_TOOLS = ["get_price_history", "get_fundamentals", "compute_technicals"]
    AMBIGUITY_THRESHOLD = 30  # absolute composite score below this → ambiguous

    def plan(self, task, available_tools, memory_context=None):
        tool_set = set(available_tools)
        steps = []
        step_id = 0

        # Phase 1: quick tools
        for tool_name in self.QUICK_TOOLS:
            if tool_name in tool_set:
                steps.append(PlanStep(
                    step_id=step_id,
                    action=f"call:{tool_name}",
                    rationale="Quick assessment phase",
                ))
                step_id += 1

        steps.append(PlanStep(
            step_id=step_id,
            action="quick_synthesize",
            rationale="Evaluate if signal is clear or ambiguous",
        ))
        step_id += 1

        # Phase 2: deep tools (conditional — executed only if ambiguous)
        deep_tools = [t for t in available_tools if t not in self.QUICK_TOOLS]
        for tool_name in deep_tools:
            steps.append(PlanStep(
                step_id=step_id,
                action=f"call_if_ambiguous:{tool_name}",
                rationale="Deep analysis (only if quick signal was ambiguous)",
            ))
            step_id += 1

        steps.append(PlanStep(
            step_id=step_id,
            action="synthesize",
            rationale="Final prediction from all collected data",
        ))
        return steps

    def replan(self, task, completed_steps, remaining_steps, available_tools,
               memory_context=None):
        # Check if quick_synthesize determined signal is clear
        for step in completed_steps:
            if step.action == "quick_synthesize" and step.outcome:
                try:
                    score = abs(float(step.outcome))
                except (ValueError, TypeError):
                    score = 0
                if score >= self.AMBIGUITY_THRESHOLD:
                    # Signal is clear — skip deep tools
                    return [s for s in remaining_steps
                            if not s.action.startswith("call_if_ambiguous")]
        return remaining_steps


# ---------------------------------------------------------------------------
# V2 Planners: COT, Reflexion, HypothesisTest
# ---------------------------------------------------------------------------

class COTReasoningPlanner(BasePlanner):
    """Chain-of-thought: analyze each dimension sequentially, reason through conflicts.

    Step-by-step: trend → valuation → sentiment → risk → synthesize_cot.
    Forces structured reasoning rather than throwing all data at the LLM.
    """

    name = "cot_reasoning"
    description = "Chain-of-thought: step-by-step dimension analysis then synthesize."

    REASONING_CHAIN = [
        ("trend", ["get_price_history", "compute_momentum", "compute_technicals"]),
        ("valuation", ["get_fundamentals", "run_dcf_model"]),
        ("sentiment", ["get_analyst_data", "get_earnings_data"]),
        ("risk", ["compute_quant_risk", "get_options_data"]),
    ]

    def plan(self, task, available_tools, memory_context=None):
        steps = []
        step_id = 0
        tool_set = set(available_tools)

        for dimension, preferred in self.REASONING_CHAIN:
            usable = [t for t in preferred if t in tool_set]
            for tool_name in usable:
                steps.append(PlanStep(
                    step_id=step_id,
                    action=f"call:{tool_name}",
                    rationale=f"[COT step: {dimension}] Gather {tool_name}",
                ))
                step_id += 1
            # Sub-synthesize each dimension
            steps.append(PlanStep(
                step_id=step_id,
                action=f"sub_synthesize:{dimension}",
                rationale=f"Reason about {dimension} evidence",
            ))
            step_id += 1

        # Final COT synthesis
        steps.append(PlanStep(
            step_id=step_id,
            action="synthesize",
            rationale="Chain-of-thought: integrate all dimensions step by step",
        ))
        return steps

    def replan(self, task, completed_steps, remaining_steps, available_tools,
               memory_context=None):
        return [s for s in remaining_steps if s.success is not False]


class ReflexionPlanner(BasePlanner):
    """Predict → self-check confidence → if uncertain, gather more evidence.

    Phase 1: Quick prediction with core tools.
    Phase 2: Self-reflection — is the signal confident enough?
    Phase 3: If not confident, run additional tools and re-predict.
    Inspired by Reflexion (Shinn 2023) and TTSR (2026).
    """

    name = "reflexion"
    description = "Quick predict → self-reflect → gather more evidence if uncertain."

    QUICK_TOOLS = ["get_price_history", "get_fundamentals", "compute_technicals"]
    DEEP_TOOLS = ["compute_momentum", "run_dcf_model", "get_analyst_data",
                  "compute_quant_risk", "get_options_data"]
    CONFIDENCE_THRESHOLD = 0.6

    def plan(self, task, available_tools, memory_context=None):
        steps = []
        step_id = 0
        tool_set = set(available_tools)

        # Phase 1: Quick tools
        for tool_name in self.QUICK_TOOLS:
            if tool_name in tool_set:
                steps.append(PlanStep(
                    step_id=step_id,
                    action=f"call:{tool_name}",
                    rationale="Phase 1: quick assessment",
                ))
                step_id += 1

        # Phase 2: Quick prediction + self-reflection
        steps.append(PlanStep(
            step_id=step_id,
            action="quick_synthesize",
            rationale="Phase 2: initial prediction + confidence check",
        ))
        step_id += 1

        # Phase 3: Deep tools (conditional on low confidence)
        for tool_name in self.DEEP_TOOLS:
            if tool_name in tool_set:
                steps.append(PlanStep(
                    step_id=step_id,
                    action=f"call_if_ambiguous:{tool_name}",
                    rationale="Phase 3: gather more evidence (reflexion)",
                ))
                step_id += 1

        # Final synthesis with all evidence
        steps.append(PlanStep(
            step_id=step_id,
            action="synthesize",
            rationale="Revised prediction incorporating reflexion evidence",
        ))
        return steps

    def replan(self, task, completed_steps, remaining_steps, available_tools,
               memory_context=None):
        for step in completed_steps:
            if step.action == "quick_synthesize" and step.outcome:
                try:
                    score = abs(float(step.outcome))
                except (ValueError, TypeError):
                    score = 0
                if score >= self.CONFIDENCE_THRESHOLD * 100:
                    return [s for s in remaining_steps
                            if not s.action.startswith("call_if_ambiguous")]
        return remaining_steps


class HypothesisTestPlanner(BasePlanner):
    """Form bull/bear hypotheses, gather targeted evidence for each, weigh.

    Phase 1: Core data → form initial hypotheses.
    Phase 2: Bull case tools (valuation, growth indicators).
    Phase 3: Bear case tools (risk, overvaluation indicators).
    Phase 4: Weigh evidence → final prediction.
    Bayesian-inspired: prior (core) + evidence (targeted).
    """

    name = "hypothesis_test"
    description = "Form bull/bear hypotheses, gather targeted evidence, weigh."

    CORE_TOOLS = ["get_price_history", "get_fundamentals"]
    BULL_TOOLS = ["run_dcf_model", "compute_momentum", "get_analyst_data"]
    BEAR_TOOLS = ["compute_quant_risk", "score_risk", "get_options_data"]

    def plan(self, task, available_tools, memory_context=None):
        steps = []
        step_id = 0
        tool_set = set(available_tools)

        # Phase 1: Core data
        for tool_name in self.CORE_TOOLS:
            if tool_name in tool_set:
                steps.append(PlanStep(
                    step_id=step_id,
                    action=f"call:{tool_name}",
                    rationale="Core data for hypothesis formation",
                ))
                step_id += 1

        steps.append(PlanStep(
            step_id=step_id,
            action="sub_synthesize:hypotheses",
            rationale="Form bull and bear hypotheses from core data",
        ))
        step_id += 1

        # Phase 2: Bull case evidence
        for tool_name in self.BULL_TOOLS:
            if tool_name in tool_set:
                steps.append(PlanStep(
                    step_id=step_id,
                    action=f"call:{tool_name}",
                    rationale=f"[Bull case] Evidence from {tool_name}",
                ))
                step_id += 1

        steps.append(PlanStep(
            step_id=step_id,
            action="sub_synthesize:bull_evidence",
            rationale="Evaluate strength of bull case evidence",
        ))
        step_id += 1

        # Phase 3: Bear case evidence
        for tool_name in self.BEAR_TOOLS:
            if tool_name in tool_set:
                steps.append(PlanStep(
                    step_id=step_id,
                    action=f"call:{tool_name}",
                    rationale=f"[Bear case] Evidence from {tool_name}",
                ))
                step_id += 1

        steps.append(PlanStep(
            step_id=step_id,
            action="sub_synthesize:bear_evidence",
            rationale="Evaluate strength of bear case evidence",
        ))
        step_id += 1

        # Phase 4: Weigh hypotheses
        steps.append(PlanStep(
            step_id=step_id,
            action="synthesize",
            rationale="Weigh bull vs bear evidence → final prediction",
        ))
        return steps

    def replan(self, task, completed_steps, remaining_steps, available_tools,
               memory_context=None):
        return [s for s in remaining_steps if s.success is not False]


# ---------------------------------------------------------------------------
# Planner pool with LinUCB selection
# ---------------------------------------------------------------------------

class PlannerPool:
    """Manages a pool of planners and selects among them via LinUCB.

    EAEL v2: supports dynamic registration of LLM-generated planners,
    pool pruning, and per-planner score tracking.
    """

    # Map of planner name → class for selective registration
    ALL_PLANNERS: dict[str, type] = {
        "sequential": SequentialPlanner,
        "decompose": DecomposePlanner,
        "adaptive": AdaptivePlanner,
        "cot_reasoning": COTReasoningPlanner,
        "reflexion": ReflexionPlanner,
        "hypothesis_test": HypothesisTestPlanner,
    }

    # Built-in planner names that should never be pruned
    BUILTIN_PLANNERS = frozenset(ALL_PLANNERS.keys())

    def __init__(self, alpha: float = 1.0, enabled_planners: list[str] | None = None):
        self._planners: dict[str, BasePlanner] = {}
        self._alpha = alpha

        self._dim: int | None = None
        self._A: dict[str, np.ndarray] = {}
        self._b: dict[str, np.ndarray] = {}

        # EAEL v2: per-planner score tracking for pruning
        self._planner_scores: dict[str, list[float]] = defaultdict(list)
        self._planner_day_created: dict[str, int] = {}
        self._dynamic_planner_sources: dict[str, str] = {}  # name → source code

        # Register only enabled planners (default: all)
        if enabled_planners is None:
            enabled_planners = list(self.ALL_PLANNERS.keys())
        for name in enabled_planners:
            cls = self.ALL_PLANNERS.get(name)
            if cls is not None:
                self.register(cls())

    def register(self, planner: BasePlanner) -> None:
        self._planners[planner.name] = planner

    def register_dynamic(
        self,
        planner: BasePlanner,
        source_code: str | None = None,
        day_created: int = 0,
    ) -> None:
        """Register an LLM-generated planner with explore bonus.

        The planner enters the pool on probation — tracked separately for
        potential pruning if it underperforms.
        """
        self._planners[planner.name] = planner
        self._planner_day_created[planner.name] = day_created
        if source_code:
            self._dynamic_planner_sources[planner.name] = source_code

        # Initialize LinUCB arm for the new planner if bandits are active
        if self._dim is not None:
            self._A[planner.name] = np.eye(self._dim)
            self._b[planner.name] = np.zeros(self._dim)

        logger.info(
            f"Registered dynamic planner '{planner.name}' "
            f"(pool size: {len(self._planners)})"
        )

    def get(self, name: str) -> BasePlanner:
        return self._planners[name]

    @property
    def all_names(self) -> list[str]:
        return list(self._planners.keys())

    @property
    def dynamic_names(self) -> list[str]:
        """Names of LLM-generated (non-builtin) planners."""
        return [n for n in self._planners if n not in self.BUILTIN_PLANNERS]

    def record_score(self, planner_name: str, score: float) -> None:
        """Record an episode score for a planner (for pruning decisions)."""
        self._planner_scores[planner_name].append(score)

    def get_planner_stats(self) -> dict[str, dict]:
        """Get per-planner performance statistics."""
        stats = {}
        for name in self._planners:
            scores = self._planner_scores.get(name, [])
            stats[name] = {
                "n_episodes": len(scores),
                "avg_score": float(np.mean(scores)) if scores else 0.0,
                "is_dynamic": name not in self.BUILTIN_PLANNERS,
                "day_created": self._planner_day_created.get(name),
            }
        return stats

    def prune_worst_planner(self, min_episodes: int = 10) -> str | None:
        """Remove the worst-performing dynamic planner from the pool.

        Only prunes LLM-generated planners, never built-in ones.
        Only prunes planners with at least min_episodes of data.

        Returns:
            Name of pruned planner, or None if nothing was pruned.
        """
        worst_name = None
        worst_score = float("inf")

        for name in self.dynamic_names:
            scores = self._planner_scores.get(name, [])
            if len(scores) < min_episodes:
                continue
            avg = float(np.mean(scores))
            if avg < worst_score:
                worst_score = avg
                worst_name = name

        if worst_name is not None:
            del self._planners[worst_name]
            self._A.pop(worst_name, None)
            self._b.pop(worst_name, None)
            logger.info(
                f"Pruned planner '{worst_name}' "
                f"(avg_score={worst_score:.3f}, pool size: {len(self._planners)})"
            )
        return worst_name

    def _init_bandit(self, dim: int) -> None:
        """Lazily initialize LinUCB matrices when feature dim is known."""
        if self._dim is not None:
            return
        self._dim = dim
        for name in self._planners:
            self._A[name] = np.eye(dim)
            self._b[name] = np.zeros(dim)

    def select(
        self,
        task_features: list[float] | np.ndarray,
        candidates: list[str] | None = None,
    ) -> str:
        """Select a planner via LinUCB given task features.

        Args:
            task_features: Context vector for the current task.
            candidates: Subset of planners to consider (default: all).

        Returns:
            Name of the selected planner.
        """
        x = np.array(task_features, dtype=float).flatten()
        self._init_bandit(len(x))

        if candidates is None:
            candidates = self.all_names

        best_name = candidates[0]
        best_ucb = -np.inf

        for name in candidates:
            A_inv = np.linalg.inv(self._A[name])
            theta = A_inv @ self._b[name]
            ucb = float(theta @ x + self._alpha * np.sqrt(x @ A_inv @ x))
            if ucb > best_ucb:
                best_ucb = ucb
                best_name = name

        return best_name

    def update(self, name: str, task_features: list[float] | np.ndarray,
               reward: float) -> None:
        """Update LinUCB parameters after observing reward."""
        x = np.array(task_features, dtype=float).flatten()
        self._init_bandit(len(x))
        self._A[name] += np.outer(x, x)
        self._b[name] += reward * x

    def state_dict(self) -> dict:
        """Serialize bandit state for persistence."""
        return {
            "dim": self._dim,
            "alpha": self._alpha,
            "A": {k: v.tolist() for k, v in self._A.items()},
            "b": {k: v.tolist() for k, v in self._b.items()},
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore bandit state."""
        self._dim = state["dim"]
        self._alpha = state.get("alpha", self._alpha)
        self._A = {k: np.array(v) for k, v in state["A"].items()}
        self._b = {k: np.array(v) for k, v in state["b"].items()}
