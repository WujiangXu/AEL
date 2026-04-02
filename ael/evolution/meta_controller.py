"""Meta-controller: hierarchical bandit selecting (planner, tools, memory_policy).

EAEL v2: Per-tool Thompson Sampling replaces fixed tool presets (2^12 possible
subsets). Richer memory policies with context formatting control.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any

from ael.evolution.bandits import ThompsonSampling, LinUCB


@dataclass
class TaskFeatures:
    """Feature vector extracted from a task for bandit context."""
    sector_id: int = 0        # encoded sector (0-10)
    volatility: float = 0.0   # 30-day realized vol
    market_cap_log: float = 0.0  # log(market_cap)
    data_richness: float = 0.0   # fraction of tools that have data
    momentum_30d: float = 0.0    # 30-day return
    has_options: bool = True
    has_analyst: bool = True

    def to_vector(self) -> np.ndarray:
        return np.array([
            self.sector_id / 10.0,  # normalize to [0,1]
            min(self.volatility, 1.0),
            self.market_cap_log / 15.0,  # log(1T) ≈ 13.8
            self.data_richness,
            np.clip(self.momentum_30d, -0.5, 0.5) + 0.5,  # center at 0.5
            float(self.has_options),
            float(self.has_analyst),
        ], dtype=float)


# ---------------------------------------------------------------------------
# Memory policies — EAEL v2: richer strategies with context formatting
# ---------------------------------------------------------------------------

DEFAULT_MEMORY_POLICIES = {
    "none": {
        "enabled_tiers": [], "retrieval_top_k": 0,
        "format": "none",
    },
    "recent_window": {
        "enabled_tiers": ["episodic"], "retrieval_top_k": 5,
        "format": "sliding_window", "keep_first": 1, "keep_last": 3,
    },
    "full_detailed": {
        "enabled_tiers": ["episodic", "semantic", "procedural"],
        "retrieval_top_k": 5, "format": "full",
    },
    "compressed": {
        "enabled_tiers": ["semantic", "procedural"],
        "retrieval_top_k": 6, "format": "ranked_truncate",
        "token_budget": 300,
    },
    "aggressive_learner": {
        "enabled_tiers": ["episodic", "semantic", "procedural"],
        "retrieval_top_k": 8, "format": "ranked_truncate",
        "token_budget": 500, "consolidation_interval": 5,
        "promotion_threshold": 0.55,
    },
}

# Backward compat alias
MEMORY_POLICIES = DEFAULT_MEMORY_POLICIES


class MemoryPolicyRegistry:
    """Mutable registry of memory policies. Starts with defaults, can grow via evolution."""

    def __init__(self, policies: dict[str, dict] | None = None):
        self._policies = dict(policies or DEFAULT_MEMORY_POLICIES)

    def add(self, name: str, policy: dict) -> None:
        self._policies[name] = policy

    def get(self, name: str) -> dict:
        return self._policies.get(name, self._policies["none"])

    @property
    def names(self) -> list[str]:
        return list(self._policies.keys())

    @property
    def policies(self) -> dict[str, dict]:
        return dict(self._policies)

# Legacy tool config presets — kept for backward compatibility
TOOL_CONFIGS = {
    "all": None,  # None = use all registered tools
    "core": ["get_price_history", "get_fundamentals", "compute_technicals", "score_composite_signal"],
    "fundamental": ["get_price_history", "get_fundamentals", "get_analyst_data", "run_dcf_model", "score_risk"],
    "technical": ["get_price_history", "compute_technicals", "compute_quant_risk", "compute_momentum", "get_options_data"],
    "sentiment": ["get_price_history", "get_fundamentals", "get_analyst_data", "compute_technicals", "compute_momentum"],
}


class MetaController:
    """Hierarchical bandit that selects (planner, tools, memory_policy).

    EAEL v2 changes:
    - Per-tool Thompson Sampling replaces fixed tool presets
    - Richer memory policies with formatting control
    - warm_start_from_llm() for cold-start priors
    - Morning strategy overrides
    """

    def __init__(
        self,
        planner_names: list[str],
        tool_config_names: list[str] | None = None,
        memory_policy_names: list[str] | None = None,
        feature_dim: int = 7,
        linucb_alpha: float = 1.0,
        evolve_planner: bool = True,
        evolve_tools: bool = True,
        evolve_memory: bool = True,
        per_tool_selection: bool = False,
        all_tool_names: list[str] | None = None,
    ):
        self.evolve_planner = evolve_planner
        self.evolve_tools = evolve_tools
        self.evolve_memory = evolve_memory
        self.per_tool_selection = per_tool_selection
        self._all_tool_names = all_tool_names or []
        self._episodes_elapsed = 0

        # Level 1: planner selection (contextual)
        self.planner_bandit = LinUCB(
            arm_names=planner_names,
            dim=feature_dim,
            alpha=linucb_alpha,
        )

        # Level 2: tool selection
        if per_tool_selection and all_tool_names:
            # EAEL v2: per-tool Thompson Sampling
            self.tool_bandit = ThompsonSampling(all_tool_names)
        else:
            # Legacy: preset-based selection
            if tool_config_names is None:
                tool_config_names = list(TOOL_CONFIGS.keys())
            self.tool_bandit = ThompsonSampling(tool_config_names)

        # Level 3: memory policy selection
        self.memory_registry = MemoryPolicyRegistry()
        if memory_policy_names is None:
            memory_policy_names = self.memory_registry.names
        self.memory_bandit = ThompsonSampling(memory_policy_names)

        # Morning strategy overrides (set by morning_planning)
        self._morning_overrides: dict[str, Any] = {}

        # Default (fixed) selections when evolution is disabled
        self._fixed_planner = planner_names[0]
        self._fixed_tools = (tool_config_names or list(TOOL_CONFIGS.keys()))[0]
        self._fixed_memory = memory_policy_names[0]

    def set_morning_overrides(self, overrides: dict[str, Any]) -> None:
        """Set morning strategy overrides for today's selections."""
        self._morning_overrides = overrides

    def clear_morning_overrides(self) -> None:
        self._morning_overrides = {}

    def select(
        self,
        task_features: TaskFeatures | np.ndarray,
        ticker: str | None = None,
        frozen: bool = False,
    ) -> dict[str, str]:
        """Select a (planner, tool_config, memory_policy) triple.

        EAEL v2: checks morning_overrides first, then falls back to bandits.
        When frozen=True, uses greedy exploitation (no exploration).
        """
        if isinstance(task_features, TaskFeatures):
            ctx = task_features.to_vector()
        else:
            ctx = np.array(task_features, dtype=float)

        # Check morning overrides for planner
        planner_overrides = self._morning_overrides.get("planner_overrides", {})
        override_planner = planner_overrides.get(ticker) if ticker else None

        # Planner
        if override_planner and override_planner in [
            name for name in self.planner_bandit.arms
        ]:
            planner = override_planner
        elif self.evolve_planner:
            if frozen:
                planner = self.planner_bandit.exploit(ctx)
            else:
                planner = self.planner_bandit.select(ctx)
        else:
            planner = self._fixed_planner

        # Tool config
        if self.evolve_tools:
            if self.per_tool_selection:
                # Per-tool mode returns a special marker; actual selection
                # happens in select_tools()
                tool_config = "__per_tool__"
            else:
                tool_config = self.tool_bandit.select()
        else:
            tool_config = self._fixed_tools

        # Memory policy
        if self.evolve_memory:
            memory_policy = self.memory_bandit.select()
        else:
            memory_policy = self._fixed_memory

        return {
            "planner": planner,
            "tool_config": tool_config,
            "memory_policy": memory_policy,
        }

    def select_tools(
        self,
        tool_registry=None,
    ) -> list[str]:
        """EAEL v2: Per-tool Thompson Sampling — sample each tool, take top-K.

        Each tool's Beta distribution independently converges based on its
        ground-truth directional accuracy.
        """
        if not self.per_tool_selection or tool_registry is None:
            return self._all_tool_names

        tool_scores = {}
        for tool_name in self._all_tool_names:
            profile = tool_registry.get_profile(tool_name)
            alpha = profile.directional_hits + 1
            beta = profile.directional_misses + 1
            tool_scores[tool_name] = np.random.beta(alpha, beta)

        # Dynamic K: start with all tools, shrink as we learn which are bad
        K = max(4, len(self._all_tool_names) - self._episodes_elapsed // 30)
        ranked = sorted(tool_scores, key=tool_scores.get, reverse=True)

        # Respect morning priorities
        priorities = self._morning_overrides.get("tool_priorities", [])
        if priorities:
            # Ensure priority tools are always included
            priority_set = set(priorities) & set(self._all_tool_names)
            selected = list(priority_set)
            for t in ranked:
                if t not in priority_set and len(selected) < K:
                    selected.append(t)
            return selected

        return ranked[:K]

    def update(
        self,
        selection: dict[str, str],
        task_features: TaskFeatures | np.ndarray,
        reward: float,
        module_credits: dict[str, float] | None = None,
        planner_alpha: float | None = None,
        alpha_z: float | None = None,
    ) -> None:
        """Update bandits after observing outcome."""
        if isinstance(task_features, TaskFeatures):
            ctx = task_features.to_vector()
        else:
            ctx = np.array(task_features, dtype=float)

        if module_credits is None:
            module_credits = {"planner": reward, "tools": reward, "memory": reward}

        self._episodes_elapsed += 1

        # Planner update: use z-scored alpha (value-add over equal-weight)
        # if available, otherwise fall back to blended reward + credit
        if self.evolve_planner:
            if alpha_z is not None:
                # Z-scored alpha: properly normalized allocation skill
                planner_reward = np.clip((alpha_z + 1) / 2, 0.05, 0.95)
            elif planner_alpha is not None:
                planner_reward = np.clip((planner_alpha * 200 + 1) / 2, 0, 1)
            else:
                planner_reward = np.clip(
                    reward * 0.5 + module_credits.get("planner", 0) * 0.5, 0, 1
                )
            self.planner_bandit.update(selection["planner"], ctx, planner_reward)

        # Tool config update — allow negative credit to flow through
        if self.evolve_tools:
            tool_credit = module_credits.get("tools", 0)
            tool_reward = np.clip(
                reward * 0.4 + (tool_credit + 1) / 2 * 0.6, 0.05, 0.95
            )
            tool_config = selection.get("tool_config", "")
            if tool_config != "__per_tool__" and tool_config in self.tool_bandit.arms:
                self.tool_bandit.update(tool_config, tool_reward)

        # Memory policy update.
        # When credit is uniform (all modules get 0.333), use a constant
        # reward so the bandit explores equally — variable rewards from
        # z-scored returns are noise under uniform credit.
        # When FCC provides real credit, allow negative credit to flow
        # through so the bandit learns to avoid bad policies.
        if self.evolve_memory:
            mem_credit = module_credits.get("memory", 0.333)
            is_uniform = abs(mem_credit - 0.333) < 0.01
            if is_uniform:
                mem_reward = 0.5  # constant → equal exploration
            else:
                # Map credit [-1, 1] → reward [0, 1], blend with outcome
                mem_reward = np.clip(
                    reward * 0.4 + (mem_credit + 1) / 2 * 0.6, 0.05, 0.95
                )
            self.memory_bandit.update(selection["memory_policy"], mem_reward)

    def update_per_tool(
        self,
        tool_name: str,
        reward: float,
    ) -> None:
        """EAEL v2: Update a single tool's Thompson prior based on directional accuracy."""
        if self.per_tool_selection and tool_name in self.tool_bandit.arms:
            self.tool_bandit.update(tool_name, reward)

    def get_tool_list(self, config_name: str) -> list[str] | None:
        """Resolve tool config name to actual tool list."""
        if config_name == "__per_tool__":
            return None  # Caller should use select_tools() instead
        return TOOL_CONFIGS.get(config_name)

    def get_memory_policy(self, policy_name: str) -> dict:
        """Resolve memory policy name to config dict."""
        return self.memory_registry.get(policy_name)

    def state_dict(self) -> dict:
        return {
            "planner_bandit": self.planner_bandit.state_dict(),
            "tool_bandit": self.tool_bandit.state_dict(),
            "memory_bandit": self.memory_bandit.state_dict(),
            "evolve_planner": self.evolve_planner,
            "evolve_tools": self.evolve_tools,
            "evolve_memory": self.evolve_memory,
            "per_tool_selection": self.per_tool_selection,
            "episodes_elapsed": self._episodes_elapsed,
        }

    def load_state_dict(self, state: dict) -> None:
        self.planner_bandit.load_state_dict(state["planner_bandit"])
        self.tool_bandit.load_state_dict(state["tool_bandit"])
        self.memory_bandit.load_state_dict(state["memory_bandit"])
        self.evolve_planner = state.get("evolve_planner", self.evolve_planner)
        self.evolve_tools = state.get("evolve_tools", self.evolve_tools)
        self.evolve_memory = state.get("evolve_memory", self.evolve_memory)
        self.per_tool_selection = state.get("per_tool_selection", self.per_tool_selection)
        self._episodes_elapsed = state.get("episodes_elapsed", 0)
