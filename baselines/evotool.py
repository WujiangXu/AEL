"""EvoTool baseline (2026).

Implements evolutionary optimization of tool-use policies:
- Maintains a population of tool-selection policies (which tools to use)
- Mutates policies based on blame attribution (which tools caused failures)
- Selects policies via fitness-proportional selection

Key difference from AEL:
- AEL uses Thompson Sampling bandits for tool selection (Bayesian)
- EvoTool uses evolutionary mutation + selection (genetic algorithm style)
- EvoTool has NO memory system, NO planner evolution, NO credit assignment
- Pure tool-policy optimization with blame-aware mutation

Reference: https://arxiv.org/abs/2603.04900
"""

from __future__ import annotations

import logging
import random
from typing import Any, Callable

import numpy as np

from ael.benchmarks.finance import PredictionResult
from ael.llm_predictor import build_prediction_prompt, parse_prediction_response, _fallback_prediction

logger = logging.getLogger(__name__)

# All available tools
ALL_TOOLS = [
    "get_price_history", "get_fundamentals", "get_analyst_data",
    "get_options_data", "get_earnings_data",
    "compute_technicals", "compute_quant_risk", "compute_momentum",
    "compute_correlations", "run_dcf_model", "score_risk",
    "score_composite_signal",
]


class ToolPolicy:
    """A tool-selection policy: which tools to use and in what order."""

    def __init__(self, tools: list[str], fitness: float = 0.5):
        self.tools = tools
        self.fitness = fitness
        self.n_evals = 0
        self.total_score = 0.0

    def update_fitness(self, score: float) -> None:
        self.n_evals += 1
        # Exponential moving average
        alpha = 2 / (self.n_evals + 1)
        self.fitness = alpha * score + (1 - alpha) * self.fitness

    def mutate(self, blame_tools: list[str], rng: random.Random) -> "ToolPolicy":
        """Create a mutated copy: remove blamed tools, possibly add new ones."""
        new_tools = [t for t in self.tools if t not in blame_tools]

        # Add a random tool not currently in the list
        available = [t for t in ALL_TOOLS if t not in new_tools]
        if available and rng.random() < 0.5:
            new_tools.append(rng.choice(available))

        # Random swap mutation
        if len(new_tools) > 2 and rng.random() < 0.3:
            i, j = rng.sample(range(len(new_tools)), 2)
            new_tools[i], new_tools[j] = new_tools[j], new_tools[i]

        # Ensure at least 3 tools
        while len(new_tools) < 3:
            available = [t for t in ALL_TOOLS if t not in new_tools]
            if available:
                new_tools.append(rng.choice(available))
            else:
                break

        return ToolPolicy(tools=new_tools, fitness=self.fitness * 0.9)


class EvoToolPredictor:
    """EvoTool: evolutionary tool-use policy optimization."""

    def __init__(self, model: str, temperature: float = 0.3,
                 population_size: int = 5, seed: int = 42):
        self.model = model
        self.temperature = temperature
        self.rng = random.Random(seed)

        # Initialize population with diverse tool policies
        self.population: list[ToolPolicy] = []
        for i in range(population_size):
            n_tools = self.rng.randint(4, len(ALL_TOOLS))
            tools = self.rng.sample(ALL_TOOLS, n_tools)
            self.population.append(ToolPolicy(tools=tools))

        self._current_policy: ToolPolicy | None = None
        self._last_tool_scores: dict[str, float] = {}

    def predict(
        self,
        task: dict,
        tool_outputs: dict[str, Any],
        memory_hits: list | None = None,
    ) -> PredictionResult:
        """Make a prediction using the selected tool policy's outputs."""
        from ael.llm_utils import llm_call

        ticker = task.get("ticker", "UNKNOWN")
        date = task.get("date", "")
        horizons = task.get("horizons", [1, 2, 4])

        prompt = build_prediction_prompt(task, tool_outputs, memory_context=None)

        try:
            response = llm_call(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=1024,
            )
            text = response.choices[0].message.content
            return parse_prediction_response(text, ticker, date, horizons)
        except Exception as e:
            logger.error(f"EvoTool LLM call failed: {e}")
            return _fallback_prediction(ticker, date, horizons)

    def select_policy(self) -> ToolPolicy:
        """Select a tool policy via fitness-proportional selection."""
        fitnesses = np.array([max(p.fitness, 0.01) for p in self.population])
        probs = fitnesses / fitnesses.sum()
        idx = np.random.choice(len(self.population), p=probs)
        self._current_policy = self.population[idx]
        return self._current_policy

    def get_active_tools(self) -> list[str]:
        """Return the tool list from the currently selected policy."""
        if self._current_policy is None:
            self.select_policy()
        return self._current_policy.tools

    def update(
        self, score: float, prediction: dict, actual: dict,
        tool_outputs: dict[str, Any],
    ) -> None:
        """Update fitness + blame-aware mutation."""
        if self._current_policy is None:
            return

        correct = prediction.get("direction") == actual.get("direction")

        # Update fitness of current policy
        self._current_policy.update_fitness(1.0 if correct else 0.0)

        # Blame attribution: which tools gave misleading signals?
        blame_tools = []
        if not correct:
            blame_tools = self._attribute_blame(tool_outputs, actual)

        # Evolutionary step: replace worst policy with mutant of best
        if len(self.population) >= 2:
            self.population.sort(key=lambda p: p.fitness, reverse=True)
            best = self.population[0]
            worst_idx = len(self.population) - 1

            # Mutate best policy, replacing worst
            if self.population[worst_idx].n_evals >= 3:
                mutant = best.mutate(blame_tools, self.rng)
                self.population[worst_idx] = mutant

    def _attribute_blame(self, tool_outputs: dict, actual: dict) -> list[str]:
        """Simple blame: tools whose signals contradicted the actual direction."""
        actual_dir = actual.get("direction", "flat")
        blamed = []

        tech = tool_outputs.get("compute_technicals", {})
        if tech:
            outlook = tech.get("technical_outlook", "")
            if actual_dir == "down" and "BUY" in outlook:
                blamed.append("compute_technicals")
            elif actual_dir == "up" and "SELL" in outlook:
                blamed.append("compute_technicals")

        mom = tool_outputs.get("compute_momentum", {})
        if mom:
            trend_dir = mom.get("trend", {}).get("direction", "")
            if actual_dir == "down" and trend_dir == "UP":
                blamed.append("compute_momentum")
            elif actual_dir == "up" and trend_dir == "DOWN":
                blamed.append("compute_momentum")

        return blamed


def create_evotool_predictor(
    model: str = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
    temperature: float = 0.3,
    population_size: int = 5,
    seed: int = 42,
) -> tuple[Callable, EvoToolPredictor]:
    """Create an EvoTool predictor."""
    predictor = EvoToolPredictor(model, temperature, population_size, seed)
    return predictor.predict, predictor
