"""FactorMiner-style baseline (2026).

Implements skill + experience memory for financial alpha discovery.
This is the closest published competitor to EAEL — it also uses:
- Skill extraction from successful episodes
- Experience memory for retrieval
- But NO joint evolution of planners/tools/memory policies
- NO bandit-based component selection
- NO credit assignment across modules

Our implementation follows the FactorMiner pattern:
- Skills = successful tool combinations that led to correct predictions
- Experience = past predictions with outcomes, retrieved by similarity
- Skill-guided prediction = bias tool selection toward learned skills

Reference: https://arxiv.org/abs/2602.14670
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from ael.benchmarks.finance import PredictionResult
from ael.llm_predictor import build_prediction_prompt, parse_prediction_response, _fallback_prediction

logger = logging.getLogger(__name__)


class Skill:
    """A learned skill = successful tool combination pattern."""

    def __init__(self, name: str, tool_sequence: list[str], success_rate: float, n_uses: int):
        self.name = name
        self.tool_sequence = tool_sequence
        self.success_rate = success_rate
        self.n_uses = n_uses

    def update(self, success: bool) -> None:
        self.n_uses += 1
        # Exponential moving average
        alpha = 2 / (self.n_uses + 1)
        self.success_rate = alpha * (1.0 if success else 0.0) + (1 - alpha) * self.success_rate


class Experience:
    """A past prediction with its outcome."""

    def __init__(self, ticker: str, date: str, tools_used: list[str],
                 predicted_dir: str, actual_dir: str, correct: bool):
        self.ticker = ticker
        self.date = date
        self.tools_used = tools_used
        self.predicted_dir = predicted_dir
        self.actual_dir = actual_dir
        self.correct = correct


class FactorMinerPredictor:
    """FactorMiner-style predictor with skills and experience memory."""

    def __init__(self, model: str, temperature: float = 0.3,
                 max_skills: int = 15, max_experiences: int = 200):
        self.model = model
        self.temperature = temperature
        self.skills: dict[str, Skill] = {}
        self.experiences: list[Experience] = []
        self.max_skills = max_skills
        self.max_experiences = max_experiences

    def predict(
        self,
        task: dict,
        tool_outputs: dict[str, Any],
        memory_hits: list | None = None,
    ) -> PredictionResult:
        """Make a prediction using skill-guided context + experience memory."""
        from ael.llm_utils import llm_call

        ticker = task.get("ticker", "UNKNOWN")
        date = task.get("date", "")
        horizons = task.get("horizons", [1, 2, 4])

        # Build memory context from experiences
        relevant_exp = self._retrieve_experiences(ticker, top_k=5)
        memory_context = None
        if relevant_exp or self.skills:
            memory_context = []
            # Add skill summaries
            for skill in sorted(self.skills.values(), key=lambda s: -s.success_rate)[:3]:
                memory_context.append({
                    "tier": "skill",
                    "content": f"Skill '{skill.name}': tools={skill.tool_sequence}, "
                               f"success_rate={skill.success_rate:.2f} (n={skill.n_uses})",
                })
            # Add relevant experiences
            for exp in relevant_exp:
                result = "correct" if exp.correct else "wrong"
                memory_context.append({
                    "tier": "experience",
                    "content": f"{exp.ticker}@{exp.date}: predicted {exp.predicted_dir}, "
                               f"actual {exp.actual_dir} ({result}). "
                               f"Tools: {', '.join(exp.tools_used[:5])}",
                })

        prompt = build_prediction_prompt(task, tool_outputs, memory_context)

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
            logger.error(f"FactorMiner LLM call failed: {e}")
            return _fallback_prediction(ticker, date, horizons)

    def update(
        self,
        ticker: str,
        date: str,
        tools_used: list[str],
        prediction: dict,
        actual: dict,
    ) -> None:
        """Update skills and experiences after seeing the outcome."""
        pred_dir = prediction.get("direction", "flat")
        actual_dir = actual.get("direction", "flat")
        correct = pred_dir == actual_dir

        # Store experience
        exp = Experience(ticker, date, tools_used, pred_dir, actual_dir, correct)
        self.experiences.append(exp)
        if len(self.experiences) > self.max_experiences:
            self.experiences.pop(0)

        # Extract/update skills from successful tool combinations
        if correct and len(tools_used) >= 2:
            skill_name = "_".join(sorted(tools_used[:4]))
            if skill_name in self.skills:
                self.skills[skill_name].update(True)
            else:
                if len(self.skills) < self.max_skills:
                    self.skills[skill_name] = Skill(
                        name=skill_name,
                        tool_sequence=tools_used[:4],
                        success_rate=1.0,
                        n_uses=1,
                    )
        elif not correct:
            # Penalize any matching skills
            skill_name = "_".join(sorted(tools_used[:4]))
            if skill_name in self.skills:
                self.skills[skill_name].update(False)

    def _retrieve_experiences(self, ticker: str, top_k: int = 5) -> list[Experience]:
        """Retrieve relevant past experiences by ticker match."""
        # Prioritize same-ticker experiences, then recent ones
        same_ticker = [e for e in reversed(self.experiences) if e.ticker == ticker]
        others = [e for e in reversed(self.experiences) if e.ticker != ticker]
        results = same_ticker[:top_k]
        remaining = top_k - len(results)
        if remaining > 0:
            results.extend(others[:remaining])
        return results


def create_factorminer_predictor(
    model: str = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
    temperature: float = 0.3,
) -> tuple[Callable, FactorMinerPredictor]:
    """Create a FactorMiner-style predictor.

    Returns:
        (predict_fn, predictor_instance)
    """
    predictor = FactorMinerPredictor(model, temperature)
    return predictor.predict, predictor
