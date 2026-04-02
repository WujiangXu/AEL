"""Reflexion baseline (Shinn et al., 2023).

Implements verbal self-reflection: after each episode, the agent appends a
self-critique to a growing context window. No structured memory — just an
accumulating list of reflection strings prepended to future prompts.

Key difference from AEL:
- No tiered memory (episodic/semantic/procedural)
- No bandit-based policy selection
- No tool evolution
- Just growing context with verbal reflections

Reference: https://arxiv.org/abs/2303.11366
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

from ael.benchmarks.finance import PredictionResult
from ael.llm_predictor import build_prediction_prompt, parse_prediction_response, _fallback_prediction

logger = logging.getLogger(__name__)


class ReflexionPredictor:
    """Reflexion-style predictor that accumulates verbal self-reflections.

    After each prediction is evaluated, a reflection is generated and
    appended to a growing context that informs future predictions.
    """

    def __init__(self, model: str, max_reflections: int = 20, temperature: float = 0.3):
        self.model = model
        self.temperature = temperature
        self.max_reflections = max_reflections
        self.reflections: list[str] = []  # Growing context

    def predict(
        self,
        task: dict,
        tool_outputs: dict[str, Any],
        memory_hits: list | None = None,
    ) -> PredictionResult:
        """Make a prediction with accumulated reflections as context."""
        from ael.llm_utils import llm_call

        ticker = task.get("ticker", "UNKNOWN")
        date = task.get("date", "")
        horizons = task.get("horizons", [1, 2, 4])

        # Build base prompt (same as AEL)
        prompt = build_prediction_prompt(task, tool_outputs, memory_context=None)

        # Prepend accumulated reflections
        if self.reflections:
            recent = self.reflections[-self.max_reflections:]
            reflection_block = "\n## Past Reflections\n" + "\n".join(
                f"- {r}" for r in recent
            ) + "\n\n"
            prompt = reflection_block + prompt

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
            logger.error(f"Reflexion LLM call failed: {e}")
            return _fallback_prediction(ticker, date, horizons)

    def reflect(
        self, ticker: str, date: str, prediction: dict, actual: dict,
        tool_outputs: dict | None = None,
    ) -> None:
        """Generate a verbal self-reflection via LLM after seeing the outcome.

        This is the core Reflexion mechanism — the LLM itself generates a
        free-form critique, which is richer than a templated string.
        """
        from ael.llm_utils import llm_call

        pred_dir = prediction.get("direction", "flat")
        actual_dir = actual.get("direction", "flat")
        actual_ret = actual.get("return_pct", 0)
        correct = pred_dir == actual_dir

        # Build context for LLM self-critique
        signals_summary = ""
        if tool_outputs:
            for tool, output in list(tool_outputs.items())[:4]:
                if isinstance(output, dict):
                    key_vals = ", ".join(f"{k}={v}" for k, v in list(output.items())[:3]
                                         if not isinstance(v, (list, dict)))
                    signals_summary += f"  {tool}: {key_vals}\n"

        prompt = (
            f"You predicted {pred_dir} for {ticker}@{date}. "
            f"Actual: {actual_dir} (return={actual_ret:.4f}). "
            f"{'Correct!' if correct else 'Wrong.'}\n"
            f"Signals used:\n{signals_summary}\n"
            f"Write a 1-sentence reflection on what to do differently next time. "
            f"Be specific about which signals to trust or ignore."
        )

        try:
            response = llm_call(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=100,
            )
            reflection = response.choices[0].message.content.strip()
        except Exception:
            # Fallback to template if LLM fails
            if correct:
                reflection = f"{ticker}@{date}: Correct ({pred_dir}, ret={actual_ret:.4f})."
            else:
                reflection = (
                    f"{ticker}@{date}: Wrong — predicted {pred_dir}, "
                    f"actual {actual_dir} ({actual_ret:.4f})."
                )

        self.reflections.append(reflection)


def create_reflexion_predictor(
    model: str = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
    temperature: float = 0.3,
    max_reflections: int = 20,
) -> tuple[Callable, ReflexionPredictor]:
    """Create a Reflexion predictor.

    Returns:
        (predict_fn, predictor_instance) — predict_fn has same signature as
        AEL's predict_fn. predictor_instance needed for calling reflect().
    """
    predictor = ReflexionPredictor(model, max_reflections, temperature)
    return predictor.predict, predictor
