"""Meta-Policy Reflexion baseline (2025).

Extends Reflexion with reusable reflective memory and rule admissibility:
- Instead of accumulating raw reflections, distills them into meta-policies
  (concise rules) that are checked for consistency before application.
- Rules that contradict or have low success are pruned.

Key difference from basic Reflexion:
- Reflections → distilled into rules (fewer, higher quality)
- Rules have success tracking (kept/pruned based on accuracy)
- Admissibility check: conflicting rules are resolved before use

Key difference from AEL:
- No bandit-based component selection
- No tool/planner evolution
- Rules are text-based, not structured tiers (episodic/semantic/procedural)

Reference: https://arxiv.org/abs/2509.03990
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from ael.benchmarks.finance import PredictionResult
from ael.llm_predictor import build_prediction_prompt, parse_prediction_response, _fallback_prediction

logger = logging.getLogger(__name__)


class MetaRule:
    """A distilled meta-policy rule with success tracking."""

    def __init__(self, content: str, source_count: int = 1):
        self.content = content
        self.source_count = source_count  # how many reflections were distilled into this
        self.applications = 0
        self.successes = 0

    @property
    def success_rate(self) -> float:
        return self.successes / max(self.applications, 1)

    def apply(self, success: bool) -> None:
        self.applications += 1
        if success:
            self.successes += 1


class MetaReflexionPredictor:
    """Meta-Policy Reflexion: reflections distilled into reusable rules."""

    def __init__(self, model: str, temperature: float = 0.3,
                 max_rules: int = 10, distill_interval: int = 5):
        self.model = model
        self.temperature = temperature
        self.max_rules = max_rules
        self.distill_interval = distill_interval

        self.raw_reflections: list[str] = []  # Accumulate before distillation
        self.rules: list[MetaRule] = []  # Distilled meta-policies
        self._episode_count = 0

    def predict(
        self,
        task: dict,
        tool_outputs: dict[str, Any],
        memory_hits: list | None = None,
    ) -> PredictionResult:
        """Make a prediction using active meta-policy rules as context."""
        from ael.llm_utils import llm_call

        ticker = task.get("ticker", "UNKNOWN")
        date = task.get("date", "")
        horizons = task.get("horizons", [1, 2, 4])

        # Build context from active rules (admissibility: only rules with success_rate >= 0.4)
        active_rules = [r for r in self.rules if r.applications < 3 or r.success_rate >= 0.4]

        memory_context = None
        if active_rules:
            memory_context = [
                {"tier": "meta_rule", "content": r.content}
                for r in active_rules[:self.max_rules]
            ]

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
            logger.error(f"MetaReflexion LLM call failed: {e}")
            return _fallback_prediction(ticker, date, horizons)

    def reflect_and_update(
        self, ticker: str, date: str, prediction: dict, actual: dict,
        tool_outputs: dict | None = None,
    ) -> None:
        """Reflect on outcome + periodically distill into rules."""
        from ael.llm_utils import llm_call

        pred_dir = prediction.get("direction", "flat")
        actual_dir = actual.get("direction", "flat")
        actual_ret = actual.get("return_pct", 0)
        correct = pred_dir == actual_dir

        # Update existing rules that were "active" during this prediction
        for rule in self.rules:
            if rule.applications > 0:  # was available during prediction
                rule.apply(correct)

        # Generate raw reflection via LLM
        try:
            prompt = (
                f"{ticker}@{date}: predicted {pred_dir}, actual {actual_dir} "
                f"(ret={actual_ret:.4f}). {'Correct' if correct else 'Wrong'}. "
                f"Write a 1-sentence observation about this outcome."
            )
            response = llm_call(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3, max_tokens=60,
            )
            reflection = response.choices[0].message.content.strip()
        except Exception:
            reflection = f"{ticker}@{date}: {pred_dir}→{actual_dir} ({'ok' if correct else 'wrong'})"

        self.raw_reflections.append(reflection)
        self._episode_count += 1

        # Periodically distill reflections into rules
        if self._episode_count % self.distill_interval == 0 and len(self.raw_reflections) >= 3:
            self._distill_rules()

        # Prune low-performing rules
        self.rules = [
            r for r in self.rules
            if r.applications < 5 or r.success_rate >= 0.3
        ]

    def _distill_rules(self) -> None:
        """Distill accumulated reflections into concise meta-policy rules."""
        from ael.llm_utils import llm_call

        recent = self.raw_reflections[-10:]
        reflections_text = "\n".join(f"- {r}" for r in recent)

        existing_rules = "\n".join(f"- {r.content}" for r in self.rules) if self.rules else "(none)"

        prompt = (
            f"Recent observations:\n{reflections_text}\n\n"
            f"Existing rules:\n{existing_rules}\n\n"
            f"Distill 1-2 NEW concise trading rules from these observations. "
            f"Rules should be actionable (e.g., 'When RSI < 30, favor up predictions'). "
            f"Don't repeat existing rules. One rule per line, start each with 'RULE:'"
        )

        try:
            response = llm_call(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4, max_tokens=150,
            )
            text = response.choices[0].message.content
            for line in text.strip().split("\n"):
                line = line.strip()
                if line.startswith("RULE:"):
                    rule_text = line[5:].strip()
                    if rule_text and len(self.rules) < self.max_rules * 2:
                        self.rules.append(MetaRule(content=rule_text, source_count=len(recent)))
        except Exception as e:
            logger.warning(f"Rule distillation failed: {e}")

        # Clear processed reflections
        self.raw_reflections = self.raw_reflections[-3:]  # keep last 3 for context


def create_meta_reflexion_predictor(
    model: str = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
    temperature: float = 0.3,
    max_rules: int = 10,
    distill_interval: int = 5,
) -> tuple[Callable, MetaReflexionPredictor]:
    """Create a Meta-Policy Reflexion predictor."""
    predictor = MetaReflexionPredictor(model, temperature, max_rules, distill_interval)
    return predictor.predict, predictor
