"""ExpeL baseline (Zhao et al., 2024).

Implements experience extraction and reuse: after each episode, extract
"lessons" (success/failure patterns) and store them as flat text entries.
Future episodes retrieve lessons by ticker/sector similarity.

Key difference from AEL:
- Flat lesson store (no tiered episodic/semantic/procedural)
- No bandit-based policy selection
- No tool or planner evolution
- Retrieval by simple keyword matching, not hybrid vector+BM25

Reference: https://arxiv.org/abs/2308.10144
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from ael.benchmarks.finance import PredictionResult
from ael.llm_predictor import build_prediction_prompt, parse_prediction_response, _fallback_prediction

logger = logging.getLogger(__name__)


class Lesson:
    """A single extracted lesson from an episode."""

    def __init__(self, ticker: str, sector: str, content: str, success: bool):
        self.ticker = ticker
        self.sector = sector
        self.content = content
        self.success = success
        self.use_count = 0


class ExperienceStore:
    """Flat store of extracted lessons, retrieved by keyword similarity."""

    def __init__(self, max_lessons: int = 100):
        self.lessons: list[Lesson] = []
        self.max_lessons = max_lessons

    def add(self, lesson: Lesson) -> None:
        self.lessons.append(lesson)
        if len(self.lessons) > self.max_lessons:
            # Evict oldest unused lesson
            self.lessons.sort(key=lambda l: l.use_count)
            self.lessons.pop(0)

    def retrieve(self, ticker: str, sector: str = "", top_k: int = 5) -> list[Lesson]:
        """Retrieve relevant lessons by ticker and sector match."""
        scored = []
        for lesson in self.lessons:
            score = 0
            if lesson.ticker == ticker:
                score += 2.0
            if sector and lesson.sector == sector:
                score += 1.0
            if lesson.success:
                score += 0.5  # Prefer successful lessons
            scored.append((score, lesson))

        scored.sort(key=lambda x: -x[0])
        results = [lesson for _, lesson in scored[:top_k]]
        for lesson in results:
            lesson.use_count += 1
        return results


class ExPeLPredictor:
    """ExpeL-style predictor with experience extraction and reuse."""

    def __init__(self, model: str, temperature: float = 0.3, max_lessons: int = 100):
        self.model = model
        self.temperature = temperature
        self.store = ExperienceStore(max_lessons)

    def predict(
        self,
        task: dict,
        tool_outputs: dict[str, Any],
        memory_hits: list | None = None,
    ) -> PredictionResult:
        """Make a prediction using retrieved lessons as context."""
        from ael.llm_utils import llm_call

        ticker = task.get("ticker", "UNKNOWN")
        date = task.get("date", "")
        horizons = task.get("horizons", [1, 2, 4])
        sector = task.get("sector", "")

        # Retrieve relevant lessons
        lessons = self.store.retrieve(ticker, sector, top_k=5)
        memory_context = None
        if lessons:
            memory_context = [
                {"tier": "lesson", "content": l.content}
                for l in lessons
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
            logger.error(f"ExpeL LLM call failed: {e}")
            return _fallback_prediction(ticker, date, horizons)

    def extract_lesson(
        self,
        ticker: str,
        date: str,
        sector: str,
        prediction: dict,
        actual: dict,
        tool_names: list[str],
    ) -> None:
        """Extract a generalized lesson via LLM from the episode outcome.

        Core ExpeL mechanism: the LLM analyzes the trajectory and extracts
        a reusable rule/pattern, not just a templated success/failure string.
        """
        from ael.llm_utils import llm_call

        pred_dir = prediction.get("direction", "flat")
        actual_dir = actual.get("direction", "flat")
        actual_ret = actual.get("return_pct", 0)
        correct = pred_dir == actual_dir
        tools_str = ", ".join(tool_names[:5]) if tool_names else "none"

        prompt = (
            f"Episode: {ticker}@{date} (sector={sector})\n"
            f"Prediction: {pred_dir} | Actual: {actual_dir} (return={actual_ret:.4f})\n"
            f"Tools used: {tools_str}\n"
            f"Result: {'CORRECT' if correct else 'WRONG'}\n\n"
            f"Extract a single reusable lesson (1-2 sentences) that generalizes "
            f"from this experience. Focus on which signals/tools to trust or "
            f"avoid for similar situations. Start with 'RULE:'"
        )

        try:
            response = llm_call(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=100,
            )
            content = response.choices[0].message.content.strip()
        except Exception:
            # Fallback to template
            if correct:
                content = f"SUCCESS: {ticker} {pred_dir} correct using {tools_str}."
            else:
                content = f"FAILURE: {ticker} predicted {pred_dir}, actual {actual_dir} using {tools_str}."

        lesson = Lesson(ticker=ticker, sector=sector, content=content, success=correct)
        self.store.add(lesson)


def create_expel_predictor(
    model: str = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
    temperature: float = 0.3,
    max_lessons: int = 100,
) -> tuple[Callable, ExPeLPredictor]:
    """Create an ExpeL predictor.

    Returns:
        (predict_fn, predictor_instance) — predict_fn has same signature as
        AEL's predict_fn. predictor_instance needed for calling extract_lesson().
    """
    predictor = ExPeLPredictor(model, temperature, max_lessons)
    return predictor.predict, predictor
