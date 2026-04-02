"""LLM-based prediction synthesis for AEL framework.

V3: Unified LLM calling via litellm. Works with any provider:
  - "gpt-4o-mini"                       → OpenAI
  - "claude-sonnet-4-20250514"          → Anthropic
  - "ollama/llama3.2:3b"               → Ollama (local, free)
  - "openrouter/google/gemini-2.0-flash" → OpenRouter
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable

from ael.benchmarks.finance import PredictionResult

logger = logging.getLogger(__name__)


def build_prediction_prompt(
    task: dict,
    tool_outputs: dict[str, Any],
    memory_context: list[dict] | None = None,
) -> str:
    """Build a structured prompt for the LLM to make a price prediction.

    Args:
        task: Task specification (ticker, date, horizons).
        tool_outputs: Dict of tool_name → output from executed tools.
        memory_context: Retrieved memory entries.

    Returns:
        Prompt string for the LLM.
    """
    ticker = task.get("ticker", "UNKNOWN")
    date = task.get("date", "unknown date")
    horizons = task.get("horizons", [1, 7, 14])

    # Only include computed signals (not raw data tools)
    _SIGNAL_PREFIXES = ("compute_", "score_", "run_dcf")

    # Detect frequency from horizon values (bars ≤ 10 = hourly, else daily)
    is_hourly = max(horizons) <= 10
    unit = "bar" if is_hourly else "d"

    sections = []
    sections.append(f"# Stock Prediction: {ticker} @ {date}")
    sections.append(f"Horizons: {', '.join(str(h) + unit for h in horizons)}\n")

    # Compact signal summary — only computed/scored tools
    sections.append("## Signals\n")
    has_signals = False
    for tool_name, output in tool_outputs.items():
        if not any(tool_name.startswith(p) for p in _SIGNAL_PREFIXES):
            continue
        has_signals = True
        if isinstance(output, dict):
            # Flatten to key: value lines (compact, no nested JSON)
            label = tool_name.replace("compute_", "").replace("score_", "").replace("run_", "")
            sections.append(f"**{label}**:")
            for k, v in output.items():
                if isinstance(v, float):
                    sections.append(f"  {k}: {v:.4f}")
                elif isinstance(v, dict):
                    # One level of nesting max
                    flat = ", ".join(f"{kk}={vv:.4f}" if isinstance(vv, float) else f"{kk}={vv}"
                                     for kk, vv in v.items())
                    sections.append(f"  {k}: {flat}")
                else:
                    sections.append(f"  {k}: {v}")
        else:
            sections.append(f"**{tool_name}**: {str(output)[:300]}")
    if not has_signals:
        sections.append("(no computed signals available)")
    sections.append("")

    # Memory context: separate rules (procedural) from context (semantic)
    if memory_context:
        rules = [m for m in memory_context if m.get("tier") == "procedural"]
        context = [m for m in memory_context if m.get("tier") in ("semantic", "episodic")]

        if rules:
            sections.append("## Learned Rules\n")
            for i, mem in enumerate(rules, 1):
                sections.append(f"{i}. {mem.get('content', '')}")
            sections.append("")

        if context:
            sections.append("## Similar Past Situations\n")
            for mem in context:
                sections.append(f"- {mem.get('content', '')}")
            sections.append("")

    # Instructions
    sections.append(
        "## Task\n"
        "Predict price movement for each horizon using ONLY the signals above.\n"
        "Respond in JSON:\n"
        "```json\n"
        "{\n"
    )
    for h in horizons:
        sections.append(
            f'  "{h}": {{"direction": "up/down/flat", "magnitude": 0.03, '
            f'"confidence": 0.7, "reasoning": "..."}}'
            + ("," if h != horizons[-1] else "")
        )
    sections.append("}\n```")

    return "\n".join(sections)


def parse_prediction_response(
    response: str,
    ticker: str,
    date: str,
    horizons: list[int],
) -> PredictionResult:
    """Parse LLM response into a PredictionResult.

    Handles both clean JSON and JSON embedded in markdown code blocks.
    """
    # Try to extract JSON from the response
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try to find bare JSON
        json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            logger.warning(f"Could not parse LLM response for {ticker}@{date}")
            return _fallback_prediction(ticker, date, horizons)

    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError:
        logger.warning(f"Invalid JSON in LLM response for {ticker}@{date}")
        return _fallback_prediction(ticker, date, horizons)

    predictions = {}
    for h in horizons:
        key = str(h)
        if key in parsed:
            entry = parsed[key]
            predictions[h] = {
                "direction": entry.get("direction", "flat"),
                "magnitude": float(entry.get("magnitude", 0.0)),
                "confidence": float(entry.get("confidence", 0.5)),
            }
        elif h in parsed:
            entry = parsed[h]
            predictions[h] = {
                "direction": entry.get("direction", "flat"),
                "magnitude": float(entry.get("magnitude", 0.0)),
                "confidence": float(entry.get("confidence", 0.5)),
            }
        else:
            predictions[h] = {"direction": "flat", "magnitude": 0.0, "confidence": 0.3}

    return PredictionResult(
        ticker=ticker,
        date=date,
        predictions=predictions,
        raw_output=response,
    )


def _fallback_prediction(
    ticker: str, date: str, horizons: list[int]
) -> PredictionResult:
    """Return neutral prediction when LLM parsing fails."""
    return PredictionResult(
        ticker=ticker,
        date=date,
        predictions={
            h: {"direction": "flat", "magnitude": 0.0, "confidence": 0.2}
            for h in horizons
        },
        raw_output="PARSE_FAILED",
    )


# =====================================================================
# Unified LLM Predictor via litellm
# =====================================================================

def create_llm_predictor(
    model: str = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> Callable:
    """Create a unified predict_fn via litellm. Works with any provider:
      - "gpt-4o-mini"                        → OpenAI
      - "claude-sonnet-4-20250514"            → Anthropic
      - "ollama/llama3.2:3b"                 → Ollama (local, free)
      - "openrouter/google/gemini-2.0-flash"  → OpenRouter

    Returns a callable: (task, tool_outputs, memory_hits) → PredictionResult
    """
    try:
        from ael.llm_utils import llm_call
    except ImportError:
        logger.warning("llm_utils not available; falling back to heuristic predictor")
        return create_heuristic_predictor()

    def predict_fn(
        task: dict,
        tool_outputs: dict[str, Any],
        memory_hits: list | None = None,
    ) -> PredictionResult:
        ticker = task.get("ticker", "UNKNOWN")
        date = task.get("date", "")
        horizons = task.get("horizons", [1, 7, 14])

        memory_context = None
        if memory_hits:
            memory_context = [
                {"tier": getattr(h, "tier", "unknown"),
                 "content": getattr(h, "content", str(h))}
                for h in memory_hits
            ]

        prompt = build_prediction_prompt(task, tool_outputs, memory_context)

        try:
            response = llm_call(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = response.choices[0].message.content
            return parse_prediction_response(text, ticker, date, horizons)
        except Exception as e:
            logger.error(f"LLM call failed for {ticker}@{date}: {e}")
            return _fallback_prediction(ticker, date, horizons)

    return predict_fn


def create_heuristic_predictor() -> Callable:
    """Create a predict_fn that uses heuristic rules (no LLM call).

    This is the baseline predictor and also the fallback when no API key
    is available. Uses tool outputs directly with simple scoring rules.
    """
    import numpy as np

    def predict_fn(
        task: dict,
        tool_outputs: dict[str, Any],
        memory_hits: list | None = None,
    ) -> PredictionResult:
        ticker = task.get("ticker", "UNKNOWN")
        date = task.get("date", "")
        horizons = task.get("horizons", [1, 7, 14])

        # Collect signals from tool outputs
        signals = []

        # Technical score
        tech = tool_outputs.get("compute_technicals", {})
        if tech:
            ts = tech.get("technical_score", 2.5)
            signals.append(("technical", (ts - 2.5) / 2.5))

        # Momentum
        mom = tool_outputs.get("compute_momentum", {})
        if mom:
            ret_20 = mom.get("return_20d", 0) or 0
            signals.append(("momentum", np.clip(ret_20 * 5, -1, 1)))

        # DCF
        dcf = tool_outputs.get("run_dcf_model", {})
        if dcf:
            upside = dcf.get("implied_upside", 0) or 0
            signals.append(("dcf", np.clip(upside, -1, 1)))

        # Analyst
        analyst = tool_outputs.get("get_analyst_data", {})
        if analyst:
            rec = analyst.get("recommendation_mean")
            if rec:
                signals.append(("analyst", (3 - rec) / 2))

        # Composite signal
        composite = tool_outputs.get("score_composite_signal", {})
        if composite:
            cs = composite.get("composite_score", 0)
            signals.append(("composite", np.clip(cs / 50, -1, 1)))

        # Aggregate
        if signals:
            avg_signal = np.mean([s[1] for s in signals])
            confidence = min(0.9, 0.3 + abs(avg_signal) * 0.6)
        else:
            avg_signal = 0
            confidence = 0.2

        # Direction
        if abs(avg_signal) < 0.1:
            direction = "flat"
        elif avg_signal > 0:
            direction = "up"
        else:
            direction = "down"

        base_magnitude = abs(avg_signal) * 0.05

        predictions = {}
        for h in horizons:
            scale = np.sqrt(h)
            predictions[h] = {
                "direction": direction,
                "magnitude": float(base_magnitude * scale),
                "confidence": float(confidence * (1 - 0.05 * np.log(max(h, 1)))),
            }

        return PredictionResult(
            ticker=ticker,
            date=date,
            predictions=predictions,
        )

    return predict_fn
