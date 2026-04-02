"""EAEL reflection module: cold-start, daily reflection, morning planning, planner evolution.

Implements the human-like day-level loop where an LLM agent reads about its
job, forms strategy, executes, reflects on outcomes + market context, and
continuously redesigns its own workflow including generating new planner code.
"""

from __future__ import annotations

import ast
import inspect
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ael.llm_utils import llm_call
from ael.planner.base import BasePlanner
from ael.types import PlanStep

logger = logging.getLogger(__name__)

# Data cache directory
DATA_DIR = Path(
    os.environ.get(
        "AEL_DATA_DIR",
        Path(__file__).parent.parent.parent / "data",
    )
)

# Sector mapping for side info
SECTOR_MAP = {
    "Technology": ["AAPL", "MSFT", "NVDA", "AMD", "CRM", "PANW"],
    "Communication Services": ["GOOGL", "META", "NFLX"],
    "Consumer Cyclical": ["AMZN", "TSLA", "ABNB", "DKNG", "ROKU"],
    "Healthcare": ["JNJ", "UNH"],
    "Financials": ["JPM", "GS", "SOFI", "UPST", "BILL"],
    "Energy": ["XOM"],
    "Consumer Defensive": ["PG"],
    "Industrials": ["CAT", "PATH"],
    "Utilities": ["NEE"],
}

TICKER_TO_SECTOR = {}
for sector, tickers in SECTOR_MAP.items():
    for t in tickers:
        TICKER_TO_SECTOR[t] = sector


# ---------------------------------------------------------------------------
# Side Information
# ---------------------------------------------------------------------------

def compute_daily_side_info(
    day: str,
    tickers: list[str],
    data_dir: Path | None = None,
) -> dict[str, Any]:
    """Compute market context for a given day from cached price data.

    This information is NOT given to the predictor (no leaking). It is given
    to the reflection LLM to help understand WHY tools/planners succeeded or
    failed.

    Returns:
        Dict with market_return, volatility_regime, sector_performance,
        cross_correlation, dispersion, etc.
    """
    if data_dir is None:
        data_dir = DATA_DIR

    side_info: dict[str, Any] = {"date": day}
    day_ts = pd.Timestamp(day)

    # SPY as market proxy
    spy_path = data_dir / "prices" / "SPY.csv"
    if spy_path.exists():
        spy_df = pd.read_csv(spy_path, index_col=0, parse_dates=True)
        spy_df = spy_df[spy_df.index <= day_ts]
        if len(spy_df) >= 2:
            spy_close = spy_df["Close"]
            side_info["market_return"] = float(
                spy_close.iloc[-1] / spy_close.iloc[-2] - 1
            )
            if len(spy_close) >= 5:
                side_info["market_trend_5d"] = float(
                    spy_close.iloc[-1] / spy_close.iloc[-5] - 1
                )
            if len(spy_close) >= 30:
                spy_vol = float(
                    spy_close.pct_change().dropna().tail(30).std() * np.sqrt(252)
                )
                side_info["volatility_regime"] = (
                    "high" if spy_vol > 0.25 else "normal"
                )
                side_info["spy_30d_vol"] = spy_vol

    # Per-ticker returns today → sector performance + dispersion
    ticker_returns = {}
    for ticker in tickers:
        price_path = data_dir / "prices" / f"{ticker}.csv"
        if not price_path.exists():
            continue
        df = pd.read_csv(price_path, index_col=0, parse_dates=True)
        df = df[df.index <= day_ts]
        if len(df) >= 2:
            ret = float(df["Close"].iloc[-1] / df["Close"].iloc[-2] - 1)
            ticker_returns[ticker] = ret

    if ticker_returns:
        # Sector performance
        sector_rets: dict[str, list[float]] = defaultdict(list)
        for t, r in ticker_returns.items():
            sector = TICKER_TO_SECTOR.get(t, "Other")
            sector_rets[sector].append(r)
        side_info["sector_performance"] = {
            s: float(np.mean(rs)) for s, rs in sector_rets.items()
        }

        # Dispersion (std of returns across tickers)
        rets = list(ticker_returns.values())
        side_info["dispersion"] = float(np.std(rets))

        # Cross-correlation (average pairwise from recent data)
        side_info["ticker_returns_today"] = ticker_returns

    # Average pairwise correlation (30d rolling)
    returns_df = _load_ticker_returns(tickers, day_ts, data_dir, window=30)
    if returns_df is not None and returns_df.shape[1] >= 2:
        corr = returns_df.corr()
        # Average off-diagonal
        mask = np.ones(corr.shape, dtype=bool)
        np.fill_diagonal(mask, False)
        side_info["cross_correlation"] = float(corr.values[mask].mean())

    return side_info


def _load_ticker_returns(
    tickers: list[str],
    up_to: pd.Timestamp,
    data_dir: Path,
    window: int = 30,
) -> pd.DataFrame | None:
    """Load aligned returns for multiple tickers up to a given date."""
    all_closes = {}
    for ticker in tickers:
        path = data_dir / "prices" / f"{ticker}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df = df[df.index <= up_to]
        if len(df) >= window:
            all_closes[ticker] = df["Close"].tail(window + 1)
    if len(all_closes) < 2:
        return None
    combined = pd.DataFrame(all_closes).dropna()
    return combined.pct_change().dropna()


# ---------------------------------------------------------------------------
# Cold Start Agent
# ---------------------------------------------------------------------------

def cold_start(
    tool_registry,
    planner_pool,
    tickers_metadata: dict[str, dict],
    config,
) -> dict[str, Any]:
    """LLM analyzes components and generates warm-start priors.

    Called once at episode 0 before any execution. Reads tool descriptions,
    planner descriptions, and ticker metadata to generate informed initial
    preferences instead of uniform priors.

    Returns:
        Dict with 'tool_priors' and 'planner_priors' that can be used to
        warm-start Thompson/LinUCB bandits.
    """
    llm_model = config.cold_start.llm_model or config.llm_model

    # Build context
    tool_descriptions = {}
    for name in tool_registry.all_names:
        tool = tool_registry.get(name)
        tool_descriptions[name] = getattr(tool, "description", name)

    planner_descriptions = {}
    for name in planner_pool.all_names:
        planner = planner_pool.get(name)
        planner_descriptions[name] = planner.description

    prompt = f"""You are a quantitative analyst starting a new job at an investment firm.

Your task: analyze the tools and strategies available to you and form initial preferences
for which tools and strategies to use for different types of stocks.

## Available Analysis Tools
{json.dumps(tool_descriptions, indent=2)}

## Available Planning Strategies
{json.dumps(planner_descriptions, indent=2)}

## Stock Universe
{json.dumps(tickers_metadata, indent=2)}

## Instructions
For each tool, rate its expected usefulness on a scale of 1-5 (1=rarely useful, 5=essential).
Consider that some tools may be more useful for certain stock types.

For each planner strategy, rate its expected effectiveness (1-5).

Return your analysis as JSON with exactly this structure:
{{
  "tool_priors": {{
    "<tool_name>": {{"rating": <1-5>, "reason": "<brief>"}},
    ...
  }},
  "planner_priors": {{
    "<planner_name>": {{"rating": <1-5>, "reason": "<brief>"}},
    ...
  }},
  "sector_tool_preferences": {{
    "Technology": ["<preferred_tool_1>", "<preferred_tool_2>", ...],
    "Communication Services": [...],
    "Consumer Cyclical": [...]
  }},
  "high_volatility_strategy": "<planner_name>",
  "low_volatility_strategy": "<planner_name>"
}}"""

    try:
        response = llm_call(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=config.llm_temperature,
            max_tokens=2048,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        logger.info(f"Cold-start: LLM generated priors for {len(result.get('tool_priors', {}))} tools, "
                     f"{len(result.get('planner_priors', {}))} planners")
        return result
    except Exception as e:
        logger.warning(f"Cold-start LLM call failed: {e}, using uniform priors")
        return {}


def apply_cold_start_priors(
    priors: dict[str, Any],
    meta_controller,
    tool_registry,
) -> None:
    """Apply LLM-generated cold-start priors to bandit parameters.

    For tools: set Thompson Beta(alpha, beta) based on LLM ratings.
    For planners: add bias terms to LinUCB.
    """
    # Tool priors → Thompson alpha/beta
    tool_priors = priors.get("tool_priors", {})
    for tool_name, info in tool_priors.items():
        rating = info.get("rating", 3)
        # rating 1-5 → alpha from 1 to 4, beta inverse
        alpha = max(1.0, rating * 0.8)
        beta = max(1.0, (6 - rating) * 0.8)
        profile = tool_registry._profiles.get(tool_name)
        if profile:
            profile.directional_hits = int(alpha - 1)
            profile.directional_misses = int(beta - 1)

    # Planner priors → LinUCB bias
    planner_priors = priors.get("planner_priors", {})
    for planner_name, info in planner_priors.items():
        rating = info.get("rating", 3)
        # Add synthetic observations to bias the planner bandit
        if hasattr(meta_controller, "planner_bandit"):
            bandit = meta_controller.planner_bandit
            if planner_name in bandit.arms:
                # Add a small bias via synthetic reward
                bias_reward = rating / 5.0
                ctx = np.ones(bandit.dim) * 0.5  # neutral context
                for _ in range(max(1, rating - 2)):
                    bandit.update(planner_name, ctx, bias_reward)


# ---------------------------------------------------------------------------
# Daily Reflection Agent
# ---------------------------------------------------------------------------

def daily_reflection(
    day_results: list[dict],
    side_info: dict[str, Any],
    memory_store,
    config,
    recent_reflections: list[dict] | None = None,
    evolve_planners: bool = False,
) -> dict[str, Any]:
    """Evening reflection: analyze today's results and generate insights.

    Called after all tickers for one day are done.

    Args:
        day_results: List of per-ticker episode summaries for today.
        side_info: Market context (from compute_daily_side_info).
        memory_store: Memory backend for retrieving historical patterns.
        config: ExperimentConfig.
        recent_reflections: Last few days' reflections for context.

    Returns:
        Structured reflection dict with keys: key_insight, tool_updates,
        planner_critique, tomorrow_strategy, structural_problem.
    """
    llm_model = config.reflection.llm_model or config.llm_model

    # Format day results
    results_text = _format_day_results(day_results)
    tool_accuracy_text = _format_tool_accuracy(day_results)
    planner_scores_text = _format_planner_scores(day_results)

    # Retrieve historical patterns from memory
    memory_context = ""
    try:
        hits = memory_store.retrieve(
            query={"task_type": "daily_reflection"},
            tiers=["semantic", "procedural"],
            top_k=3,
        )
        if hits:
            memory_context = "\n".join(f"- {h.content}" for h in hits)
    except Exception:
        pass

    # Recent reflections context
    recent_text = ""
    if recent_reflections:
        for r in recent_reflections[-3:]:
            recent_text += f"\n  Day {r.get('date', '?')}: {r.get('key_insight', 'N/A')}"

    # Build prompt conditionally based on whether planner evolution is active
    planner_section = ""
    planner_json = ""
    planner_instruction = ""
    if evolve_planners:
        planner_section = f"\n## Per-Planner Performance Today\n{planner_scores_text}\n"
        planner_json = """,
  "planner_critique": {{
    "failing_planner": "<name of worst planner today, or null if all adequate>",
    "successful_planner": "<name of best planner today, or null>",
    "reason": "<1 sentence why the failing planner struggled>"
  }},
  "structural_problem": {{
    "detected": <true if a planner or component is fundamentally mismatched with the current regime>,
    "description": "<1 sentence describing what structural issue was found, or empty string>"
  }}"""
    else:
        planner_instruction = "\nIMPORTANT: Do NOT recommend specific tools or planners. Focus only on understanding market patterns."

    prompt = f"""You are a quantitative analyst reviewing today's trading predictions.

## Today's Market Context
{json.dumps(side_info, indent=2, default=str)}

## Results Across {len(day_results)} Episodes
{results_text}

## Per-Tool Accuracy Today
{tool_accuracy_text}
{planner_section}
## Historical Patterns (from memory)
{memory_context or "No historical patterns yet."}

## Recent Reflections
{recent_text or "First day of analysis."}

## Generate a CAUSAL insight as JSON:
{{
  "key_insight": "<What market conditions caused today's outcomes? 2-3 sentences focusing on WHY, not just WHAT. Be specific about which signals were reliable vs misleading.>",
  "market_regime": "<bullish/bearish/mixed/sector_rotation/range_bound>",
  "pattern_confidence": <0.0-1.0>,
  "tomorrow_outlook": "<1 sentence about what to watch for tomorrow>"{planner_json}
}}{planner_instruction}"""

    try:
        response = llm_call(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=config.llm_temperature,
            max_tokens=1024,
            response_format={"type": "json_object"},
        )
        reflection = json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.warning(f"Daily reflection LLM call failed: {e}")
        reflection = {
            "key_insight": "LLM reflection unavailable",
            "market_regime": "unknown",
            "pattern_confidence": 0.0,
            "tomorrow_outlook": "Continue with current configuration",
            "planner_critique": {"failing_planner": None, "successful_planner": None},
            "structural_problem": {"detected": False, "description": ""},
        }

    reflection["date"] = side_info.get("date", "")

    # NOTE: Do NOT store reflection to semantic memory — it pollutes
    # allocation memory retrieval with noisy LLM narratives. Instead,
    # reflection insight is injected directly into the allocation prompt
    # via portfolio_runner → build_portfolio_prompt().

    logger.info(f"Daily reflection: {reflection.get('key_insight', 'N/A')}")
    return reflection


def _format_day_results(day_results: list[dict]) -> str:
    """Format day results for the reflection prompt."""
    lines = []
    for r in day_results:
        ticker = r.get("ticker", "?")
        planner = r.get("planner", "?")
        score = r.get("score", 0)
        dir_acc = r.get("dir_acc", 0)
        tools = r.get("tools_used", [])
        tools_str = ", ".join(tools[:5]) if tools else "N/A"
        lines.append(
            f"  {ticker}: planner={planner}, score={score:.3f}, "
            f"dir_acc={dir_acc:.1%}, tools=[{tools_str}]"
        )
    return "\n".join(lines) if lines else "No results."


def _format_tool_accuracy(day_results: list[dict]) -> str:
    """Compute and format per-tool accuracy for today."""
    tool_stats: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in day_results:
        for tool_name, eval_info in r.get("tool_evals", {}).items():
            correct = eval_info.get("correct")
            if correct is not None:
                tool_stats[tool_name]["total"] += 1
                if correct:
                    tool_stats[tool_name]["correct"] += 1
    lines = []
    for tool_name, stats in sorted(tool_stats.items()):
        total = stats["total"]
        acc = stats["correct"] / total if total > 0 else 0
        lines.append(f"  {tool_name}: {stats['correct']}/{total} ({acc:.0%})")
    return "\n".join(lines) if lines else "No tool evaluations today."


def _format_planner_scores(day_results: list[dict]) -> str:
    """Compute and format per-planner scores for today."""
    planner_scores: dict[str, list[float]] = defaultdict(list)
    for r in day_results:
        planner = r.get("planner", "unknown")
        planner_scores[planner].append(r.get("score", 0))
    lines = []
    for planner, scores in sorted(planner_scores.items()):
        avg = np.mean(scores)
        lines.append(f"  {planner}: avg_score={avg:.3f} (n={len(scores)})")
    return "\n".join(lines) if lines else "No planner scores today."


# ---------------------------------------------------------------------------
# Morning Planning Agent
# ---------------------------------------------------------------------------

def morning_planning(
    yesterday_reflection: dict[str, Any],
    side_info: dict[str, Any],
    memory_store,
    config,
) -> dict[str, Any]:
    """Morning planning: review yesterday's reflection and form today's strategy.

    Returns a strategy dict that can override/adjust bandit selections for
    specific tickers or conditions.
    """
    llm_model = config.reflection.llm_model or config.llm_model

    # Retrieve accumulated procedural rules
    procedural_rules = ""
    try:
        hits = memory_store.retrieve(
            query={"task_type": "procedural_strategy"},
            tiers=["procedural"],
            top_k=5,
        )
        if hits:
            procedural_rules = "\n".join(f"- {h.content}" for h in hits)
    except Exception:
        pass

    prompt = f"""You are a quantitative analyst preparing for today's trading session.

## Yesterday's Reflection
{json.dumps(yesterday_reflection, indent=2, default=str)}

## Today's Market Context (pre-market)
{json.dumps(side_info, indent=2, default=str)}

## Accumulated Strategy Rules
{procedural_rules or "No accumulated rules yet."}

## Task
Based on yesterday's lessons and today's conditions, provide a brief
market outlook. Do NOT recommend specific tools or planners — the
selection system handles that automatically. Return JSON:
{{
  "market_outlook": "<1-2 sentence outlook for today>",
  "regime": "<bullish/bearish/mixed/sector_rotation/range_bound>",
  "confidence": <0.0-1.0>
}}"""

    try:
        response = llm_call(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=config.llm_temperature,
            max_tokens=200,
            response_format={"type": "json_object"},
        )
        strategy = json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.warning(f"Morning planning LLM call failed: {e}")
        strategy = {
            "market_outlook": "Continue monitoring",
            "regime": "unknown",
            "confidence": 0.5,
        }

    logger.info(f"Morning outlook: {strategy.get('market_outlook', 'N/A')}")
    return strategy


# ---------------------------------------------------------------------------
# Planner Evolution (Code-Level)
# ---------------------------------------------------------------------------

def evolve_planner(
    reflection: dict[str, Any],
    planner_pool,
    recent_trajectories: list,
    config,
) -> Any | None:
    """LLM generates a new planner class based on reflection insights.

    Triggered when the reflection agent identifies a structural problem that
    can't be fixed by parameter tuning or prompt engineering.

    Returns:
        New BasePlanner instance if successful, None otherwise.
    """
    if not config.reflection.evolve_planners:
        return None

    llm_model = config.reflection.llm_model or config.llm_model

    planner_critique = reflection.get("planner_critique", {})
    failing_planner_name = planner_critique.get("failing_planner")

    # Get the failing planner's source code
    planner_code = "# No specific planner code available"
    if failing_planner_name:
        try:
            failing_planner = planner_pool.get(failing_planner_name)
            planner_code = inspect.getsource(failing_planner.__class__)
        except (KeyError, OSError):
            pass

    # Gather example failures and successes
    failed_examples = _format_failed_trajectories(
        recent_trajectories, failing_planner_name, limit=3
    )
    success_examples = _format_successful_trajectories(
        recent_trajectories, limit=3
    )

    structural_desc = reflection.get("structural_problem", {}).get("description", "")

    prompt = f"""You are an expert Python developer improving an AI agent's planning strategy.

CURRENT PLANNER CODE (that is failing):
```python
{planner_code}
```

FAILURE ANALYSIS:
{planner_critique.get('failure_reason', 'Unknown failure mode')}

STRUCTURAL PROBLEM:
{structural_desc}

EXAMPLE FAILURES:
{failed_examples}

EXAMPLE SUCCESSES (other planners on similar tasks):
{success_examples}

BASE CLASS (your planner must inherit from this):
```python
class BasePlanner(ABC):
    name: str = ""
    description: str = ""
    def plan(self, task, available_tools, memory_context=None) -> list[PlanStep]: ...
    def replan(self, task, completed_steps, remaining_steps, available_tools, memory_context=None) -> list[PlanStep]: ...
```

AVAILABLE ACTIONS in PlanStep:
- "call:<tool_name>" — call a tool
- "call_if_ambiguous:<tool_name>" — conditional call
- "sub_synthesize:<label>" — intermediate reasoning
- "synthesize" — final prediction
- "quick_synthesize" — quick confidence check

PlanStep constructor: PlanStep(step_id=<int>, action=<str>, rationale=<str>)

AVAILABLE TOOL NAMES (use these exact strings):
get_price_history, get_fundamentals, get_analyst_data, get_options_data,
get_earnings_data, compute_technicals, compute_quant_risk, compute_momentum,
compute_correlations, run_dcf_model, score_risk, score_composite_signal

Generate an improved planner class. Requirements:
1. Must inherit BasePlanner and implement plan() and replan()
2. Must return list[PlanStep]
3. Must address the specific failure mode identified above
4. Name it descriptively (e.g., RegimeAwarePlanner, MomentumFirstPlanner)
5. Keep it simple — under 80 lines
6. Do NOT import anything — BasePlanner and PlanStep are already available

Return ONLY the Python class code, no explanation or markdown fences."""

    try:
        response = llm_call(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=2048,
        )
        code_str = _extract_python_code(response.choices[0].message.content)

        new_planner = validate_and_instantiate(code_str)
        if new_planner is not None:
            # Check pool size limit
            if len(planner_pool.all_names) >= config.reflection.max_planner_pool_size:
                pruned = planner_pool.prune_worst_planner()
                if pruned:
                    logger.info(f"Pruned planner '{pruned}' to make room")

            planner_pool.register_dynamic(new_planner)
            logger.info(f"Evolved new planner: {new_planner.name}")
            return new_planner
        else:
            logger.warning("Evolved planner failed validation")
    except Exception as e:
        logger.warning(f"Planner evolution failed: {e}")

    return None


def validate_and_instantiate(code_str: str) -> BasePlanner | None:
    """Safely validate and instantiate LLM-generated planner code.

    1. Syntax check via ast.parse
    2. Execute in isolated namespace
    3. Find the new BasePlanner subclass
    4. Smoke test: can it produce a valid plan?
    """
    # 1. Syntax check
    try:
        ast.parse(code_str)
    except SyntaxError as e:
        logger.warning(f"Generated planner has syntax error: {e}")
        return None

    # 2. Execute in isolated namespace
    namespace = {
        "BasePlanner": BasePlanner,
        "PlanStep": PlanStep,
        "np": np,
        "ABC": __import__("abc").ABC,
        "abstractmethod": __import__("abc").abstractmethod,
    }
    try:
        exec(code_str, namespace)
    except Exception as e:
        logger.warning(f"Generated planner execution error: {e}")
        return None

    # 3. Find the new planner class
    for obj in namespace.values():
        if (
            isinstance(obj, type)
            and issubclass(obj, BasePlanner)
            and obj is not BasePlanner
            and hasattr(obj, "name")
            and obj.name  # must have a non-empty name
        ):
            try:
                instance = obj()
            except Exception as e:
                logger.warning(f"Failed to instantiate generated planner: {e}")
                continue

            # 4. Smoke test
            test_task = {"ticker": "AAPL", "date": "2025-01-15", "task_type": "price_prediction"}
            test_tools = [
                "get_price_history", "get_fundamentals", "compute_technicals",
                "compute_momentum", "run_dcf_model",
            ]
            try:
                steps = instance.plan(test_task, test_tools)
                if isinstance(steps, list) and len(steps) > 0:
                    # Verify all steps are PlanStep instances
                    if all(isinstance(s, PlanStep) for s in steps):
                        return instance
                    else:
                        logger.warning("Generated planner returned non-PlanStep objects")
                else:
                    logger.warning("Generated planner returned empty/invalid plan")
            except Exception as e:
                logger.warning(f"Generated planner smoke test failed: {e}")

    logger.warning("No valid BasePlanner subclass found in generated code")
    return None


def _extract_python_code(text: str) -> str:
    """Extract Python code from LLM response, stripping markdown fences."""
    # Try to extract from code blocks
    if "```python" in text:
        parts = text.split("```python")
        if len(parts) > 1:
            code = parts[1].split("```")[0]
            return code.strip()
    if "```" in text:
        parts = text.split("```")
        if len(parts) > 1:
            code = parts[1]
            # Remove language identifier if present
            if code.startswith("python"):
                code = code[6:]
            return code.strip()
    # Assume the whole text is code
    return text.strip()


def _format_failed_trajectories(
    trajectories: list,
    failing_planner: str | None,
    limit: int = 3,
) -> str:
    """Format recent failed trajectories for the evolution prompt."""
    failed = []
    for t in reversed(trajectories):
        if t.outcome_score < 0 and (
            failing_planner is None or t.planner_id == failing_planner
        ):
            ticker = t.task_features.get("ticker", "?")
            tools = [tc.tool_name for tc in t.tools_invoked if tc.success]
            failed.append(
                f"  {ticker}: planner={t.planner_id}, score={t.outcome_score:.3f}, "
                f"tools={tools[:5]}"
            )
            if len(failed) >= limit:
                break
    return "\n".join(failed) if failed else "No recent failures."


def _format_successful_trajectories(
    trajectories: list,
    limit: int = 3,
) -> str:
    """Format recent successful trajectories for the evolution prompt."""
    success = []
    for t in reversed(trajectories):
        if t.outcome_score > 0.3:
            ticker = t.task_features.get("ticker", "?")
            tools = [tc.tool_name for tc in t.tools_invoked if tc.success]
            success.append(
                f"  {ticker}: planner={t.planner_id}, score={t.outcome_score:.3f}, "
                f"tools={tools[:5]}"
            )
            if len(success) >= limit:
                break
    return "\n".join(success) if success else "No recent successes."


# ──────────────────────────────────────────────────────────────
# Memory Policy Evolution
# ──────────────────────────────────────────────────────────────

def evolve_memory_policy(
    reflection: dict,
    current_policies: dict[str, dict],
    bandit_stats: dict[str, dict],
    recent_trajectories: list,
    config,
) -> dict | None:
    """Use LLM to design a new memory retrieval policy.

    Analogous to evolve_planner() but produces a policy dict instead of code.
    Triggered when memory bandit average reward is consistently low.

    Args:
        reflection: Latest daily reflection output.
        current_policies: Name → policy dict for all current policies.
        bandit_stats: Thompson Sampling state per policy arm.
        recent_trajectories: Last ~50 trajectories for context.
        config: ExperimentConfig for LLM model selection.

    Returns:
        A new policy dict with a "name" key, or None if evolution fails.
    """
    llm_model = config.reflection.llm_model or config.llm_model

    # Summarize current policy performance
    policy_summary = []
    for name, stats in bandit_stats.items():
        alpha = stats.get("alpha", 1)
        beta = stats.get("beta", 1)
        mean = alpha / (alpha + beta)
        pulls = stats.get("pull_count", 0)
        policy_summary.append(f"  {name}: mean_reward={mean:.3f}, pulls={pulls}")

    # Summarize memory usage in recent trajectories
    mem_usage = {"used": 0, "empty": 0, "tiers_seen": set()}
    for t in recent_trajectories[-30:]:
        if t.memory_retrievals:
            mem_usage["used"] += 1
            for h in t.memory_retrievals:
                mem_usage["tiers_seen"].add(h.tier)
        else:
            mem_usage["empty"] += 1

    prompt = f"""You are designing memory retrieval strategies for a stock prediction agent.

The agent has a three-tier memory system:
- **episodic**: Raw per-episode records (ticker, date, planner, tool evaluations, outcome)
- **semantic**: Patterns extracted from episodic (e.g., "tool X has 80% accuracy on tech stocks")
- **procedural**: High-confidence strategies promoted from semantic (e.g., "prioritize momentum tools for NVDA")

Current memory policies and their Thompson Sampling performance:
{chr(10).join(policy_summary)}

Memory usage in last 30 episodes:
  Episodes with memory hits: {mem_usage["used"]}
  Episodes with no memory: {mem_usage["empty"]}
  Tiers seen: {sorted(mem_usage["tiers_seen"]) if mem_usage["tiers_seen"] else "none"}

Daily reflection insights:
  {reflection.get("memory_critique", "No memory-specific critique available.")}

Available retrieval formats:
- "none": no memory retrieval
- "full": return all retrieved memories verbatim
- "sliding_window": keep first N + last M memories (params: keep_first, keep_last)
- "ranked_truncate": sort by relevance, cut at token budget (params: token_budget)

Design a NEW memory policy that addresses the weaknesses above. Consider:
- Which tiers have useful content vs noise?
- Should retrieval_top_k be higher or lower?
- Would a focused retrieval (single tier) outperform multi-tier?
- What token budget balances signal vs context length?

Respond in this exact JSON format (no markdown, no explanation):
{{"name": "descriptive_name", "enabled_tiers": ["episodic", "semantic", "procedural"], "retrieval_top_k": 5, "format": "ranked_truncate", "token_budget": 400}}"""

    try:
        response = llm_call(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=256,
        )
        text = response.choices[0].message.content.strip()

        # Parse JSON from response
        import re
        json_match = re.search(r"\{[^}]+\}", text)
        if not json_match:
            logger.warning("Memory policy evolution: no JSON found in response")
            return None

        import json as _json
        policy = _json.loads(json_match.group(0))

        # Validate required fields
        name = policy.get("name", "")
        if not name or name in current_policies:
            logger.warning(f"Memory policy evolution: invalid or duplicate name '{name}'")
            return None

        valid_formats = {"none", "full", "sliding_window", "ranked_truncate"}
        if policy.get("format") not in valid_formats:
            logger.warning(f"Memory policy evolution: invalid format '{policy.get('format')}'")
            return None

        valid_tiers = {"episodic", "semantic", "procedural"}
        tiers = policy.get("enabled_tiers", [])
        if not all(t in valid_tiers for t in tiers):
            logger.warning("Memory policy evolution: invalid tier names")
            return None

        logger.info(f"Evolved new memory policy: {name} — {policy}")
        return policy

    except Exception as e:
        logger.warning(f"Memory policy evolution failed: {e}")
        return None


# ──────────────────────────────────────────────────────────────
# Architecture Reflection (Level 3): Self-modifying agent design
# ──────────────────────────────────────────────────────────────

def architecture_reflection(
    recent_results: list[dict],
    memory_diagnostics: dict,
    current_code: dict,
    config,
) -> dict | None:
    """Meta-level reflection on agent architecture.

    When performance degrades after a regime change, this function asks the
    meta-LLM to diagnose WHY the system failed and generate a code-level fix.

    The LLM sees the actual retrieval/prompt code and proposes modifications.
    Called during training+val only (frozen during test).

    Args:
        recent_results: Last 2 days of episode results with scores.
        memory_diagnostics: {
            "retrieved_memories": [...],  # what was actually retrieved
            "current_regime": "bearish",
            "memory_regimes": {"bullish": 3, "bearish": 1, "unknown": 2},
            "performance_before": avg score before regime change,
            "performance_after": avg score after regime change,
        }
        current_code: {
            "relevance_score": source code of _relevance_score(),
            "portfolio_prompt_snippet": key section of prompt template,
        }
        config: ExperimentConfig.

    Returns:
        Dict with diagnosis, target, code_patch, expected_impact — or None.
    """
    meta_model = config.reflection.meta_llm_model or config.llm_model

    # Format recent performance
    perf_before = memory_diagnostics.get("performance_before", 0)
    perf_after = memory_diagnostics.get("performance_after", 0)
    current_regime = memory_diagnostics.get("current_regime", "unknown")
    mem_regimes = memory_diagnostics.get("memory_regimes", {})
    retrieved = memory_diagnostics.get("retrieved_memories", [])

    retrieved_text = ""
    for i, mem in enumerate(retrieved[:5], 1):
        regime_tag = mem.get("regime", "unknown")
        content_preview = mem.get("content", "")[:150]
        retrieved_text += f"  {i}. [{regime_tag}] {content_preview}\n"

    relevance_code = current_code.get("relevance_score", "# not available")
    prompt_snippet = current_code.get("portfolio_prompt_snippet", "# not available")

    prompt = f"""You are a meta-learning architect analyzing why an AI trading agent's performance degraded.

## Performance Drop
Before regime change: avg score = {perf_before:.3f}
After regime change:  avg score = {perf_after:.3f}  (dropped {perf_before - perf_after:.3f})
Current market regime: {current_regime}

## Retrieved Memories (used in last predictions)
{retrieved_text or "  (no memories retrieved)"}

Memory regime distribution: {mem_regimes}
Issue: {sum(v for k, v in mem_regimes.items() if k != current_regime)} of {sum(mem_regimes.values())} retrieved memories are from a DIFFERENT regime than current.

## Current Memory Retrieval Code
```python
{relevance_code[:800]}
```

## Current Portfolio Prompt (key section)
```python
{prompt_snippet[:500]}
```

## Diagnosis Task
1. Identify the ROOT CAUSE of the performance drop
2. Determine if it's a memory/retrieval design flaw vs. just market noise
3. If it's a design flaw, propose a CODE FIX

## Propose a Fix
Generate a Python function that fixes the issue. Choose ONE target:

Target "relevance_score_patch": A function that takes (query, entry, base_score) and returns adjusted score.
Example: penalize entries from different regimes.

Target "metadata_patch": A function that takes (metadata_dict) and adds regime tags.
Example: add market_regime field based on recent returns.

Target "prompt_patch": A function that takes (prompt_str, regime_context) and returns modified prompt.
Example: prepend regime warning to prompt.

Return JSON:
{{
  "diagnosis": "<root cause in 1-2 sentences>",
  "is_design_flaw": true/false,
  "target": "relevance_score_patch|metadata_patch|prompt_patch|no_fix",
  "code_patch": "<complete Python function code>",
  "expected_impact": "<how this fixes the problem>"
}}

If the performance drop is just market noise (not a design flaw), set is_design_flaw=false and target="no_fix"."""

    try:
        response = llm_call(
            model=meta_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1024,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)

        if not result.get("is_design_flaw", False) or result.get("target") == "no_fix":
            logger.info(f"Architecture reflection: no design flaw detected — {result.get('diagnosis', '')[:80]}")
            return None

        logger.info(f"Architecture reflection diagnosed: {result.get('diagnosis', '')[:80]}")
        logger.info(f"Proposed fix target: {result.get('target')}")
        return result

    except Exception as e:
        logger.warning(f"Architecture reflection failed: {e}")
        return None
