"""Factored Counterfactual Credit (FCC) assignment.

Four levels of credit assignment:
1. Structural: deterministic from trace (tool fail → tool blame, etc.)
2. LLM Counterfactual: LLM reasons about each module's causal contribution
3. Counterfactual probes: swap one module, partial re-run
4. Exact Shapley: 2^3=8 coalitions over (planner, tools, memory)

The LLM level (2) is a novel contribution: instead of expensive re-runs,
the LLM analyzes module outputs against ground truth to assess causality.
"""

from __future__ import annotations

import itertools
import json
import logging
from typing import Any, Callable

from ael.types import TrajectoryRecord

logger = logging.getLogger(__name__)


# Module indices for Shapley computation
MODULES = ["planner", "tools", "memory"]


def structural_credit(trajectory: TrajectoryRecord) -> dict[str, float]:
    """Level 1: Deterministic credit from execution trace.

    Rules:
    - Tool failure → negative tool credit proportional to failures
    - Plan step dead-end (no useful output) → negative planner credit
    - Irrelevant memory retrieval → negative memory credit
    - Remaining credit distributed proportionally to successful contributions
    """
    credits = {"planner": 0.0, "tools": 0.0, "memory": 0.0}

    # --- Tool credit ---
    tool_calls = trajectory.tools_invoked
    if tool_calls:
        n_success = sum(1 for t in tool_calls if t.success)
        n_fail = sum(1 for t in tool_calls if not t.success)
        total = n_success + n_fail
        if total > 0:
            credits["tools"] = (n_success - n_fail) / total  # [-1, 1]

    # --- Memory credit ---
    retrievals = trajectory.memory_retrievals
    if retrievals:
        useful = sum(1 for m in retrievals if m.was_useful)
        not_useful = sum(1 for m in retrievals if m.was_useful is False)
        total = useful + not_useful
        if total > 0:
            credits["memory"] = (useful - not_useful) / total
    else:
        credits["memory"] = 0.0  # no memory used → neutral

    # --- Planner credit ---
    plan_steps = trajectory.plan_steps
    if plan_steps:
        completed = sum(1 for s in plan_steps if s.success is True)
        failed = sum(1 for s in plan_steps if s.success is False)
        total = completed + failed
        if total > 0:
            credits["planner"] = (completed - failed) / total

    return credits


def counterfactual_credit(
    trajectory: TrajectoryRecord,
    outcome_score: float,
    baseline_outcomes: dict[str, float],
) -> dict[str, float]:
    """Level 2: Counterfactual credit via module swaps.

    For each module, compare the actual outcome with the outcome when
    that module is replaced with a baseline (default) version.

    Args:
        trajectory: The actual execution trace.
        outcome_score: The actual outcome score.
        baseline_outcomes: Dict mapping module name to the outcome score
            achieved when that module was replaced with its baseline.
            e.g., {"planner": 0.3, "tools": 0.1, "memory": 0.4}

    Returns:
        Credit dict: module → marginal contribution.
    """
    credits = {}
    for module in MODULES:
        if module in baseline_outcomes:
            # Credit = actual - counterfactual (what happens without this module)
            credits[module] = outcome_score - baseline_outcomes[module]
        else:
            credits[module] = 0.0

    # Normalize to sum to 1 if all positive
    total = sum(abs(v) for v in credits.values())
    if total > 0:
        credits = {k: v / total for k, v in credits.items()}

    return credits


def shapley_credit(
    outcome_fn: Callable[[frozenset[str]], float],
) -> dict[str, float]:
    """Level 3: Exact Shapley values over 3 modules (8 coalitions).

    Args:
        outcome_fn: Function that takes a frozenset of module names
            (subset of {"planner", "tools", "memory"}) and returns the
            outcome score for that coalition. Empty set = baseline score.

    Returns:
        Shapley value for each module.
    """
    n = len(MODULES)
    shapley = {m: 0.0 for m in MODULES}

    # Precompute all coalition values (2^3 = 8)
    coalition_values: dict[frozenset[str], float] = {}
    for r in range(n + 1):
        for combo in itertools.combinations(MODULES, r):
            coalition = frozenset(combo)
            coalition_values[coalition] = outcome_fn(coalition)

    # Compute Shapley value for each module
    for module in MODULES:
        others = [m for m in MODULES if m != module]
        n_others = len(others)

        for r in range(n_others + 1):
            for combo in itertools.combinations(others, r):
                S = frozenset(combo)
                S_with = S | {module}

                # Marginal contribution of adding this module to coalition S
                marginal = coalition_values[S_with] - coalition_values[S]

                # Shapley weight: |S|!(n-|S|-1)! / n!
                import math
                weight = (
                    math.factorial(len(S))
                    * math.factorial(n - len(S) - 1)
                    / math.factorial(n)
                )
                shapley[module] += weight * marginal

    return shapley


def llm_credit(
    trajectory: TrajectoryRecord,
    outcome_score: float,
    ground_truth: dict | None = None,
    model: str = "",
) -> dict[str, float]:
    """LLM-driven counterfactual credit: the LLM reasons about each module's
    causal contribution by examining actual outputs against ground truth.

    Instead of expensive re-runs (traditional counterfactual) or coalition
    enumeration (Shapley), the LLM performs causal analysis in one call:
    "Given what actually happened, which module helped/hurt most?"

    Args:
        trajectory: Full execution trace with module outputs.
        outcome_score: Actual outcome (-1 to 1).
        ground_truth: The realized outcome (e.g., ticker returns, correct answer).
        model: LLM model to use for credit analysis.

    Returns:
        Credit dict: module → credit score in [-1, 1].
    """
    from ael.llm_utils import llm_call

    # Build compact summaries of each module's contribution
    # --- Planner ---
    plan_summary = f"Strategy: {trajectory.planner_id}"
    if trajectory.plan_steps:
        n_steps = len(trajectory.plan_steps)
        n_ok = sum(1 for s in trajectory.plan_steps if s.success is True)
        plan_summary += f" ({n_ok}/{n_steps} steps succeeded)"
    if trajectory.prediction:
        weights = trajectory.prediction.get("weights", {})
        if weights:
            plan_summary += f"\nAllocation: {json.dumps(weights, default=str)[:300]}"

    # --- Tools ---
    # For portfolio mode (many tickers × tools), summarize per-ticker
    tool_outputs = getattr(trajectory, "_tool_outputs", {})
    if tool_outputs and isinstance(tool_outputs, dict) and len(tool_outputs) > 1:
        # Portfolio mode: per-ticker weight, tool signals, and actual return
        weights = trajectory.prediction.get("weights", {}) if trajectory.prediction else {}
        per_ticker_gt = ground_truth.get("per_ticker", {}) if ground_truth else {}

        tool_signals = []
        for ticker, tools in list(tool_outputs.items())[:10]:
            # Weight allocated
            w = weights.get(ticker, 0)
            w_str = f"weight={w * 100:.1f}%"

            # Tool signal direction
            mom = tools.get("compute_momentum", {})
            score = tools.get("score_composite_signal", {})
            tech = tools.get("compute_technicals", {})
            sig_parts = []
            if mom:
                mom4 = mom.get("return_4bar", "?")
                sig_parts.append(f"mom4={mom4:+.4f}" if isinstance(mom4, (int, float)) else f"mom4={mom4}")
            if score:
                sig_parts.append(f"signal={score.get('signal', '?')}")
            if tech:
                rsi = tech.get("rsi_14", {})
                rsi_val = rsi.get("value", rsi) if isinstance(rsi, dict) else rsi
                sig_parts.append(f"rsi={rsi_val}")
            # Infer direction from momentum sign
            direction = "neutral"
            if mom and isinstance(mom.get("return_4bar"), (int, float)):
                direction = "bullish" if mom["return_4bar"] > 0 else "bearish"
            elif score and score.get("signal") in ("buy", "sell"):
                direction = "bullish" if score["signal"] == "buy" else "bearish"
            tools_tag = f"tools={direction}({', '.join(sig_parts)})" if sig_parts else "tools=none"

            # Actual return from ground truth
            tk_gt = per_ticker_gt.get(ticker, {})
            tk_ret = tk_gt.get("return", None)
            ret_str = f"return={tk_ret * 100:+.1f}%" if isinstance(tk_ret, (int, float)) else "return=N/A"

            tool_signals.append(f"{ticker}: {w_str}, {tools_tag}, {ret_str}")
        tools_summary = "\n".join(tool_signals)
    else:
        # Single-ticker mode: individual tool calls
        tool_signals = []
        for tc in trajectory.tools_invoked[:8]:
            if tc.success and tc.output:
                out_str = str(tc.output)[:150]
                tool_signals.append(f"{tc.tool_name}: {out_str}")
            elif not tc.success:
                tool_signals.append(f"{tc.tool_name}: FAILED ({tc.error or 'unknown'})")
        tools_summary = "\n".join(tool_signals) if tool_signals else "No tools invoked"

    # --- Memory ---
    mem_summary = "No memory retrieved"
    if trajectory.memory_retrievals:
        mem_parts = []
        for m in trajectory.memory_retrievals[:3]:
            useful_tag = ""
            if m.was_useful is True:
                useful_tag = " [useful]"
            elif m.was_useful is False:
                useful_tag = " [not useful]"
            mem_parts.append(f"[{m.tier}] {m.content[:100]}{useful_tag}")
        mem_summary = "\n".join(mem_parts)

    # --- Ground truth ---
    gt_str = ""
    if ground_truth:
        gt_str = f"\nGround truth: {json.dumps(ground_truth, default=str)[:300]}"

    # --- Prediction ---
    pred_str = json.dumps(trajectory.prediction, default=str)[:200] if trajectory.prediction else "N/A"

    outcome_label = "GOOD" if outcome_score > 0 else "BAD"

    prompt = f"""You are a credit assignment analyst. An AI agent used 3 modules to make a prediction. Analyze which module helped or hurt.

## Module Outputs

### Planner
{plan_summary}

### Tools (signals gathered)
{tools_summary}

### Memory (past experience retrieved)
{mem_summary}

## Result
Prediction: {pred_str}
Outcome score: {outcome_score:+.3f} ({outcome_label}){gt_str}

## Task
Assign credit to each module: a score from -1.0 (actively harmful) to +1.0 (most helpful).
Consider:
- Did the tools provide signals that pointed toward the correct outcome?
- Did memory retrieve relevant past experience, or irrelevant noise?
- Did the planner choose an appropriate strategy for this situation?

Reply with ONLY a JSON object: {{"planner": 0.X, "tools": 0.X, "memory": 0.X}}"""

    try:
        response = llm_call(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # low temp for consistent attribution
            max_tokens=60,
        )
        text = response.choices[0].message.content.strip()

        # Parse JSON response
        import re
        # Try direct JSON parse
        try:
            credits = json.loads(text)
        except json.JSONDecodeError:
            # Extract JSON from text
            m = re.search(r'\{[^}]+\}', text)
            if m:
                credits = json.loads(m.group())
            else:
                raise ValueError(f"Could not parse LLM credit response: {text[:100]}")

        # Validate and clamp
        result = {}
        for module in MODULES:
            val = float(credits.get(module, 0.0))
            result[module] = max(-1.0, min(1.0, val))

        return result

    except Exception as e:
        logger.warning(f"LLM credit assignment failed: {e}, falling back to structural")
        return structural_credit(trajectory)


def compute_fcc(
    trajectory: TrajectoryRecord,
    outcome_score: float,
    baseline_outcomes: dict[str, float] | None = None,
    outcome_fn: Callable[[frozenset[str]], float] | None = None,
    method: str = "fcc",
    weights: tuple[float, float, float] = (0.2, 0.3, 0.5),
    ground_truth: dict | None = None,
    llm_model: str = "",
) -> dict[str, float]:
    """Compute Factored Counterfactual Credit using the specified method.

    Args:
        trajectory: Execution trace.
        outcome_score: Actual outcome score.
        baseline_outcomes: For counterfactual level (module → baseline score).
        outcome_fn: For Shapley level (coalition → outcome score).
        method: "structural", "counterfactual", "shapley", "fcc",
                "llm" (LLM-only), or "llm_fcc" (LLM replaces counterfactual in FCC).
        weights: (structural, counterfactual/llm, shapley) weights for FCC combination.
        ground_truth: Realized outcome for LLM credit analysis.
        llm_model: Model identifier for LLM credit calls.

    Returns:
        Final credit assignment dict.
    """
    if method == "structural":
        return structural_credit(trajectory)

    if method == "counterfactual":
        if baseline_outcomes is None:
            return structural_credit(trajectory)
        return counterfactual_credit(trajectory, outcome_score, baseline_outcomes)

    if method == "shapley":
        if outcome_fn is None:
            return structural_credit(trajectory)
        return shapley_credit(outcome_fn)

    if method == "llm":
        return llm_credit(trajectory, outcome_score, ground_truth, llm_model)

    # llm_fcc: structural + LLM counterfactual + Shapley (LLM replaces re-runs)
    if method == "llm_fcc":
        struct = structural_credit(trajectory)
        llm_cf = llm_credit(trajectory, outcome_score, ground_truth, llm_model)

        if outcome_fn is not None:
            shap = shapley_credit(outcome_fn)
        else:
            shap = llm_cf  # fall back to LLM credit if no Shapley fn

        w_s, w_c, w_sh = weights
        combined = {}
        for module in MODULES:
            combined[module] = (
                w_s * struct[module]
                + w_c * llm_cf[module]
                + w_sh * shap[module]
            )
        return combined

    # FCC: combine all three levels
    struct = structural_credit(trajectory)

    if baseline_outcomes is not None:
        counter = counterfactual_credit(trajectory, outcome_score, baseline_outcomes)
    else:
        counter = struct

    if outcome_fn is not None:
        shap = shapley_credit(outcome_fn)
    else:
        shap = counter

    # Weighted combination (configurable)
    w_s, w_c, w_sh = weights
    combined = {}
    for module in MODULES:
        combined[module] = (
            w_s * struct[module]
            + w_c * counter[module]
            + w_sh * shap[module]
        )

    return combined


def uniform_credit() -> dict[str, float]:
    """Baseline: equal credit to all modules."""
    return {m: 1.0 / len(MODULES) for m in MODULES}


def llm_uniform_credit(
    trajectory: TrajectoryRecord,
    outcome_score: float,
    ground_truth: dict | None = None,
    model: str = "",
    alpha: float = 0.5,
) -> dict[str, float]:
    """Blend uniform credit with LLM causal reasoning.

    Simpler than FCC: no heuristic baselines, no Shapley coalitions.
    Uniform provides stability; LLM provides the causal signal.

    credit = alpha * uniform(0.333) + (1-alpha) * llm_credit

    Args:
        alpha: Mixing weight. 0.0 = pure LLM, 1.0 = pure uniform.
    """
    uni = uniform_credit()
    llm = llm_credit(trajectory, outcome_score, ground_truth, model)
    return {
        m: alpha * uni[m] + (1 - alpha) * llm[m]
        for m in MODULES
    }


def portfolio_credit(
    weights: dict[str, float],
    ticker_returns: dict[str, float],
    tool_outputs: dict[str, dict],
    memory_hits: list,
    equal_weight: float,
    planner_alpha: float | None = None,
    no_memory_baseline: float | None = None,
) -> dict[str, float]:
    """Portfolio-aware credit using actual per-ticker data.

    Unlike heuristic baselines that degrade to uniform, this uses the
    causal structure of portfolio allocation:
    - Planner credit: did active allocation beat equal-weight?
    - Tool credit: did tool signals match reality on overweighted stocks?
    - Memory credit: did having memory improve outcome vs not having it?

    Args:
        weights: Allocation weights {ticker: weight}.
        ticker_returns: Realized returns {ticker: return_pct}.
        tool_outputs: Per-ticker tool outputs {ticker: {tool: output}}.
        memory_hits: Memory retrievals for this episode.
        equal_weight: 1/N baseline weight.
        planner_alpha: Pre-computed planner_return - equal_return (if available).
        no_memory_baseline: Mean outcome_score from episodes without memory.
    """
    credits = {"planner": 0.0, "tools": 0.0, "memory": 0.0}

    # --- Planner credit: alpha (value-add over equal-weight) ---
    if planner_alpha is not None:
        # Normalize: typical alpha is ±0.001, scale to [-1, 1]
        credits["planner"] = float(max(-1.0, min(1.0, planner_alpha * 200)))
    else:
        # Compute from weights vs equal-weight
        port_ret = sum(weights.get(t, 0) * ticker_returns.get(t, 0)
                       for t in ticker_returns)
        eq_ret = sum(equal_weight * ticker_returns.get(t, 0)
                     for t in ticker_returns)
        alpha = port_ret - eq_ret
        credits["planner"] = float(max(-1.0, min(1.0, alpha * 200)))

    # --- Tool credit: weighted directional accuracy ---
    # For each ticker the planner bet on, did the tools point the right way?
    tool_score = 0.0
    total_weight = 0.0
    for ticker, weight in weights.items():
        if weight < 0.01:
            continue
        actual_ret = ticker_returns.get(ticker, 0)
        if abs(actual_ret) < 1e-6:
            continue  # flat — no signal to evaluate

        actual_dir = 1.0 if actual_ret > 0 else -1.0
        ticker_tools = tool_outputs.get(ticker, {})

        # Aggregate tool signal direction for this ticker
        signals = []
        for tool_name, output in ticker_tools.items():
            if not isinstance(output, dict):
                continue
            # Extract directional signal (reuse existing logic)
            from ael.evolution.loop import EvolutionLoop
            sig = EvolutionLoop.extract_tool_direction(tool_name, output)
            if sig == "up":
                signals.append(1.0)
            elif sig == "down":
                signals.append(-1.0)

        if signals:
            avg_signal = sum(signals) / len(signals)
            # +1 if tools agreed with actual, -1 if disagreed
            agreement = avg_signal * actual_dir
            # Weight by allocation — overweighted stocks matter more
            deviation = abs(weight - equal_weight)
            tool_score += agreement * (weight + deviation)
            total_weight += weight + deviation

    if total_weight > 0:
        credits["tools"] = float(max(-1.0, min(1.0, tool_score / total_weight)))

    # --- Memory credit: differential (with memory vs baseline without) ---
    if memory_hits:
        if no_memory_baseline is not None:
            # outcome relative to no-memory baseline
            port_ret = sum(weights.get(t, 0) * ticker_returns.get(t, 0)
                           for t in ticker_returns)
            credits["memory"] = float(max(-1.0, min(1.0,
                (port_ret - no_memory_baseline) * 100)))
        else:
            # No baseline data yet — neutral
            credits["memory"] = 0.0
    else:
        credits["memory"] = 0.0  # no memory used → neutral

    return credits
