"""Run paper baselines (Reflexion, ExpeL, FactorMiner, MetaReflexion, EvoTool)
on the portfolio allocation task.

Instead of per-ticker prediction, each baseline allocates portfolio weights
across all tickers at every bar. The baseline's learning mechanism (reflections,
lessons, skills, rules, tool-policy evolution) is driven by portfolio-level
outcomes (return, drawdown) rather than per-ticker directional accuracy.

Usage:
    # Run all paper baselines on D-small
    python -m baselines.run_portfolio_baselines --dataset d_small --parallel 3

    # Run a specific baseline
    python -m baselines.run_portfolio_baselines --dataset d_small --method reflexion --seed 42

    # Dry run
    python -m baselines.run_portfolio_baselines --dataset d_small --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ael.benchmarks.backtest import get_trading_bars, get_trading_days
from ael.benchmarks.finance import get_realized_returns
from ael.benchmarks.portfolio import (
    PortfolioState,
    build_portfolio_prompt,
    compute_portfolio_metrics,
    parse_portfolio_response,
    _equal_weight,
)
from ael.llm_utils import llm_call
from ael.tools.finance_tools import set_frequency, register_all_finance_tools
from ael.tools.registry import ToolRegistry
from ael.planner.pool import PlannerPool

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CONFIGS_DIR = PROJECT_ROOT / "experiments" / "configs"
RESULTS_DIR = PROJECT_ROOT / "logs" / "baselines"

PAPER_BASELINES = ["reflexion", "expel", "factorminer", "meta_reflexion", "evotool"]

LLM_MODEL = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"


def load_dataset_config(dataset_name: str) -> dict:
    path = CONFIGS_DIR / f"dataset_{dataset_name}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)["benchmark"]


# ---------------------------------------------------------------------------
# Baseline creation (reuse from run_baselines.py)
# ---------------------------------------------------------------------------

def _create_predictor(method: str, model: str, seed: int = 42):
    """Create the appropriate baseline predictor.

    Returns (predict_fn, predictor_obj). We only use predictor_obj for
    its learning methods (reflect, extract_lesson, etc.), NOT predict_fn,
    because portfolio mode calls llm_call() directly with the portfolio prompt.
    """
    if method == "reflexion":
        from baselines.reflexion import create_reflexion_predictor
        return create_reflexion_predictor(model=model)
    elif method == "expel":
        from baselines.expel import create_expel_predictor
        return create_expel_predictor(model=model)
    elif method == "factorminer":
        from baselines.factorminer import create_factorminer_predictor
        return create_factorminer_predictor(model=model)
    elif method == "meta_reflexion":
        from baselines.meta_reflexion import create_meta_reflexion_predictor
        return create_meta_reflexion_predictor(model=model)
    elif method == "evotool":
        from baselines.evotool import create_evotool_predictor
        return create_evotool_predictor(model=model, seed=seed)
    else:
        raise ValueError(f"Unknown baseline: {method}")


# ---------------------------------------------------------------------------
# Baseline memory context extraction
# ---------------------------------------------------------------------------

def _get_baseline_memory_context(method: str, predictor_obj) -> list[dict] | None:
    """Extract the baseline's accumulated memory/context for injection into
    the portfolio prompt.

    Each baseline stores its learned knowledge differently. We convert it into
    the list[dict] format that build_portfolio_prompt() expects for memory_context.
    """
    entries = []

    if method == "reflexion":
        # Reflexion: growing list of verbal self-reflections
        if predictor_obj.reflections:
            recent = predictor_obj.reflections[-predictor_obj.max_reflections:]
            for r in recent:
                entries.append({"tier": "episodic", "content": r})

    elif method == "expel":
        # ExpeL: flat lesson store — retrieve all recent lessons
        for lesson in predictor_obj.store.lessons[-10:]:
            tier = "procedural" if lesson.success else "episodic"
            entries.append({"tier": tier, "content": lesson.content})

    elif method == "factorminer":
        # FactorMiner: skills + experience memory
        for skill in sorted(predictor_obj.skills.values(), key=lambda s: -s.success_rate)[:5]:
            entries.append({
                "tier": "procedural",
                "content": f"Skill '{skill.name}': tools={skill.tool_sequence}, "
                           f"success_rate={skill.success_rate:.2f} (n={skill.n_uses})",
            })
        for exp in predictor_obj.experiences[-5:]:
            result = "correct" if exp.correct else "wrong"
            entries.append({
                "tier": "episodic",
                "content": f"{exp.ticker}@{exp.date}: predicted {exp.predicted_dir}, "
                           f"actual {exp.actual_dir} ({result})",
            })

    elif method == "meta_reflexion":
        # MetaReflexion: distilled meta-policy rules
        active_rules = [
            r for r in predictor_obj.rules
            if r.applications < 3 or r.success_rate >= 0.4
        ]
        for rule in active_rules[:predictor_obj.max_rules]:
            entries.append({"tier": "procedural", "content": rule.content})
        # Also include recent raw reflections as episodic context
        for r in predictor_obj.raw_reflections[-3:]:
            entries.append({"tier": "episodic", "content": r})

    elif method == "evotool":
        # EvoTool: no memory system, but expose the current policy's tool list
        if predictor_obj._current_policy:
            tools_str = ", ".join(predictor_obj._current_policy.tools)
            entries.append({
                "tier": "procedural",
                "content": f"Active tool policy (fitness={predictor_obj._current_policy.fitness:.2f}): {tools_str}",
            })

    return entries if entries else None


# ---------------------------------------------------------------------------
# Baseline learning updates (portfolio-level)
# ---------------------------------------------------------------------------

def _update_predictor_portfolio(
    method: str,
    predictor_obj,
    bar_time: str,
    weights: dict[str, float],
    port_return: float,
    ticker_returns: dict[str, float],
    tools_used: list[str],
    tool_outputs: dict[str, dict],
) -> None:
    """Call the baseline's learning update with portfolio-level outcome info.

    Instead of per-ticker (prediction vs actual direction), we pass
    portfolio allocation outcome: what we allocated, what we got back.
    We adapt each baseline's learning interface to accept this.
    """
    # Build a summary of the allocation outcome for reflection prompts
    top_allocs = sorted(weights.items(), key=lambda x: -x[1])[:5]
    alloc_summary = ", ".join(f"{t}={w:.0%}" for t, w in top_allocs)

    # Determine if it was a "good" allocation (positive return)
    success = port_return > 0

    # Synthetic prediction/actual dicts for baselines that expect them
    prediction = {
        "direction": "up" if port_return > 0 else "down",
        "return_pct": port_return,
        "weights": weights,
    }
    actual = {
        "direction": "up" if port_return > 0 else "down",
        "return_pct": port_return,
    }

    if method == "reflexion":
        # Reflect on portfolio allocation outcome
        # Override the normal reflect() with portfolio-specific context
        _reflexion_portfolio_reflect(
            predictor_obj, bar_time, alloc_summary, port_return,
            ticker_returns, tool_outputs,
        )

    elif method == "expel":
        # Extract lesson about allocation patterns
        _expel_portfolio_extract(
            predictor_obj, bar_time, alloc_summary, port_return,
            ticker_returns, tools_used,
        )

    elif method == "factorminer":
        # Update skills and experiences with portfolio outcome
        # Treat "PORTFOLIO" as the ticker and use success/failure
        predictor_obj.update(
            ticker="PORTFOLIO",
            date=bar_time,
            tools_used=tools_used,
            prediction=prediction,
            actual=actual,
        )

    elif method == "meta_reflexion":
        # Reflect and update rules based on portfolio outcome
        _meta_reflexion_portfolio_update(
            predictor_obj, bar_time, alloc_summary, port_return,
            ticker_returns, tool_outputs,
        )

    elif method == "evotool":
        # Update tool policy fitness based on portfolio return
        # Scale return to [0, 1] score: 1% return -> ~0.75 score
        score = np.clip(0.5 + port_return * 25, 0, 1)

        # Flatten tool_outputs for blame attribution
        flat_tool_outputs = {}
        for ticker, ticker_tools in tool_outputs.items():
            for tool_name, output in ticker_tools.items():
                # Keep the last ticker's output for each tool (for blame)
                flat_tool_outputs[tool_name] = output

        predictor_obj.update(
            score=score,
            prediction=prediction,
            actual=actual,
            tool_outputs=flat_tool_outputs,
        )


def _reflexion_portfolio_reflect(
    predictor_obj,
    bar_time: str,
    alloc_summary: str,
    port_return: float,
    ticker_returns: dict[str, float],
    tool_outputs: dict[str, dict],
) -> None:
    """Generate a Reflexion-style verbal self-reflection for portfolio outcome."""
    # Build ticker-level return summary
    ret_parts = []
    for ticker, ret in sorted(ticker_returns.items(), key=lambda x: -abs(x[1]))[:5]:
        ret_parts.append(f"{ticker}={ret:+.3%}")
    ret_summary = ", ".join(ret_parts)

    # Build signals summary (compact)
    signals_summary = ""
    for ticker, ticker_tools in list(tool_outputs.items())[:3]:
        for tool, output in list(ticker_tools.items())[:2]:
            if isinstance(output, dict):
                key_vals = ", ".join(
                    f"{k}={v}" for k, v in list(output.items())[:2]
                    if not isinstance(v, (list, dict))
                )
                if key_vals:
                    signals_summary += f"  {ticker}/{tool}: {key_vals}\n"

    prompt = (
        f"Portfolio allocation @{bar_time}: [{alloc_summary}]\n"
        f"Portfolio return: {port_return:+.4%}\n"
        f"Ticker returns: {ret_summary}\n"
        f"Signals:\n{signals_summary}\n"
        f"Write a 1-sentence reflection on this allocation outcome. "
        f"What allocation pattern should you adjust for next time?"
    )

    try:
        response = llm_call(
            model=predictor_obj.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=100,
        )
        reflection = response.choices[0].message.content.strip()
    except Exception:
        if port_return > 0:
            reflection = (
                f"@{bar_time}: Good allocation [{alloc_summary}] → {port_return:+.3%}."
            )
        else:
            reflection = (
                f"@{bar_time}: Bad allocation [{alloc_summary}] → {port_return:+.3%}. "
                f"Top movers: {ret_summary}."
            )

    predictor_obj.reflections.append(reflection)


def _expel_portfolio_extract(
    predictor_obj,
    bar_time: str,
    alloc_summary: str,
    port_return: float,
    ticker_returns: dict[str, float],
    tools_used: list[str],
) -> None:
    """Extract an ExpeL-style lesson from portfolio allocation outcome."""
    from baselines.expel import Lesson

    # Winners and losers
    winners = [t for t, r in ticker_returns.items() if r > 0.001]
    losers = [t for t, r in ticker_returns.items() if r < -0.001]
    success = port_return > 0
    tools_str = ", ".join(tools_used[:5]) if tools_used else "none"

    prompt = (
        f"Portfolio allocation @{bar_time}: [{alloc_summary}]\n"
        f"Result: return={port_return:+.4%} ({'GOOD' if success else 'BAD'})\n"
        f"Winners: {', '.join(winners) if winners else 'none'}\n"
        f"Losers: {', '.join(losers) if losers else 'none'}\n"
        f"Tools used: {tools_str}\n\n"
        f"Extract a single reusable lesson (1-2 sentences) about portfolio allocation "
        f"patterns. Focus on which sectors/tickers to overweight or underweight, "
        f"and which signals to trust. Start with 'RULE:'"
    )

    try:
        response = llm_call(
            model=predictor_obj.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=100,
        )
        content = response.choices[0].message.content.strip()
    except Exception:
        if success:
            content = f"SUCCESS: Allocation [{alloc_summary}] returned {port_return:+.3%}."
        else:
            content = f"FAILURE: Allocation [{alloc_summary}] returned {port_return:+.3%}."

    lesson = Lesson(ticker="PORTFOLIO", sector="multi", content=content, success=success)
    predictor_obj.store.add(lesson)


def _meta_reflexion_portfolio_update(
    predictor_obj,
    bar_time: str,
    alloc_summary: str,
    port_return: float,
    ticker_returns: dict[str, float],
    tool_outputs: dict[str, dict],
) -> None:
    """Meta-Reflexion: reflect on portfolio outcome and periodically distill rules."""
    success = port_return > 0

    # Update existing rules
    for rule in predictor_obj.rules:
        if rule.applications > 0:
            rule.apply(success)

    # Generate raw reflection
    ret_parts = []
    for ticker, ret in sorted(ticker_returns.items(), key=lambda x: -abs(x[1]))[:3]:
        ret_parts.append(f"{ticker}={ret:+.3%}")
    ret_summary = ", ".join(ret_parts)

    try:
        prompt = (
            f"Portfolio @{bar_time}: [{alloc_summary}] → {port_return:+.4%}. "
            f"{'Good' if success else 'Bad'}. Top movers: {ret_summary}. "
            f"Write a 1-sentence observation about this allocation outcome."
        )
        response = llm_call(
            model=predictor_obj.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=60,
        )
        reflection = response.choices[0].message.content.strip()
    except Exception:
        reflection = (
            f"@{bar_time}: [{alloc_summary}] → {port_return:+.3%} "
            f"({'ok' if success else 'bad'})"
        )

    predictor_obj.raw_reflections.append(reflection)
    predictor_obj._episode_count += 1

    # Periodically distill reflections into rules
    if (predictor_obj._episode_count % predictor_obj.distill_interval == 0
            and len(predictor_obj.raw_reflections) >= 3):
        predictor_obj._distill_rules()

    # Prune low-performing rules
    predictor_obj.rules = [
        r for r in predictor_obj.rules
        if r.applications < 5 or r.success_rate >= 0.3
    ]


# ---------------------------------------------------------------------------
# Core portfolio backtest for a single paper baseline
# ---------------------------------------------------------------------------

def run_single_baseline(
    method: str,
    dataset: str,
    seed: int,
    max_episodes: int | None = None,
) -> dict:
    """Run a single paper baseline on the portfolio allocation task."""
    run_name = f"pf_{method}_{dataset}_s{seed}"
    out_dir = RESULTS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check if already done
    results_file = out_dir / f"backtest_{run_name}.json"
    if results_file.exists():
        logger.info(f"[SKIP] {run_name}: already completed")
        return {"name": run_name, "status": "skipped"}

    logger.info(f"[START] {run_name}")
    start_time = time.time()

    try:
        config = load_dataset_config(dataset)
        np.random.seed(seed)

        tickers = config["tickers"]
        frequency = config.get("frequency", "1h")
        flat_threshold = config.get("flat_threshold", 0.001)
        train_end = config.get("train_end_date", "")
        val_end = config.get("val_end_date", "")

        # Set frequency for tools
        set_frequency(frequency)

        # Build tool registry + planner
        tool_registry = ToolRegistry()
        register_all_finance_tools(tool_registry)
        planner_pool = PlannerPool(enabled_planners=["sequential"])
        planner = planner_pool.get("sequential")

        # Create baseline predictor (we use predictor_obj for learning, not predict_fn)
        _, predictor_obj = _create_predictor(method, LLM_MODEL, seed=seed)

        # Initialize portfolio
        portfolio = PortfolioState()
        portfolio.set_weights(_equal_weight(tickers))

        # Get trading bars
        if frequency == "1h":
            all_bars = get_trading_bars(config["start_date"], config["end_date"], tickers)
            bars_by_day = {}
            for bar in all_bars:
                day = bar.split(" ")[0]
                bars_by_day.setdefault(day, []).append(bar)
            trading_days = sorted(bars_by_day.keys())
        else:
            trading_days = get_trading_days(config["start_date"], config["end_date"])
            bars_by_day = {d: [d] for d in trading_days}

        # Run portfolio backtest
        score_history = []
        allocation_history = []
        phase_returns = {"train": [], "val": [], "test": []}
        episode_count = 0

        for day in trading_days:
            if max_episodes and episode_count >= max_episodes:
                break

            # Determine phase
            phase = "test"
            if train_end and day <= train_end:
                phase = "train"
            elif val_end and day <= val_end:
                phase = "val"

            # Freeze learning in test phase
            frozen = (phase == "test")

            for bar_time in bars_by_day.get(day, [day]):
                if max_episodes and episode_count >= max_episodes:
                    break

                # ----- 1. Run tools for ALL tickers -----
                all_tool_outputs = {}
                all_tools_used = set()

                for ticker in tickers:
                    ticker_outputs = {}

                    # EvoTool: select policy and use its tool list
                    if method == "evotool":
                        predictor_obj.select_policy()
                        available_tools = predictor_obj.get_active_tools()
                    else:
                        available_tools = tool_registry.all_names

                    task = {"ticker": ticker, "date": bar_time}
                    if frequency == "1h":
                        task["as_of_datetime"] = bar_time

                    plan = planner.plan(task, available_tools)
                    for step in plan:
                        if step.action.startswith("call:"):
                            tool_name = step.action.split(":", 1)[1]
                            kwargs = {"ticker": ticker}
                            if task.get("as_of_datetime") and tool_name in (
                                "get_price_history", "compute_technicals",
                                "compute_quant_risk", "compute_momentum",
                                "compute_correlations",
                            ):
                                kwargs["as_of_datetime"] = task["as_of_datetime"]
                            try:
                                result = tool_registry.call(tool_name, **kwargs)
                                if result.success:
                                    ticker_outputs[tool_name] = result.output
                                    all_tools_used.add(tool_name)
                            except Exception:
                                pass

                    all_tool_outputs[ticker] = ticker_outputs

                tools_used = sorted(all_tools_used)

                # ----- 2. Build portfolio prompt with baseline's memory -----
                memory_context = _get_baseline_memory_context(method, predictor_obj)

                prompt = build_portfolio_prompt(
                    bar_time, tickers, all_tool_outputs,
                    portfolio, memory_context,
                )

                # ----- 3. LLM allocation call -----
                try:
                    response = llm_call(
                        model=LLM_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3,
                        max_tokens=512,
                    )
                    text = response.choices[0].message.content
                    new_weights = parse_portfolio_response(text, tickers)
                except Exception as e:
                    logger.warning(f"LLM allocation failed @{bar_time}: {e}")
                    new_weights = _equal_weight(tickers)

                # ----- 4. Apply allocation -----
                turnover = portfolio.set_weights(new_weights)

                # ----- 5. Get realized returns for all tickers -----
                ticker_returns = {}
                for ticker in tickers:
                    realized = get_realized_returns(
                        ticker, bar_time, [1],
                        frequency=frequency, flat_threshold=flat_threshold,
                    )
                    if 1 in realized:
                        ticker_returns[ticker] = realized[1]["return_pct"]

                # ----- 6. Update portfolio state -----
                port_return = portfolio.update(ticker_returns)
                portfolio.history[-1]["turnover"] = turnover

                phase_returns[phase].append(port_return)

                # Outcome score: scale so 1% return ~ 0.5 score
                outcome_score = float(np.clip(port_return * 50, -1, 1))

                score_history.append({
                    "episode": episode_count,
                    "bar_time": bar_time,
                    "phase": phase,
                    "port_return": port_return,
                    "total_value": portfolio.total_value,
                    "drawdown": portfolio.drawdown,
                    "turnover": turnover,
                    "score": outcome_score,
                })

                allocation_history.append({
                    "bar_time": bar_time,
                    "phase": phase,
                    "weights": dict(new_weights),
                    "port_return": port_return,
                })

                # ----- 7. Update baseline's learning (not in test phase) -----
                if not frozen:
                    _update_predictor_portfolio(
                        method, predictor_obj,
                        bar_time, new_weights, port_return,
                        ticker_returns, tools_used, all_tool_outputs,
                    )

                logger.info(
                    f"[{run_name}] Bar {episode_count}: {bar_time} [{phase}] "
                    f"return={port_return:+.3%} total={portfolio.total_value:.4f} "
                    f"turnover={turnover:.3f}"
                )

                episode_count += 1

        elapsed = time.time() - start_time

        # ----- Compute metrics -----
        overall_metrics = compute_portfolio_metrics(portfolio.history)

        phase_metrics = {}
        for p, rets in phase_returns.items():
            if rets:
                rets_arr = np.array(rets)
                vol = float(np.std(rets_arr)) if len(rets_arr) > 1 else 0
                ann_factor = 252 * 4
                ann_ret = float(np.mean(rets_arr)) * ann_factor
                ann_vol = vol * np.sqrt(ann_factor) if vol > 0 else 0
                phase_metrics[p] = {
                    "avg_return": float(np.mean(rets_arr)),
                    "total_return": float(np.prod(1 + rets_arr) - 1),
                    "volatility": vol,
                    "sharpe_ratio": ann_ret / ann_vol if ann_vol > 0 else 0,
                    "max_drawdown": float(max(
                        (h.get("drawdown", 0) for h in portfolio.history
                         if any(s["bar_time"] == h.get("bar_time") and s["phase"] == p
                                for s in score_history)),
                        default=0,
                    )),
                    "n_bars": len(rets),
                }

        result = {
            "method": method,
            "dataset": dataset,
            "seed": seed,
            "metrics": overall_metrics,
            "phase_metrics": phase_metrics,
            "score_history": score_history,
            "allocation_history": allocation_history[-20:],
            "total_episodes": episode_count,
            "elapsed": elapsed,
            "portfolio_final": {
                "total_value": portfolio.total_value,
                "max_drawdown": overall_metrics.get("max_drawdown", 0),
                "sharpe_ratio": overall_metrics.get("sharpe_ratio", 0),
            },
        }

        with open(results_file, "w") as f:
            json.dump(result, f, indent=2, default=str)

        test_m = phase_metrics.get("test", overall_metrics)
        sharpe = test_m.get("sharpe_ratio", 0)
        total_ret = test_m.get("total_return", 0)
        logger.info(
            f"[DONE] {run_name}: test return={total_ret:+.3%} "
            f"sharpe={sharpe:.3f} ({elapsed:.0f}s)"
        )
        return {"name": run_name, "status": "success", "elapsed": elapsed}

    except Exception as e:
        logger.error(f"[FAIL] {run_name}: {e}")
        import traceback
        traceback.print_exc()
        return {"name": run_name, "status": "failed", "error": str(e)}


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def _print_summary(all_results: list[dict], dataset: str) -> None:
    """Print a comparison table of all paper baselines on portfolio task."""
    print(f"\n{'='*90}")
    print(f"PAPER BASELINE PORTFOLIO RESULTS -- {dataset}")
    print(f"{'='*90}")
    print(
        f"{'Method':<22} {'Return%':>10} {'Sharpe':>10} {'MaxDD%':>10} "
        f"{'AvgTurnover':>12} {'Bars':>6} {'Time':>8}"
    )
    print("-" * 90)

    for r in all_results:
        if r["status"] != "success":
            status = r.get("status", "?")
            print(f"  {r['name']:<20} [{status}]")
            continue

        result_file = RESULTS_DIR / r["name"] / f"backtest_{r['name']}.json"
        if not result_file.exists():
            continue

        data = json.loads(result_file.read_text())
        test = data.get("phase_metrics", {}).get("test", data.get("metrics", {}))
        overall = data.get("metrics", {})

        total_ret = test.get("total_return", 0) * 100
        sharpe = test.get("sharpe_ratio", 0)
        max_dd = test.get("max_drawdown", overall.get("max_drawdown", 0)) * 100
        avg_to = overall.get("avg_turnover", 0)
        n_bars = data.get("total_episodes", 0)
        elapsed = r.get("elapsed", 0)

        print(
            f"{r['name']:<22} "
            f"{total_ret:>+10.3f} "
            f"{sharpe:>10.3f} "
            f"{max_dd:>10.3f} "
            f"{avg_to:>12.4f} "
            f"{n_bars:>6d} "
            f"{elapsed/60:>7.1f}m"
        )

    # Print final value comparison
    print(f"\n{'Method':<22} {'FinalValue':>12} {'TestSharpe':>12}")
    print("-" * 50)
    for r in all_results:
        if r["status"] != "success":
            continue
        result_file = RESULTS_DIR / r["name"] / f"backtest_{r['name']}.json"
        if not result_file.exists():
            continue
        data = json.loads(result_file.read_text())
        pf = data.get("portfolio_final", {})
        test_sharpe = data.get("phase_metrics", {}).get("test", {}).get("sharpe_ratio", 0)
        print(
            f"{r['name']:<22} "
            f"{pf.get('total_value', 1.0):>12.6f} "
            f"{test_sharpe:>12.3f}"
        )

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run paper baselines on portfolio allocation task")
    parser.add_argument("--dataset", default="d_small")
    parser.add_argument("--method", default=None,
                        help="Specific baseline (reflexion/expel/factorminer/meta_reflexion/evotool)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    methods = [args.method] if args.method else PAPER_BASELINES
    seeds = [args.seed] if args.method else args.seeds

    runs = [(m, args.dataset, s) for m in methods for s in seeds]
    logger.info(f"Paper baselines (portfolio): {len(runs)} runs, {args.parallel} workers")

    if args.dry_run:
        for m, d, s in runs:
            print(f"  pf_{m}_{d}_s{s}")
        print(f"\nTotal: {len(runs)}")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    start_time = time.time()

    if args.parallel <= 1:
        for m, d, s in runs:
            result = run_single_baseline(m, d, s, args.max_episodes)
            all_results.append(result)
    else:
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            futures = {
                executor.submit(run_single_baseline, m, d, s, args.max_episodes): f"pf_{m}_{d}_s{s}"
                for m, d, s in runs
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    result = future.result()
                    all_results.append(result)
                except Exception as e:
                    logger.error(f"[CRASH] {name}: {e}")
                    all_results.append({"name": name, "status": "crash", "error": str(e)})

    total_time = time.time() - start_time
    success = sum(1 for r in all_results if r["status"] == "success")
    failed = sum(1 for r in all_results if r["status"] in ("failed", "crash"))
    skipped = sum(1 for r in all_results if r["status"] == "skipped")

    logger.info(
        f"\nDone in {total_time:.0f}s: {success} success, {failed} failed, {skipped} skipped"
    )

    # Print summary table
    _print_summary(all_results, args.dataset)


if __name__ == "__main__":
    main()
