"""Run HyperAgent baseline on portfolio allocation task.

Fair comparison with AEL: same tools, same data, same LLM, same test freeze.

Usage:
    python -m baselines.run_hyperagent --dataset d_full --seed 42 --generations 20
    python -m baselines.run_hyperagent --dataset d_full --seeds 42 123 456
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ael.benchmarks.backtest import get_trading_bars
from ael.benchmarks.finance import get_realized_returns
from ael.benchmarks.portfolio import PortfolioState, compute_portfolio_metrics, _equal_weight
from ael.tools.finance_tools import set_frequency, register_all_finance_tools
from ael.tools.registry import ToolRegistry
from ael.planner.pool import PlannerPool

from baselines.hyperagent import HyperAgentPortfolio

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CONFIGS_DIR = PROJECT_ROOT / "experiments" / "configs"
RESULTS_DIR = PROJECT_ROOT / "logs" / "baselines"
LLM_MODEL = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"


def load_dataset_config(dataset_name: str) -> dict:
    path = CONFIGS_DIR / f"dataset_{dataset_name}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)["benchmark"]


def run_single(
    dataset: str,
    seed: int,
    n_generations: int = 10,
) -> dict:
    """Run HyperAgent on portfolio allocation."""
    run_name = f"pf_hyperagent_{dataset}_s{seed}"
    out_dir = RESULTS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

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

        set_frequency(frequency)

        # Build tool registry
        tool_registry = ToolRegistry()
        register_all_finance_tools(tool_registry)
        planner_pool = PlannerPool(enabled_planners=["sequential"])
        planner = planner_pool.get("sequential")

        # Get all bars
        all_bars = get_trading_bars(config["start_date"], config["end_date"], tickers)

        # Phase 1: Collect training data (run tools on all training bars)
        logger.info(f"[{run_name}] Collecting training data...")
        train_bars_data = []
        tool_outputs_by_bar = {}

        for bar_time in all_bars:
            day = bar_time.split(" ")[0]
            phase = "test"
            if train_end and day <= train_end:
                phase = "train"
            elif val_end and day <= val_end:
                phase = "val"

            # Run tools for all tickers
            bar_tool_outputs = {}
            for ticker in tickers:
                ticker_outputs = {}
                task = {"ticker": ticker, "date": bar_time}
                if frequency == "1h":
                    task["as_of_datetime"] = bar_time

                plan = planner.plan(task, tool_registry.all_names)
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
                        except Exception:
                            pass

                bar_tool_outputs[ticker] = ticker_outputs

            tool_outputs_by_bar[bar_time] = bar_tool_outputs

            # Get realized returns
            ticker_returns = {}
            for ticker in tickers:
                realized = get_realized_returns(
                    ticker, bar_time, [1],
                    frequency=frequency, flat_threshold=flat_threshold,
                )
                if 1 in realized:
                    ticker_returns[ticker] = realized[1]["return_pct"]

            bar_data = {
                "bar_time": bar_time,
                "phase": phase,
                "ticker_returns": ticker_returns,
            }

            if phase in ("train", "val"):
                train_bars_data.append(bar_data)

        logger.info(f"[{run_name}] Training data: {len(train_bars_data)} bars")

        # Phase 2: Run HyperAgent evolution on training data
        agent = HyperAgentPortfolio(
            model=LLM_MODEL,
            tool_registry=tool_registry,
            n_generations=n_generations,
        )

        agent.run_evolution(
            train_bars=train_bars_data,
            tickers=tickers,
            tool_outputs_by_bar=tool_outputs_by_bar,
        )

        logger.info(f"[{run_name}] Evolution complete. Best train sharpe: {agent.best_sharpe:.3f}")

        # Phase 3: Evaluate on ALL bars (including test, frozen)
        portfolio = PortfolioState()
        portfolio.set_weights(_equal_weight(tickers))
        score_history = []
        phase_returns = {"train": [], "val": [], "test": []}

        for i, bar_time in enumerate(all_bars):
            day = bar_time.split(" ")[0]
            phase = "test"
            if train_end and day <= train_end:
                phase = "train"
            elif val_end and day <= val_end:
                phase = "val"

            tool_out = tool_outputs_by_bar.get(bar_time, {})
            portfolio_state = {
                "weights": dict(portfolio.weights),
                "total_value": portfolio.total_value,
                "history": portfolio.history[-10:],
            }

            # Use evolved allocation function (frozen — same code for all phases)
            weights = agent.allocate(bar_time, tickers, tool_out, portfolio_state)

            turnover = portfolio.set_weights(weights)

            # Get returns
            ticker_returns = {}
            for ticker in tickers:
                realized = get_realized_returns(
                    ticker, bar_time, [1],
                    frequency=frequency, flat_threshold=flat_threshold,
                )
                if 1 in realized:
                    ticker_returns[ticker] = realized[1]["return_pct"]

            port_return = portfolio.update(ticker_returns)
            portfolio.history[-1]["turnover"] = turnover
            phase_returns[phase].append(port_return)

            score_history.append({
                "episode": i,
                "bar_time": bar_time,
                "phase": phase,
                "port_return": port_return,
                "total_value": portfolio.total_value,
                "drawdown": portfolio.drawdown,
                "turnover": turnover,
                "score": float(np.clip(port_return * 50, -1, 1)),
            })

        elapsed = time.time() - start_time

        # Compute metrics
        overall_metrics = compute_portfolio_metrics(portfolio.history)

        phase_metrics = {}
        for p, rets in phase_returns.items():
            if rets:
                arr = np.array(rets)
                vol = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0
                ann_factor = 252 * 4
                ann_ret = float(np.mean(arr)) * ann_factor
                ann_vol = vol * np.sqrt(ann_factor) if vol > 0 else 0
                sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

                cum = np.cumprod(1 + arr)
                peak = np.maximum.accumulate(cum)
                dd = (cum - peak) / peak
                max_dd = float(np.min(dd)) if len(dd) > 0 else 0

                phase_metrics[p] = {
                    "total_return": float(np.prod(1 + arr) - 1),
                    "sharpe_ratio": sharpe,
                    "sharpe": sharpe,
                    "max_drawdown": max_dd,
                    "n_bars": len(rets),
                }

        result = {
            "method": "hyperagent",
            "dataset": dataset,
            "seed": seed,
            "n_generations": n_generations,
            "metrics": overall_metrics,
            "phase_metrics": phase_metrics,
            "score_history": score_history,
            "evolution_log": agent.get_evolution_log(),
            "final_code": agent.get_final_code(),
            "total_episodes": len(all_bars),
            "elapsed": elapsed,
        }

        with open(results_file, "w") as f:
            json.dump(result, f, indent=2, default=str)

        test_m = phase_metrics.get("test", {})
        test_sharpe = test_m.get("sharpe", 0)
        test_ret = test_m.get("total_return", 0)

        logger.info(
            f"[DONE] {run_name}: test sharpe={test_sharpe:.3f} "
            f"return={test_ret:+.3%} ({elapsed:.0f}s)"
        )
        return {"name": run_name, "status": "success", "elapsed": elapsed}

    except Exception as e:
        logger.error(f"[FAIL] {run_name}: {e}")
        traceback.print_exc()
        return {"name": run_name, "status": "failed", "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Run HyperAgent baseline on portfolio allocation")
    parser.add_argument("--dataset", default="d_full")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="Delete existing results and re-run")
    args = parser.parse_args()

    seeds = args.seeds or [args.seed]

    if args.force:
        import shutil
        for s in seeds:
            run_name = f"pf_hyperagent_{args.dataset}_s{s}"
            old_dir = RESULTS_DIR / run_name
            if old_dir.exists():
                shutil.rmtree(old_dir)
                print(f"  Deleted {old_dir}")

    if args.dry_run:
        for s in seeds:
            print(f"  pf_hyperagent_{args.dataset}_s{s} ({args.generations} generations)")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        result = run_single(args.dataset, seed, args.generations)

        if result["status"] == "success":
            # Print result
            rf = RESULTS_DIR / result["name"] / f"backtest_{result['name']}.json"
            data = json.loads(rf.read_text())
            test = data.get("phase_metrics", {}).get("test", {})
            print(f"\n{result['name']}: test_sharpe={test.get('sharpe', 0):.3f} "
                  f"test_return={test.get('total_return', 0):.3%}")

            # Print evolution log
            evo = data.get("evolution_log", [])
            if evo:
                print(f"  Evolution ({len(evo)} generations):")
                for e in evo:
                    status = "ACCEPT" if e["accepted"] else "reject"
                    meta = " [meta-mod]" if e.get("meta_modified") else ""
                    print(f"    Gen {e['gen']}: sharpe={e['sharpe']:.3f} {status}{meta}")


if __name__ == "__main__":
    main()
