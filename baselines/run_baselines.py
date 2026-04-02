"""Run paper baselines (Reflexion, ExpeL, FactorMiner) on hourly data.

These baselines use the same tool pipeline as AEL but with different
memory/learning mechanisms. Each makes LLM calls via Bedrock.

Usage:
    # Run all paper baselines on D-small
    python -m baselines.run_baselines --dataset d_small --parallel 3

    # Run a specific baseline
    python -m baselines.run_baselines --dataset d_small --method reflexion --seed 42

    # Dry run
    python -m baselines.run_baselines --dataset d_small --dry-run
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

from ael.config import ExperimentConfig
from ael.benchmarks.backtest import get_trading_bars, get_trading_days, extract_task_features
from ael.benchmarks.finance import (
    PredictionTask, PredictionResult, EvaluationResult,
    evaluate_prediction, compute_metrics, compute_outcome_score,
    get_realized_returns,
)
from ael.tools.finance_tools import (
    set_frequency, register_all_finance_tools,
)
from ael.tools.registry import ToolRegistry
from ael.planner.pool import PlannerPool
from ael.evolution.meta_controller import TaskFeatures

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CONFIGS_DIR = PROJECT_ROOT / "experiments" / "configs"
RESULTS_DIR = PROJECT_ROOT / "logs" / "baselines"

PAPER_BASELINES = ["reflexion", "expel", "factorminer", "meta_reflexion", "evotool"]


def load_dataset_config(dataset_name: str) -> dict:
    path = CONFIGS_DIR / f"dataset_{dataset_name}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)["benchmark"]


def run_single_baseline(
    method: str,
    dataset: str,
    seed: int,
    max_episodes: int | None = None,
) -> dict:
    """Run a single paper baseline on a dataset."""
    run_name = f"{method}_{dataset}_s{seed}"
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
        horizons = config.get("prediction_horizons", [1, 2, 4])
        flat_threshold = config.get("flat_threshold", 0.001)
        train_end = config.get("train_end_date", "")
        val_end = config.get("val_end_date", "")

        # Set frequency for tools
        set_frequency(frequency)

        # Build tool registry + planner (all baselines use same tools)
        tool_registry = ToolRegistry()
        register_all_finance_tools(tool_registry)
        planner_pool = PlannerPool(enabled_planners=["sequential"])
        planner = planner_pool.get("sequential")

        # Create baseline predictor
        model = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"
        predictor, predictor_obj = _create_predictor(method, model, seed=seed)

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

        # Run backtest
        evaluations = []
        phase_evaluations = {"train": [], "val": [], "test": []}
        score_history = []
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

                for ticker in tickers:
                    if max_episodes and episode_count >= max_episodes:
                        break

                    # Build task
                    task = PredictionTask(
                        ticker=ticker, date=bar_time, horizons=horizons,
                    ).to_dict()
                    if frequency == "1h":
                        task["as_of_datetime"] = bar_time

                    # Execute tools — EvoTool uses evolutionary policy selection
                    if method == "evotool":
                        predictor_obj.select_policy()
                        available_tools = predictor_obj.get_active_tools()
                    else:
                        available_tools = tool_registry.all_names
                    plan = planner.plan(task, available_tools)
                    tool_outputs = {}
                    tools_used = []

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
                                    tool_outputs[tool_name] = result.output
                                    tools_used.append(tool_name)
                            except Exception:
                                pass

                    # Get prediction from baseline
                    prediction = predictor(task, tool_outputs, None)

                    # Get realized returns
                    realized = get_realized_returns(
                        ticker, bar_time, horizons,
                        frequency=frequency, flat_threshold=flat_threshold,
                    )

                    if realized:
                        evals = evaluate_prediction(prediction, realized)
                        evaluations.extend(evals)
                        phase_evaluations[phase].extend(evals)

                        score = compute_outcome_score(evals)
                        score_history.append({
                            "episode": episode_count,
                            "ticker": ticker,
                            "date": bar_time,
                            "phase": phase,
                            "score": score,
                        })

                        # Update baseline's learning mechanism (not in test phase)
                        if not frozen:
                            _update_predictor(
                                method, predictor_obj, ticker, bar_time,
                                task.get("sector", ""),
                                prediction.predictions.get(horizons[0], {}),
                                realized.get(horizons[0], {}),
                                tools_used,
                                tool_outputs=tool_outputs,
                            )

                    episode_count += 1

        elapsed = time.time() - start_time

        # Compute metrics
        metrics = compute_metrics(evaluations)
        phase_metrics = {}
        for p in ("train", "val", "test"):
            if phase_evaluations[p]:
                phase_metrics[p] = compute_metrics(phase_evaluations[p])

        result = {
            "method": method,
            "dataset": dataset,
            "seed": seed,
            "metrics": metrics,
            "phase_metrics": phase_metrics,
            "score_history": score_history,
            "total_episodes": episode_count,
            "elapsed": elapsed,
        }

        with open(results_file, "w") as f:
            json.dump(result, f, indent=2, default=str)

        test_da = phase_metrics.get("test", metrics).get("overall", {}).get("directional_accuracy", 0)
        logger.info(f"[DONE] {run_name}: test DA={test_da:.3f} ({elapsed:.0f}s)")
        return {"name": run_name, "status": "success", "elapsed": elapsed}

    except Exception as e:
        logger.error(f"[FAIL] {run_name}: {e}")
        import traceback
        traceback.print_exc()
        return {"name": run_name, "status": "failed", "error": str(e)}


def _create_predictor(method: str, model: str, seed: int = 42):
    """Create the appropriate baseline predictor."""
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


def _update_predictor(method, predictor_obj, ticker, date, sector, prediction, actual,
                       tools_used, tool_outputs=None):
    """Call the baseline's learning update."""
    if method == "reflexion":
        predictor_obj.reflect(ticker, date, prediction, actual, tool_outputs=tool_outputs)
    elif method == "expel":
        predictor_obj.extract_lesson(ticker, date, sector, prediction, actual, tools_used)
    elif method == "factorminer":
        predictor_obj.update(ticker, date, tools_used, prediction, actual)
    elif method == "meta_reflexion":
        predictor_obj.reflect_and_update(ticker, date, prediction, actual, tool_outputs=tool_outputs)
    elif method == "evotool":
        predictor_obj.update(
            score=1.0 if prediction.get("direction") == actual.get("direction") else 0.0,
            prediction=prediction, actual=actual, tool_outputs=tool_outputs or {},
        )


def main():
    parser = argparse.ArgumentParser(description="Run paper baselines")
    parser.add_argument("--dataset", default="d_small")
    parser.add_argument("--method", default=None, help="Specific baseline (reflexion/expel/factorminer)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    methods = [args.method] if args.method else PAPER_BASELINES
    seeds = [args.seed] if args.method else args.seeds

    runs = [(m, args.dataset, s) for m in methods for s in seeds]
    logger.info(f"Paper baselines: {len(runs)} runs, {args.parallel} workers")

    if args.dry_run:
        for m, d, s in runs:
            print(f"  {m}_{d}_s{s}")
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
                executor.submit(run_single_baseline, m, d, s, args.max_episodes): f"{m}_{d}_s{s}"
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

    logger.info(f"\nDone in {total_time:.0f}s: {success} success, {failed} failed")

    # Print summary
    print(f"\n{'='*70}")
    print(f"PAPER BASELINE RESULTS — {args.dataset}")
    print(f"{'='*70}")
    for r in all_results:
        if r["status"] == "success":
            # Read result file
            result_file = RESULTS_DIR / r["name"] / f"backtest_{r['name']}.json"
            if result_file.exists():
                data = json.loads(result_file.read_text())
                test = data.get("phase_metrics", {}).get("test", data.get("metrics", {}))
                da = test.get("overall", {}).get("directional_accuracy", 0)
                print(f"  {r['name']}: DA={da:.3f} ({r.get('elapsed',0)/60:.0f}m)")


if __name__ == "__main__":
    main()
