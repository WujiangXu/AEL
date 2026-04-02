"""Evaluate baseline predictors on pre-cached hourly data.

Usage:
    # Run all simple baselines on D-small
    python -m baselines.evaluate --dataset d_small

    # Run a specific baseline with specific seeds
    python -m baselines.evaluate --dataset d_small --method momentum --seeds 42 123 456

    # List available methods
    python -m baselines.evaluate --list
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from baselines.predictors import SIMPLE_PREDICTORS, BasePredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CONFIGS_DIR = PROJECT_ROOT / "experiments" / "configs"
RESULTS_DIR = PROJECT_ROOT / "logs" / "baselines"


def load_dataset_config(dataset_name: str) -> dict:
    """Load dataset config from YAML."""
    path = CONFIGS_DIR / f"dataset_{dataset_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Dataset config not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)["benchmark"]


def load_prices(ticker: str, frequency: str = "1h") -> list[dict]:
    """Load price data as list of {datetime, close, high, low, open, volume}."""
    if frequency == "1h":
        path = DATA_DIR / "prices_1h" / f"{ticker}.csv"
    else:
        path = DATA_DIR / "prices" / f"{ticker}.csv"

    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "datetime": row.get("Datetime") or row.get("Date") or list(row.values())[0],
                "close": float(row["Close"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "open": float(row["Open"]),
                "volume": int(float(row["Volume"])),
            })
    return rows


def load_returns(ticker: str) -> dict[str, dict]:
    """Load pre-cached returns. Returns {datetime_str → {return_1bar, direction_1bar, ...}}."""
    path = DATA_DIR / "returns_1h" / f"{ticker}.csv"
    lookup = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            dt = row["datetime"]
            entry = {}
            for key in row:
                if key.startswith("return_") or key.startswith("direction_") or key.startswith("price_end_"):
                    val = row[key]
                    if val and val != "":
                        entry[key] = float(val) if key.startswith(("return_", "price_end_")) else val
            lookup[dt] = entry
    return lookup


def evaluate_predictor(
    predictor: BasePredictor,
    dataset_config: dict,
    seed: int = 42,
) -> dict:
    """Run a predictor on a dataset and compute metrics.

    Returns:
        Dict with per-horizon DA, MAPE, overall metrics, and per-episode details.
    """
    tickers = dataset_config["tickers"]
    start = dataset_config["start_date"]
    end = dataset_config["end_date"]
    train_end = dataset_config.get("train_end_date", "")
    val_end = dataset_config.get("val_end_date", "")
    horizons = dataset_config.get("prediction_horizons", [1, 2, 4])
    flat_threshold = dataset_config.get("flat_threshold", 0.001)
    frequency = dataset_config.get("frequency", "1h")

    # Load all data
    all_prices = {}
    all_returns = {}
    for ticker in tickers:
        all_prices[ticker] = load_prices(ticker, frequency)
        all_returns[ticker] = load_returns(ticker)

    # Evaluation accumulators
    evaluations = []  # (horizon, predicted_dir, actual_dir, magnitude_err, phase)

    for ticker in tickers:
        prices = all_prices[ticker]
        returns = all_returns[ticker]

        for i, bar in enumerate(prices):
            dt = bar["datetime"]

            # Determine phase
            date_str = dt.split(" ")[0]
            if train_end and date_str <= train_end:
                phase = "train"
            elif val_end and date_str <= val_end:
                phase = "val"
            else:
                phase = "test"

            # Skip if outside date range
            if date_str < start or date_str > end:
                continue

            # Build close history up to this bar
            close_history = [p["close"] for p in prices[:i + 1]]

            # Get predictions
            prediction = predictor.predict(close_history, horizons)

            # Get realized returns
            ret_data = returns.get(dt, {})

            for h in horizons:
                actual_dir = ret_data.get(f"direction_{h}bar")
                actual_ret = ret_data.get(f"return_{h}bar")

                if actual_dir is None or actual_ret is None:
                    continue

                pred = prediction[h]
                pred_dir = pred["direction"]
                pred_mag = pred.get("magnitude", 0.0)

                evaluations.append({
                    "ticker": ticker,
                    "datetime": dt,
                    "horizon": h,
                    "phase": phase,
                    "predicted_dir": pred_dir,
                    "actual_dir": actual_dir,
                    "predicted_mag": pred_mag,
                    "actual_return": actual_ret,
                    "correct": pred_dir == actual_dir,
                })

    # Compute metrics
    return _compute_metrics(evaluations, horizons)


def _compute_metrics(evaluations: list[dict], horizons: list[int]) -> dict:
    """Compute DA, MAPE per horizon and per phase."""
    results = {"total_evaluations": len(evaluations)}

    for phase in ("train", "val", "test", "overall"):
        if phase == "overall":
            subset = evaluations
        else:
            subset = [e for e in evaluations if e["phase"] == phase]

        if not subset:
            continue

        phase_metrics = {}
        for h in horizons:
            h_evals = [e for e in subset if e["horizon"] == h]
            if not h_evals:
                continue

            correct = sum(1 for e in h_evals if e["correct"])
            da = correct / len(h_evals)

            # MAPE
            mape_vals = []
            for e in h_evals:
                if abs(e["actual_return"]) > 1e-8:
                    mape_vals.append(abs(e["predicted_mag"] - abs(e["actual_return"])) / abs(e["actual_return"]))

            mape = np.mean(mape_vals) if mape_vals else 0.0

            phase_metrics[f"{h}bar"] = {
                "directional_accuracy": da,
                "mape": float(mape),
                "count": len(h_evals),
            }

        # Overall across horizons
        all_correct = sum(1 for e in subset if e["correct"])
        phase_metrics["overall"] = {
            "directional_accuracy": all_correct / len(subset) if subset else 0,
            "count": len(subset),
        }

        results[f"phase_{phase}" if phase != "overall" else "metrics"] = phase_metrics

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline predictors")
    parser.add_argument("--dataset", default="d_small", help="Dataset config name (d_small, d_large)")
    parser.add_argument("--method", default=None, help="Specific method to run (default: all)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--list", action="store_true", help="List available methods")
    args = parser.parse_args()

    if args.list:
        print("Available simple baselines:")
        for name, cls in SIMPLE_PREDICTORS.items():
            print(f"  {name}: {cls.__doc__.strip().split(chr(10))[0]}")
        return

    # Load dataset
    config = load_dataset_config(args.dataset)
    logger.info(f"Dataset: {args.dataset} ({config['start_date']} to {config['end_date']})")
    logger.info(f"Tickers: {config['tickers']}")
    logger.info(f"Horizons: {config.get('prediction_horizons', [1, 2, 4])} bars")

    # Select methods
    methods = {args.method: SIMPLE_PREDICTORS[args.method]} if args.method else SIMPLE_PREDICTORS

    # Run evaluations
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for name, predictor_cls in methods.items():
        seed_results = []
        for seed in args.seeds:
            predictor = predictor_cls(seed=seed) if name == "random" else predictor_cls()
            result = evaluate_predictor(predictor, config, seed=seed)
            seed_results.append(result)

            # Extract test-phase DA for display
            test_metrics = result.get("phase_test", result.get("metrics", {}))
            overall = test_metrics.get("overall", {})
            da = overall.get("directional_accuracy", 0)
            logger.info(f"  {name} (seed={seed}): test DA = {da:.3f}")

        # Average across seeds
        avg_result = _average_results(seed_results)
        all_results[name] = avg_result

        # Save per-method result
        out_path = RESULTS_DIR / f"baseline_{name}_{args.dataset}.json"
        with open(out_path, "w") as f:
            json.dump({"method": name, "dataset": args.dataset, "seeds": args.seeds,
                        "seed_results": seed_results, "avg": avg_result}, f, indent=2)

    # Print summary table
    print(f"\n{'='*70}")
    print(f"BASELINE RESULTS — {args.dataset} (test phase)")
    print(f"{'='*70}")

    horizons = config.get("prediction_horizons", [1, 2, 4])
    header = f"{'Method':<16}"
    for h in horizons:
        header += f" {'DA-' + str(h) + 'bar':>10}"
    header += f" {'Overall':>10}"
    print(header)
    print("-" * 70)

    for name, result in all_results.items():
        test = result.get("phase_test", result.get("metrics", {}))
        row = f"{name:<16}"
        for h in horizons:
            da = test.get(f"{h}bar", {}).get("directional_accuracy", 0)
            row += f" {da:>10.3f}"
        overall_da = test.get("overall", {}).get("directional_accuracy", 0)
        row += f" {overall_da:>10.3f}"
        print(row)


def _average_results(seed_results: list[dict]) -> dict:
    """Average metrics across seeds."""
    if not seed_results:
        return {}

    # Average the "metrics" (overall) and "phase_test" dicts
    avg = {}
    for key in ("metrics", "phase_train", "phase_val", "phase_test"):
        phase_dicts = [r.get(key, {}) for r in seed_results if key in r]
        if not phase_dicts:
            continue
        avg[key] = {}
        for horizon_key in phase_dicts[0]:
            vals = [d.get(horizon_key, {}) for d in phase_dicts if horizon_key in d]
            if not vals:
                continue
            avg[key][horizon_key] = {}
            for metric in ("directional_accuracy", "mape"):
                metric_vals = [v.get(metric, 0) for v in vals if metric in v]
                if metric_vals:
                    avg[key][horizon_key][metric] = float(np.mean(metric_vals))
            # Keep count from first seed
            if "count" in vals[0]:
                avg[key][horizon_key]["count"] = vals[0]["count"]
    return avg


if __name__ == "__main__":
    main()
