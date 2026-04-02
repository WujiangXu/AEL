"""Evaluate portfolio allocation baselines on hourly price data.

Usage:
    # Run all portfolio baselines on D-small
    python -m baselines.evaluate_portfolio --dataset d_small

    # Run a specific baseline
    python -m baselines.evaluate_portfolio --dataset d_small --method momentum_weighted

    # Run with custom seeds (affects nothing for deterministic baselines, kept for interface parity)
    python -m baselines.evaluate_portfolio --dataset d_large --seeds 42 123 456

    # List available methods
    python -m baselines.evaluate_portfolio --list
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

from baselines.portfolio_baselines import PORTFOLIO_BASELINES, BasePortfolioBaseline

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CONFIGS_DIR = PROJECT_ROOT / "experiments" / "configs"
RESULTS_DIR = PROJECT_ROOT / "logs" / "baselines"

# Annualization factor: 4 bars/day * ~252 trading days/year = 1008 bars/year
BARS_PER_YEAR = 1008


# ---------------------------------------------------------------------------
# Data loading (reuse patterns from baselines/evaluate.py)
# ---------------------------------------------------------------------------

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

    rows: list[dict] = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "datetime": row.get("Datetime")
                    or row.get("Date")
                    or list(row.values())[0],
                    "close": float(row["Close"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "open": float(row["Open"]),
                    "volume": int(float(row["Volume"])),
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def _date_of(dt_str: str) -> str:
    """Extract date portion from datetime string like '2025-01-06 09:30:00-05:00'."""
    return dt_str.split(" ")[0]


def simulate_portfolio(
    baseline: BasePortfolioBaseline,
    dataset_config: dict,
) -> dict:
    """Run a portfolio baseline through the dataset and compute metrics.

    At each bar we:
      1. Observe the new close prices for all tickers.
      2. Mark-to-market the portfolio using the new prices.
      3. Call baseline.allocate() to get target weights for the NEXT bar.
      4. Rebalance (compute turnover).

    Returns dict with per-bar portfolio values, weights, and aggregate metrics.
    """
    tickers = dataset_config["tickers"]
    start = dataset_config["start_date"]
    end = dataset_config["end_date"]
    train_end = dataset_config.get("train_end_date", "")
    val_end = dataset_config.get("val_end_date", "")
    frequency = dataset_config.get("frequency", "1h")

    # Load price series for all tickers
    all_price_rows: dict[str, list[dict]] = {}
    for ticker in tickers:
        all_price_rows[ticker] = load_prices(ticker, frequency)

    # Build aligned bar index: collect all unique datetimes that appear
    # across all tickers (they should be the same for hourly bars).
    # Use the first ticker's datetime sequence as the reference.
    ref_ticker = tickers[0]
    bar_datetimes = [row["datetime"] for row in all_price_rows[ref_ticker]]

    # Filter to dataset date range
    bar_indices: list[int] = []
    for i, dt in enumerate(bar_datetimes):
        d = _date_of(dt)
        if start <= d <= end:
            bar_indices.append(i)

    if not bar_indices:
        raise ValueError(f"No bars found in date range {start} to {end}")

    # Build per-ticker close arrays aligned to reference bars
    # Map ticker -> {datetime -> close} for fast lookup
    close_lookup: dict[str, dict[str, float]] = {}
    for ticker in tickers:
        close_lookup[ticker] = {
            row["datetime"]: row["close"] for row in all_price_rows[ticker]
        }

    # Full close array per ticker (from dataset start, for price history)
    # We need history BEFORE the first bar in range, so use all bars from the
    # beginning of the file up to the current bar.
    close_arrays: dict[str, list[float]] = {}
    for ticker in tickers:
        close_arrays[ticker] = [row["close"] for row in all_price_rows[ticker]]

    # ---- Simulation state ----
    initial_value = 1_000_000.0  # start with $1M
    portfolio_value = initial_value
    # Current weights (start equal)
    n = len(tickers)
    current_weights: dict[str, float] = {t: 1.0 / n for t in tickers}

    # Track results per bar
    bar_records: list[dict] = []
    prev_weights: dict[str, float] | None = None

    for step, bar_idx in enumerate(bar_indices):
        dt = bar_datetimes[bar_idx]
        d = _date_of(dt)

        # Determine phase
        if train_end and d <= train_end:
            phase = "train"
        elif val_end and d <= val_end:
            phase = "val"
        else:
            phase = "test"

        # Current close prices for all tickers
        current_prices: dict[str, float] = {}
        for ticker in tickers:
            current_prices[ticker] = close_lookup[ticker].get(dt, np.nan)

        # ---- Mark-to-market ----
        if step > 0:
            # Compute portfolio return from previous bar to current bar
            prev_dt = bar_datetimes[bar_indices[step - 1]]
            port_return = 0.0
            for ticker in tickers:
                prev_close = close_lookup[ticker].get(prev_dt, np.nan)
                curr_close = current_prices[ticker]
                if np.isnan(prev_close) or np.isnan(curr_close) or prev_close < 1e-8:
                    ticker_return = 0.0
                else:
                    ticker_return = (curr_close - prev_close) / prev_close
                port_return += current_weights[ticker] * ticker_return

            portfolio_value *= (1.0 + port_return)
        else:
            port_return = 0.0

        # ---- Compute turnover (from previous target weights to current weights) ----
        turnover = 0.0
        if prev_weights is not None:
            # Drift: after the bar return, actual weights have drifted.
            # For simplicity we measure turnover as the L1 distance between
            # old target weights and new target weights (a common approximation).
            pass  # computed below after new allocation

        # ---- Get new target weights ----
        # Build price history up to this bar for each ticker
        price_history: dict[str, list[float]] = {}
        for ticker in tickers:
            price_history[ticker] = close_arrays[ticker][: bar_idx + 1]

        new_weights = baseline.allocate(tickers, price_history)

        # Compute turnover vs previous target weights
        if prev_weights is not None:
            turnover = 0.5 * sum(
                abs(new_weights.get(t, 0) - prev_weights.get(t, 0)) for t in tickers
            )

        bar_records.append(
            {
                "bar_idx": bar_idx,
                "datetime": dt,
                "date": d,
                "phase": phase,
                "portfolio_value": portfolio_value,
                "bar_return": port_return,
                "turnover": turnover,
                "weights": dict(new_weights),
            }
        )

        prev_weights = dict(new_weights)
        current_weights = dict(new_weights)

    # ---- Compute aggregate metrics ----
    metrics = _compute_portfolio_metrics(bar_records, initial_value)
    return {"bar_records": bar_records, "metrics": metrics}


def _compute_portfolio_metrics(bar_records: list[dict], initial_value: float) -> dict:
    """Compute total return, Sharpe, max drawdown, avg turnover per phase."""
    result: dict = {}

    for phase in ("train", "val", "test", "overall"):
        if phase == "overall":
            subset = bar_records
        else:
            subset = [r for r in bar_records if r["phase"] == phase]

        if len(subset) < 2:
            continue

        values = [r["portfolio_value"] for r in subset]
        returns = [r["bar_return"] for r in subset[1:]]  # skip first bar (return=0)
        turnovers = [r["turnover"] for r in subset[1:]]

        # Total return
        total_return = (values[-1] - values[0]) / values[0]

        # Sharpe ratio (annualized)
        returns_arr = np.array(returns, dtype=np.float64)
        if len(returns_arr) > 1 and np.std(returns_arr) > 1e-12:
            sharpe = (np.mean(returns_arr) / np.std(returns_arr)) * np.sqrt(
                BARS_PER_YEAR
            )
        else:
            sharpe = 0.0

        # Max drawdown
        cummax = np.maximum.accumulate(values)
        drawdowns = (np.array(values) - cummax) / cummax
        max_drawdown = float(np.min(drawdowns))  # most negative

        # Average turnover
        avg_turnover = float(np.mean(turnovers)) if turnovers else 0.0

        phase_key = f"phase_{phase}" if phase != "overall" else "overall"
        result[phase_key] = {
            "total_return": float(total_return),
            "total_return_pct": float(total_return * 100),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_drawdown),
            "max_drawdown_pct": float(max_drawdown * 100),
            "avg_turnover": float(avg_turnover),
            "n_bars": len(subset),
            "start_value": float(values[0]),
            "end_value": float(values[-1]),
        }

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate portfolio allocation baselines"
    )
    parser.add_argument(
        "--dataset",
        default="d_small",
        help="Dataset config name (d_small, d_large, d1, d2, ...)",
    )
    parser.add_argument(
        "--method", default=None, help="Specific method to run (default: all)"
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42],
        help="Seeds (reserved for stochastic baselines)",
    )
    parser.add_argument(
        "--list", action="store_true", help="List available portfolio baselines"
    )
    args = parser.parse_args()

    if args.list:
        print("Available portfolio baselines:")
        for name, cls in PORTFOLIO_BASELINES.items():
            doc = cls.__doc__ or ""
            print(f"  {name}: {doc.strip().split(chr(10))[0]}")
        return

    # Load dataset
    config = load_dataset_config(args.dataset)
    logger.info(
        f"Dataset: {args.dataset} ({config['start_date']} to {config['end_date']})"
    )
    logger.info(f"Tickers: {config['tickers']}")

    # Select methods
    if args.method:
        if args.method not in PORTFOLIO_BASELINES:
            raise ValueError(
                f"Unknown method '{args.method}'. "
                f"Available: {list(PORTFOLIO_BASELINES.keys())}"
            )
        methods = {args.method: PORTFOLIO_BASELINES[args.method]}
    else:
        methods = PORTFOLIO_BASELINES

    # Ensure output directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict] = {}

    for name, baseline_cls in methods.items():
        logger.info(f"Running portfolio baseline: {name}")

        seed_metrics: list[dict] = []
        for seed in args.seeds:
            baseline = baseline_cls()
            result = simulate_portfolio(baseline, config)
            metrics = result["metrics"]
            seed_metrics.append(metrics)

            # Log test-phase summary
            test = metrics.get("phase_test", metrics.get("overall", {}))
            logger.info(
                f"  {name} (seed={seed}): "
                f"return={test.get('total_return_pct', 0):.2f}%, "
                f"sharpe={test.get('sharpe_ratio', 0):.3f}, "
                f"mdd={test.get('max_drawdown_pct', 0):.2f}%"
            )

        # Average across seeds
        avg_metrics = _average_metrics(seed_metrics)
        all_results[name] = avg_metrics

        # Save per-method results
        out_path = RESULTS_DIR / f"portfolio_{name}_{args.dataset}.json"
        save_data = {
            "method": name,
            "dataset": args.dataset,
            "seeds": args.seeds,
            "seed_metrics": seed_metrics,
            "avg": avg_metrics,
        }
        with open(out_path, "w") as f:
            json.dump(save_data, f, indent=2)
        logger.info(f"  Saved: {out_path}")

    # ---- Print comparison table ----
    _print_comparison_table(all_results, args.dataset)


def _average_metrics(seed_metrics: list[dict]) -> dict:
    """Average portfolio metrics across seeds."""
    if len(seed_metrics) == 1:
        return seed_metrics[0]

    avg: dict = {}
    for phase_key in ("phase_train", "phase_val", "phase_test", "overall"):
        phase_dicts = [m.get(phase_key, {}) for m in seed_metrics if phase_key in m]
        if not phase_dicts:
            continue
        avg[phase_key] = {}
        for metric in (
            "total_return",
            "total_return_pct",
            "sharpe_ratio",
            "max_drawdown",
            "max_drawdown_pct",
            "avg_turnover",
            "n_bars",
            "start_value",
            "end_value",
        ):
            vals = [d.get(metric, 0) for d in phase_dicts if metric in d]
            if vals:
                avg[phase_key][metric] = float(np.mean(vals))
    return avg


def _print_comparison_table(all_results: dict[str, dict], dataset_name: str) -> None:
    """Print a clear comparison table of all baselines."""
    phases = ["phase_test", "overall"]
    phase_labels = {"phase_test": "TEST PHASE", "overall": "OVERALL"}

    for phase_key in phases:
        label = phase_labels[phase_key]
        has_data = any(phase_key in r for r in all_results.values())
        if not has_data:
            continue

        print(f"\n{'=' * 85}")
        print(f"PORTFOLIO BASELINES -- {dataset_name} ({label})")
        print(f"{'=' * 85}")
        print(
            f"{'Method':<22} {'Return%':>10} {'Sharpe':>10} {'MaxDD%':>10} "
            f"{'AvgTurnover':>12} {'Bars':>6}"
        )
        print("-" * 85)

        for name, metrics in all_results.items():
            phase = metrics.get(phase_key, {})
            if not phase:
                continue
            print(
                f"{name:<22} "
                f"{phase.get('total_return_pct', 0):>+10.3f} "
                f"{phase.get('sharpe_ratio', 0):>10.3f} "
                f"{phase.get('max_drawdown_pct', 0):>10.3f} "
                f"{phase.get('avg_turnover', 0):>12.4f} "
                f"{int(phase.get('n_bars', 0)):>6d}"
            )

        # Highlight benchmark (equal_weight)
        ew = all_results.get("equal_weight", {}).get(phase_key, {})
        if ew:
            print("-" * 85)
            print(
                f"{'(benchmark: equal_weight)':<22} "
                f"{ew.get('total_return_pct', 0):>+10.3f} "
                f"{ew.get('sharpe_ratio', 0):>10.3f} "
                f"{ew.get('max_drawdown_pct', 0):>10.3f} "
                f"{ew.get('avg_turnover', 0):>12.4f} "
                f"{int(ew.get('n_bars', 0)):>6d}"
            )

    print()


if __name__ == "__main__":
    main()
