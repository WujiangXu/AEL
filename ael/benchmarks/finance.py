"""Finance benchmark: stock price prediction evaluation.

Task: Given a ticker and date, predict price direction and magnitude at
1/7/14-day horizons. LLM is forbidden from web search — must rely solely
on registered MCP tools and accumulated memory/experience.

Evaluation:
- Directional accuracy (predicted vs actual up/down/flat)
- MAPE (mean absolute percentage error)
- Weighted accuracy (stronger confidence → higher weight)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import yfinance as yf

logger = logging.getLogger(__name__)


@dataclass
class PredictionTask:
    """A single price prediction task."""
    ticker: str
    date: str               # YYYY-MM-DD, the "as-of" date
    horizons: list[int] = field(default_factory=lambda: [1, 7, 14])
    sector: str = ""
    task_type: str = "price_prediction"
    # Additional context
    holding: dict | None = None
    portfolio: dict | None = None
    historical: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "date": self.date,
            "horizons": self.horizons,
            "sector": self.sector,
            "task_type": self.task_type,
            "holding": self.holding,
            "portfolio": self.portfolio,
            "historical": self.historical,
        }


@dataclass
class PredictionResult:
    """Agent's prediction for a single task."""
    ticker: str
    date: str
    predictions: dict[int, dict] = field(default_factory=dict)
    # predictions[horizon] = {"direction": "up"/"down"/"flat",
    #                          "magnitude": 0.05, "confidence": 0.8}
    raw_output: str = ""


@dataclass
class EvaluationResult:
    """Evaluation of a prediction against realized returns."""
    ticker: str
    date: str
    horizon: int
    predicted_direction: str    # "up", "down", "flat"
    actual_direction: str
    predicted_magnitude: float  # predicted % change
    actual_magnitude: float     # actual % change
    confidence: float
    direction_correct: bool
    absolute_error: float       # |predicted - actual|


# Pre-loaded returns cache (populated by _load_returns_cache)
_returns_cache: dict[str, dict[str, dict]] = {}  # ticker → {datetime → {return_1bar, ...}}


def _load_returns_cache(ticker: str, frequency: str = "1d") -> dict[str, dict]:
    """Load pre-cached returns for a ticker. Cached in memory after first load."""
    if ticker in _returns_cache:
        return _returns_cache[ticker]

    import csv
    from pathlib import Path

    if frequency == "1h":
        path = Path(__file__).parent.parent.parent / "data" / "returns_1h" / f"{ticker}.csv"
    else:
        path = Path(__file__).parent.parent.parent / "data" / "returns" / f"{ticker}.csv"

    if not path.exists():
        logger.warning(f"No pre-cached returns for {ticker} at {path}")
        _returns_cache[ticker] = {}
        return {}

    lookup = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            dt = row["datetime"]
            entry = {}
            for key, val in row.items():
                if key.startswith(("return_", "direction_", "price_end_")) and val:
                    entry[key] = float(val) if key.startswith(("return_", "price_end_")) else val
            lookup[dt] = entry

    _returns_cache[ticker] = lookup
    return lookup


def get_realized_returns(
    ticker: str,
    as_of_date: str,
    horizons: list[int] = [1, 7, 14],
    frequency: str = "1d",
    flat_threshold: float = 0.005,
) -> dict[int, dict]:
    """Get realized returns at given horizons from a reference date/datetime.

    For hourly (frequency="1h"): reads from pre-cached data/returns_1h/.
    For daily (frequency="1d"): falls back to yfinance download.

    Args:
        ticker: Stock ticker symbol.
        as_of_date: The date/datetime predictions were made.
        horizons: List of horizons (bars for hourly, days for daily).
        frequency: "1h" or "1d".
        flat_threshold: Below this abs return → "flat" direction.

    Returns:
        Dict mapping horizon → {"price_start", "price_end", "return_pct", "direction"}.
    """
    if frequency == "1h":
        return _get_realized_returns_cached(ticker, as_of_date, horizons, flat_threshold)
    return _get_realized_returns_yfinance(ticker, as_of_date, horizons, flat_threshold)


def _get_realized_returns_cached(
    ticker: str, as_of_datetime: str, horizons: list[int], flat_threshold: float
) -> dict[int, dict]:
    """Read realized returns from pre-cached CSV (no API calls)."""
    cache = _load_returns_cache(ticker, frequency="1h")
    entry = cache.get(as_of_datetime, {})

    if not entry:
        return {}

    results = {}
    for h in horizons:
        ret = entry.get(f"return_{h}bar")
        direction = entry.get(f"direction_{h}bar")
        price_end = entry.get(f"price_end_{h}bar")

        if ret is None or direction is None:
            continue

        results[h] = {
            "price_start": 0.0,  # not stored in cache, not needed for eval
            "price_end": float(price_end) if price_end else 0.0,
            "return_pct": float(ret),
            "direction": direction,
        }

    return results


def _get_realized_returns_yfinance(
    ticker: str, as_of_date: str, horizons: list[int], flat_threshold: float
) -> dict[int, dict]:
    """Fetch realized returns via yfinance (daily, legacy path)."""
    import pandas as pd
    from datetime import datetime, timedelta

    start_dt = datetime.strptime(as_of_date[:10], "%Y-%m-%d")
    max_horizon = max(horizons)
    end_dt = start_dt + timedelta(days=max_horizon + 10)

    try:
        data = yf.download(
            ticker,
            start=start_dt.strftime("%Y-%m-%d"),
            end=end_dt.strftime("%Y-%m-%d"),
            progress=False,
        )
    except Exception as e:
        logger.warning(f"Failed to fetch returns for {ticker}: {e}")
        return {}

    if data.empty or len(data) < 2:
        return {}

    close = data["Close"].values.flatten()
    dates = data.index
    start_price = close[0]

    results = {}
    for h in horizons:
        target_date = start_dt + timedelta(days=h)
        future_dates = dates[dates >= pd.Timestamp(target_date)]
        if len(future_dates) == 0:
            idx = len(close) - 1
        else:
            idx = dates.get_loc(future_dates[0])

        end_price = close[idx]
        ret_pct = (end_price - start_price) / start_price

        if abs(ret_pct) < flat_threshold:
            direction = "flat"
        elif ret_pct > 0:
            direction = "up"
        else:
            direction = "down"

        results[h] = {
            "price_start": float(start_price),
            "price_end": float(end_price),
            "return_pct": float(ret_pct),
            "direction": direction,
        }

    return results


def evaluate_prediction(
    prediction: PredictionResult,
    realized: dict[int, dict],
) -> list[EvaluationResult]:
    """Compare a prediction against realized returns.

    Returns one EvaluationResult per horizon.
    """
    results = []
    for horizon, actual in realized.items():
        pred = prediction.predictions.get(horizon, {})
        pred_dir = pred.get("direction", "flat")
        pred_mag = pred.get("magnitude", 0.0)
        confidence = pred.get("confidence", 0.5)

        actual_dir = actual["direction"]
        actual_mag = actual["return_pct"]

        results.append(EvaluationResult(
            ticker=prediction.ticker,
            date=prediction.date,
            horizon=horizon,
            predicted_direction=pred_dir,
            actual_direction=actual_dir,
            predicted_magnitude=pred_mag,
            actual_magnitude=actual_mag,
            confidence=confidence,
            direction_correct=(pred_dir == actual_dir),
            absolute_error=abs(pred_mag - actual_mag),
        ))

    return results


def compute_metrics(evaluations: list[EvaluationResult]) -> dict:
    """Compute aggregate metrics from a list of evaluations.

    Returns:
        Dict with directional_accuracy, mape, weighted_accuracy per horizon
        and overall.
    """
    if not evaluations:
        return {}

    # Group by horizon
    from collections import defaultdict
    by_horizon = defaultdict(list)
    for ev in evaluations:
        by_horizon[ev.horizon].append(ev)

    metrics = {}
    all_correct = []
    all_errors = []
    all_weighted = []

    for horizon, evs in sorted(by_horizon.items()):
        correct = [e.direction_correct for e in evs]
        errors = [e.absolute_error for e in evs]
        weighted = [
            e.confidence if e.direction_correct else 0.0
            for e in evs
        ]

        dir_acc = np.mean(correct)
        mape = np.mean(errors)
        w_acc = np.mean(weighted) if weighted else 0.0

        # Use "bar" suffix for hourly, "d" for daily (auto-detect from horizon values)
        suffix = "bar" if max(by_horizon.keys()) <= 10 else "d"
        metrics[f"{horizon}{suffix}"] = {
            "directional_accuracy": float(dir_acc),
            "mape": float(mape),
            "weighted_accuracy": float(w_acc),
            "n_samples": len(evs),
        }

        all_correct.extend(correct)
        all_errors.extend(errors)
        all_weighted.extend(weighted)

    metrics["overall"] = {
        "directional_accuracy": float(np.mean(all_correct)),
        "mape": float(np.mean(all_errors)),
        "weighted_accuracy": float(np.mean(all_weighted)),
        "n_samples": len(evaluations),
    }

    return metrics


def compute_outcome_score(evaluations: list[EvaluationResult]) -> float:
    """Compute a single scalar outcome score from evaluations.

    Score in [-1, 1]:
    - Directional accuracy contributes [0, 1]
    - Low MAPE adds bonus
    - Confidence calibration bonus/penalty

    Used as the reward signal for bandit updates.
    """
    if not evaluations:
        return 0.0

    # Directional accuracy (baseline)
    dir_acc = np.mean([e.direction_correct for e in evaluations])

    # MAPE penalty (lower is better)
    mape = np.mean([e.absolute_error for e in evaluations])
    mape_bonus = max(0, 0.2 - mape)  # bonus if MAPE < 20%

    # Calibration: high confidence + correct = good, high confidence + wrong = bad
    calibration = np.mean([
        e.confidence * (1.0 if e.direction_correct else -1.0)
        for e in evaluations
    ])

    # Combine: dir_acc is primary, mape_bonus and calibration are secondary
    score = 0.6 * (dir_acc * 2 - 1) + 0.2 * mape_bonus + 0.2 * calibration
    return float(np.clip(score, -1, 1))
