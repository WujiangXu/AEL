"""Portfolio allocation baselines for comparison with AEL portfolio agent.

Each baseline allocates portfolio weights across N tickers (sum to 1.0)
at every bar, given price history up to that bar.

Baselines:
    EqualWeight       — 1/N to each ticker (benchmark)
    MomentumWeighted  — weight proportional to recent positive momentum
    InverseMomentum   — contrarian: weight inversely proportional to recent return
    MinVariance       — weight inversely proportional to recent volatility
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BasePortfolioBaseline(ABC):
    """Base class for portfolio allocation baselines."""

    name: str = "base"

    @abstractmethod
    def allocate(
        self, tickers: list[str], prices: dict[str, list[float]]
    ) -> dict[str, float]:
        """Return weight allocation. Keys = tickers, values sum to 1.0.

        Args:
            tickers: List of ticker symbols.
            prices: Dict mapping ticker -> list of close prices up to current bar.

        Returns:
            Dict mapping ticker -> weight, with values summing to 1.0.
        """
        ...

    @staticmethod
    def _equal_weights(tickers: list[str]) -> dict[str, float]:
        """Fallback: equal weight across all tickers."""
        n = len(tickers)
        w = 1.0 / n
        return {t: w for t in tickers}

    @staticmethod
    def _normalize(raw: dict[str, float]) -> dict[str, float]:
        """Normalize raw scores to sum to 1.0. Falls back to equal if all zero."""
        total = sum(raw.values())
        if total < 1e-12:
            n = len(raw)
            return {t: 1.0 / n for t in raw}
        return {t: v / total for t, v in raw.items()}


class EqualWeightBaseline(BasePortfolioBaseline):
    """Always allocate 1/N to each ticker. This is the benchmark to beat."""

    name = "equal_weight"

    def allocate(
        self, tickers: list[str], prices: dict[str, list[float]]
    ) -> dict[str, float]:
        return self._equal_weights(tickers)


class MomentumWeightedBaseline(BasePortfolioBaseline):
    """Weight proportional to recent momentum.

    Tickers with positive lookback return get higher weight.
    Negative-momentum tickers receive a small floor weight to stay invested.
    """

    name = "momentum_weighted"

    def __init__(self, lookback: int = 4, floor_weight: float = 0.02):
        self.lookback = lookback
        self.floor_weight = floor_weight

    def allocate(
        self, tickers: list[str], prices: dict[str, list[float]]
    ) -> dict[str, float]:
        n = len(tickers)

        # Need at least lookback+1 prices to compute return
        min_len = self.lookback + 1
        if any(len(prices.get(t, [])) < min_len for t in tickers):
            return self._equal_weights(tickers)

        # Compute lookback returns
        returns = {}
        for t in tickers:
            p = prices[t]
            returns[t] = (p[-1] - p[-self.lookback - 1]) / p[-self.lookback - 1]

        # Positive momentum gets proportional weight; negative gets floor
        raw = {}
        for t in tickers:
            if returns[t] > 0:
                raw[t] = returns[t]
            else:
                raw[t] = 0.0

        total_positive = sum(raw.values())
        if total_positive < 1e-12:
            # All tickers have non-positive momentum — equal weight
            return self._equal_weights(tickers)

        # Allocate: positive-momentum tickers share (1 - n*floor), rest get floor
        negative_tickers = [t for t in tickers if raw[t] <= 0]
        positive_tickers = [t for t in tickers if raw[t] > 0]
        total_floor = len(negative_tickers) * self.floor_weight
        remaining = max(1.0 - total_floor, 0.01)

        weights = {}
        for t in negative_tickers:
            weights[t] = self.floor_weight
        for t in positive_tickers:
            weights[t] = remaining * (raw[t] / total_positive)

        # Re-normalize to exactly 1.0
        return self._normalize(weights)


class InverseMomentumBaseline(BasePortfolioBaseline):
    """Contrarian: weight inversely proportional to recent return.

    Buy losers, sell (underweight) winners. Mean-reversion style.
    """

    name = "inverse_momentum"

    def __init__(self, lookback: int = 4):
        self.lookback = lookback

    def allocate(
        self, tickers: list[str], prices: dict[str, list[float]]
    ) -> dict[str, float]:
        min_len = self.lookback + 1
        if any(len(prices.get(t, [])) < min_len for t in tickers):
            return self._equal_weights(tickers)

        # Compute lookback returns
        returns = {}
        for t in tickers:
            p = prices[t]
            returns[t] = (p[-1] - p[-self.lookback - 1]) / p[-self.lookback - 1]

        # Inverse: shift so all values are positive, then normalize
        # Use: score = max_return - return_i + epsilon
        max_ret = max(returns.values())
        min_ret = min(returns.values())
        spread = max_ret - min_ret
        epsilon = max(spread * 0.1, 1e-6)

        raw = {}
        for t in tickers:
            raw[t] = max_ret - returns[t] + epsilon

        return self._normalize(raw)


class MinVarianceBaseline(BasePortfolioBaseline):
    """Weight inversely proportional to recent volatility.

    Lower-vol tickers get more weight. Risk-parity inspired.
    """

    name = "min_variance"

    def __init__(self, lookback: int = 12):
        self.lookback = lookback

    def allocate(
        self, tickers: list[str], prices: dict[str, list[float]]
    ) -> dict[str, float]:
        min_len = self.lookback + 1
        if any(len(prices.get(t, [])) < min_len for t in tickers):
            return self._equal_weights(tickers)

        # Compute recent volatility (std of bar-to-bar returns)
        vols = {}
        for t in tickers:
            p = np.array(prices[t][-(self.lookback + 1):], dtype=np.float64)
            bar_returns = np.diff(p) / p[:-1]
            vol = np.std(bar_returns)
            vols[t] = vol

        # Inverse volatility weights
        raw = {}
        for t in tickers:
            if vols[t] > 1e-10:
                raw[t] = 1.0 / vols[t]
            else:
                # Near-zero vol: give a very high weight (it's the safest asset)
                raw[t] = 1e10

        return self._normalize(raw)


# Registry of all portfolio baselines
PORTFOLIO_BASELINES = {
    "equal_weight": EqualWeightBaseline,
    "momentum_weighted": MomentumWeightedBaseline,
    "inverse_momentum": InverseMomentumBaseline,
    "min_variance": MinVarianceBaseline,
}
