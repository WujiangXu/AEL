"""Simple baseline predictors (no LLM, no external deps beyond numpy).

Each predictor takes a price history and returns a prediction dict.
All predictors are stateless — they don't learn across episodes.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod

import numpy as np


class BasePredictor(ABC):
    """Base class for all baseline predictors."""

    name: str = "base"

    @abstractmethod
    def predict(self, close_history: list[float], horizons: list[int]) -> dict[int, dict]:
        """Predict direction/magnitude for each horizon.

        Args:
            close_history: Historical close prices up to (and including) current bar.
            horizons: List of bar horizons to predict (e.g., [1, 2, 4]).

        Returns:
            Dict mapping horizon → {"direction", "magnitude", "confidence"}.
        """
        ...

    def _classify(self, magnitude: float, flat_threshold: float = 0.001) -> str:
        if abs(magnitude) < flat_threshold:
            return "flat"
        return "up" if magnitude > 0 else "down"


class RandomPredictor(BasePredictor):
    """Uniform random direction prediction. Establishes chance-level performance (~33% DA)."""

    name = "random"

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    def predict(self, close_history: list[float], horizons: list[int]) -> dict[int, dict]:
        return {
            h: {
                "direction": self._rng.choice(["up", "down", "flat"]),
                "magnitude": 0.0,
                "confidence": 0.33,
            }
            for h in horizons
        }


class BuyAndHoldPredictor(BasePredictor):
    """Always predicts 'up'. Captures market direction bias."""

    name = "buy_and_hold"

    def predict(self, close_history: list[float], horizons: list[int]) -> dict[int, dict]:
        return {
            h: {"direction": "up", "magnitude": 0.005, "confidence": 0.5}
            for h in horizons
        }


class MomentumPredictor(BasePredictor):
    """Predicts that the last-bar direction continues. Classic trend-following baseline."""

    name = "momentum"

    def predict(self, close_history: list[float], horizons: list[int]) -> dict[int, dict]:
        if len(close_history) < 2:
            return {h: {"direction": "flat", "magnitude": 0.0, "confidence": 0.3} for h in horizons}

        last_return = (close_history[-1] - close_history[-2]) / close_history[-2]
        direction = self._classify(last_return)

        return {
            h: {
                "direction": direction,
                "magnitude": abs(last_return),
                "confidence": min(0.8, 0.3 + abs(last_return) * 20),
            }
            for h in horizons
        }


class MACrossoverPredictor(BasePredictor):
    """SMA crossover: short MA > long MA → 'up'. Classical technical signal."""

    name = "ma_crossover"

    def __init__(self, short_window: int = 4, long_window: int = 12):
        self.short_window = short_window
        self.long_window = long_window

    def predict(self, close_history: list[float], horizons: list[int]) -> dict[int, dict]:
        if len(close_history) < self.long_window:
            return {h: {"direction": "flat", "magnitude": 0.0, "confidence": 0.3} for h in horizons}

        prices = np.array(close_history)
        short_ma = prices[-self.short_window:].mean()
        long_ma = prices[-self.long_window:].mean()

        spread = (short_ma - long_ma) / long_ma
        direction = self._classify(spread, flat_threshold=0.002)
        confidence = min(0.9, 0.4 + abs(spread) * 50)

        return {
            h: {
                "direction": direction,
                "magnitude": abs(spread),
                "confidence": confidence,
            }
            for h in horizons
        }


# Registry of all simple predictors
SIMPLE_PREDICTORS = {
    "random": RandomPredictor,
    "buy_and_hold": BuyAndHoldPredictor,
    "momentum": MomentumPredictor,
    "ma_crossover": MACrossoverPredictor,
}
