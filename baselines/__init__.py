"""Baseline predictors for stock price direction prediction.

Each baseline implements the same interface:
    predict(close_history: list[float]) -> dict
        Returns {"direction": "up"/"down"/"flat", "magnitude": float, "confidence": float}

Baselines are evaluated against the same pre-cached returns as AEL methods,
ensuring apples-to-apples comparison.

Simple baselines (no LLM):
    - RandomPredictor: uniform random direction
    - BuyAndHoldPredictor: always predicts "up"
    - MomentumPredictor: follows last-bar direction
    - MACrossoverPredictor: SMA(short) vs SMA(long) crossover

Paper baselines (separate venvs):
    - baselines/reflexion.py: Reflexion (Shinn et al., 2023)
    - baselines/expel.py: ExpeL (Zhao et al., 2024)
    - baselines/factorminer.py: FactorMiner-style (2026)
"""
