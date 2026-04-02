"""Bandit algorithms for module selection: Thompson Sampling and LinUCB."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass
class ThompsonSamplingArm:
    """A single arm in a Thompson Sampling bandit."""
    name: str
    alpha: float = 1.0  # Beta prior successes + 1
    beta: float = 1.0   # Beta prior failures + 1
    pull_count: int = 0

    def sample(self) -> float:
        return np.random.beta(self.alpha, self.beta)

    def update(self, reward: float) -> None:
        """Update with reward in [0, 1]."""
        self.pull_count += 1
        # Treat reward as probability of success
        self.alpha += reward
        self.beta += (1.0 - reward)

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)


class ThompsonSampling:
    """Multi-armed Thompson Sampling bandit."""

    def __init__(self, arm_names: list[str]):
        self.arms = {name: ThompsonSamplingArm(name=name) for name in arm_names}

    def select(self, candidates: list[str] | None = None) -> str:
        """Select arm with highest Thompson sample."""
        if candidates is None:
            candidates = list(self.arms.keys())
        samples = {name: self.arms[name].sample() for name in candidates}
        return max(samples, key=samples.get)

    def select_top_k(self, k: int, candidates: list[str] | None = None) -> list[str]:
        """Select top-k arms by Thompson sample."""
        if candidates is None:
            candidates = list(self.arms.keys())
        samples = {name: self.arms[name].sample() for name in candidates}
        ranked = sorted(samples, key=samples.get, reverse=True)
        return ranked[:k]

    def update(self, name: str, reward: float) -> None:
        if name in self.arms:
            self.arms[name].update(reward)

    def add_arm(self, name: str) -> None:
        if name not in self.arms:
            self.arms[name] = ThompsonSamplingArm(name=name)

    def state_dict(self) -> dict:
        return {
            name: {"alpha": arm.alpha, "beta": arm.beta, "pull_count": arm.pull_count}
            for name, arm in self.arms.items()
        }

    def load_state_dict(self, state: dict) -> None:
        for name, data in state.items():
            if name in self.arms:
                self.arms[name].alpha = data["alpha"]
                self.arms[name].beta = data["beta"]
                self.arms[name].pull_count = data["pull_count"]


class LinUCB:
    """Linear Upper Confidence Bound contextual bandit.

    Each arm maintains A (d×d) and b (d×1) for ridge regression.
    UCB = x^T theta + alpha * sqrt(x^T A^{-1} x)
    """

    def __init__(self, arm_names: list[str], dim: int, alpha: float = 1.0):
        self.alpha_0 = alpha
        self.alpha = alpha
        self.dim = dim
        self._total_pulls = 0
        self.arms: dict[str, dict] = {}
        for name in arm_names:
            self.arms[name] = {
                "A": np.eye(dim),
                "b": np.zeros(dim),
                "pull_count": 0,
            }

    def select(self, context: np.ndarray,
               candidates: list[str] | None = None) -> str:
        """Select arm with highest UCB given context vector."""
        if candidates is None:
            candidates = list(self.arms.keys())
        x = np.array(context, dtype=float).flatten()

        best_name = candidates[0]
        best_ucb = -np.inf

        for name in candidates:
            arm = self.arms[name]
            A_inv = np.linalg.inv(arm["A"])
            theta = A_inv @ arm["b"]
            ucb = float(theta @ x + self.alpha * np.sqrt(x @ A_inv @ x))
            if ucb > best_ucb:
                best_ucb = ucb
                best_name = name

        return best_name

    def exploit(self, context: np.ndarray,
                candidates: list[str] | None = None) -> str:
        """Greedy arm selection — highest θ·x with NO exploration bonus."""
        if candidates is None:
            candidates = list(self.arms.keys())
        x = np.array(context, dtype=float).flatten()

        best_name = candidates[0]
        best_val = -np.inf
        for name in candidates:
            arm = self.arms[name]
            A_inv = np.linalg.inv(arm["A"])
            theta = A_inv @ arm["b"]
            val = float(theta @ x)
            if val > best_val:
                best_val = val
                best_name = name
        return best_name

    def update(self, name: str, context: np.ndarray, reward: float) -> None:
        """Update arm parameters after observing reward."""
        x = np.array(context, dtype=float).flatten()
        arm = self.arms[name]
        arm["A"] += np.outer(x, x)
        arm["b"] += reward * x
        arm["pull_count"] += 1
        self._total_pulls += 1
        # Decay alpha: transition from exploration to exploitation
        self.alpha = self.alpha_0 / np.sqrt(1 + self._total_pulls / 10)

    def add_arm(self, name: str) -> None:
        if name not in self.arms:
            self.arms[name] = {
                "A": np.eye(self.dim),
                "b": np.zeros(self.dim),
                "pull_count": 0,
            }

    def state_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "dim": self.dim,
            "arms": {
                name: {
                    "A": arm["A"].tolist(),
                    "b": arm["b"].tolist(),
                    "pull_count": arm["pull_count"],
                }
                for name, arm in self.arms.items()
            },
        }

    def load_state_dict(self, state: dict) -> None:
        self.alpha = state["alpha"]
        self.dim = state["dim"]
        for name, data in state["arms"].items():
            if name in self.arms:
                self.arms[name]["A"] = np.array(data["A"])
                self.arms[name]["b"] = np.array(data["b"])
                self.arms[name]["pull_count"] = data["pull_count"]
