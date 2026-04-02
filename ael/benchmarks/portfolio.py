"""Portfolio allocation benchmark for AEL/EAEL.

Instead of predicting direction per ticker, the agent allocates portfolio weights
across all tickers simultaneously. This is fundamentally harder because:
- Cross-ticker correlation reasoning required
- Sector allocation and risk management
- Sequential consequences (bad allocation today = capital loss tomorrow)
- Regime adaptation (allocation that worked last week may fail this week)

Each bar = 1 episode = 1 LLM call (vs 10 in per-ticker mode).
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ael.benchmarks.finance import get_realized_returns
from ael.config import ExperimentConfig

logger = logging.getLogger(__name__)


@dataclass
class PortfolioState:
    """Tracks the portfolio across bars."""
    weights: dict[str, float] = field(default_factory=dict)
    cash: float = 1.0  # fraction in cash
    total_value: float = 1.0  # normalized to 1.0 at start
    history: list[dict] = field(default_factory=list)  # per-bar records
    peak_value: float = 1.0
    cost_per_trade: float = 0.0  # transaction cost in fraction (e.g. 0.0005 = 5bp)

    @property
    def drawdown(self) -> float:
        """Current drawdown from peak."""
        if self.peak_value <= 0:
            return 0
        return (self.peak_value - self.total_value) / self.peak_value

    def update(self, returns: dict[str, float]) -> float:
        """Update portfolio value based on realized returns.

        Args:
            returns: ticker → return for this bar (e.g., {"AAPL": 0.005, "NVDA": -0.02})

        Returns:
            Portfolio return for this bar.
        """
        port_return = 0.0
        for ticker, weight in self.weights.items():
            if ticker in returns:
                port_return += weight * returns[ticker]
        # Cash earns nothing
        port_return += self.cash * 0.0

        self.total_value *= (1 + port_return)
        self.peak_value = max(self.peak_value, self.total_value)

        self.history.append({
            "weights": dict(self.weights),
            "cash": self.cash,
            "port_return": port_return,
            "total_value": self.total_value,
            "drawdown": self.drawdown,
        })

        return port_return

    def set_weights(self, new_weights: dict[str, float]) -> float:
        """Set new portfolio weights. Returns turnover (sum of abs weight changes).

        Weights should sum to <= 1.0. Remainder goes to cash.
        """
        turnover = 0.0
        all_tickers = set(list(self.weights.keys()) + list(new_weights.keys()))
        for ticker in all_tickers:
            old = self.weights.get(ticker, 0)
            new = new_weights.get(ticker, 0)
            turnover += abs(new - old)

        self.weights = {t: w for t, w in new_weights.items() if w > 0.001}
        self.cash = max(0, 1.0 - sum(self.weights.values()))

        # Deduct transaction costs from portfolio value
        if self.cost_per_trade > 0 and turnover > 0:
            cost = turnover * self.cost_per_trade
            self.total_value *= (1 - cost)

        return turnover

    def summary_text(self) -> str:
        """Human-readable portfolio summary for the LLM prompt."""
        lines = []
        for ticker, weight in sorted(self.weights.items(), key=lambda x: -x[1]):
            # Unrealized P&L from last bar
            if len(self.history) >= 1:
                last_return = self.history[-1].get("port_return", 0)
                lines.append(f"  {ticker}: {weight:.0%}")
            else:
                lines.append(f"  {ticker}: {weight:.0%}")
        if self.cash > 0.01:
            lines.append(f"  cash: {self.cash:.0%}")
        lines.append(f"  Total value: {self.total_value:.4f} (drawdown: {self.drawdown:.1%})")
        if self.history:
            recent_returns = [h["port_return"] for h in self.history[-4:]]
            lines.append(f"  Recent returns: {', '.join(f'{r:+.2%}' for r in recent_returns)}")
        return "\n".join(lines)


def build_portfolio_prompt(
    bar_time: str,
    tickers: list[str],
    tool_outputs: dict[str, dict[str, Any]],
    portfolio: PortfolioState,
    memory_context: list[dict] | None = None,
    regime_context: dict | None = None,
    strategy_context: dict | None = None,
    yesterday_insight: dict | None = None,
) -> str:
    """Build the portfolio allocation prompt with all tickers' signals.

    Args:
        bar_time: Current bar timestamp.
        tickers: All tickers in the portfolio universe.
        tool_outputs: ticker → {tool_name → output} for all tickers.
        portfolio: Current portfolio state.
        memory_context: Retrieved memory entries.
        regime_context: Market regime information.
        strategy_context: Bandit-selected planner strategy guidance.

    Returns:
        Prompt string for the LLM.
    """
    sections = []
    sections.append(f"# Portfolio Allocation @ {bar_time}\n")

    # Current portfolio state
    sections.append("## Current Portfolio")
    sections.append(portfolio.summary_text())
    sections.append("")

    # Market regime context
    if regime_context:
        regime = regime_context.get("regime", "unknown").upper()
        avg_ret = regime_context.get("avg_return", 0)
        sections.append(f"## Market Regime: {regime}")
        sections.append(f"Recent avg return: {avg_ret:+.3%}")
        if regime_context.get("regime") == "bearish":
            sections.append("Note: Consider defensive allocation and higher cash position.")
        elif regime_context.get("regime") == "bullish":
            sections.append("Note: Momentum-following strategies may be effective.")
        sections.append("")

    # Strategy guidance from bandit-selected planner
    if strategy_context:
        planner = strategy_context.get("planner", "adaptive")
        guidance = strategy_context.get("guidance", "")
        sections.append(f"## Selected Strategy: {planner}")
        if guidance:
            sections.append(guidance)
        sections.append("")

    # Yesterday's reflection insight (injected directly, not via memory)
    # Use empirical_accuracy (tracked externally) if available,
    # otherwise fall back to LLM self-assessed confidence with hedging
    if yesterday_insight:
        insight = yesterday_insight.get("key_insight", "")
        regime = yesterday_insight.get("market_regime", "")
        confidence = yesterday_insight.get("pattern_confidence", 0)
        empirical_acc = yesterday_insight.get("_empirical_accuracy")
        # Use empirical accuracy when available; fall back to self-assessed
        inject_confidence = empirical_acc if empirical_acc is not None else confidence
        if insight and inject_confidence > 0.4:
            sections.append("## Yesterday's Market Analysis (one of several inputs)")
            sections.append(f"Regime: {regime} (confidence: {inject_confidence:.0%})")
            sections.append(f"Insight: {insight}")
            sections.append("")

    # All tickers' signals (compact)
    sections.append("## Market Signals\n")
    _SIGNAL_PREFIXES = ("compute_", "score_", "run_dcf")

    for ticker in tickers:
        ticker_tools = tool_outputs.get(ticker, {})
        sig_parts = []
        for tool_name, output in ticker_tools.items():
            if not any(tool_name.startswith(p) for p in _SIGNAL_PREFIXES):
                continue
            if isinstance(output, dict):
                # Extract key values only
                label = tool_name.replace("compute_", "").replace("score_", "").replace("run_", "")
                key_vals = []
                for k, v in output.items():
                    if isinstance(v, float):
                        key_vals.append(f"{k}={v:.2f}")
                    elif isinstance(v, dict):
                        # One level: extract the most important sub-key
                        for kk, vv in list(v.items())[:2]:
                            if isinstance(vv, (int, float)):
                                key_vals.append(f"{kk}={vv:.2f}" if isinstance(vv, float) else f"{kk}={vv}")
                            elif isinstance(vv, str):
                                key_vals.append(f"{kk}={vv}")
                    elif isinstance(v, str):
                        key_vals.append(f"{k}={v}")
                if key_vals:
                    sig_parts.append(f"{label}({', '.join(key_vals[:4])})")

        sections.append(f"**{ticker}**: {'; '.join(sig_parts) if sig_parts else 'no signals'}")
    sections.append("")

    # Memory context: rules + situations
    if memory_context:
        rules = [m for m in memory_context if m.get("tier") == "procedural"]
        context = [m for m in memory_context if m.get("tier") in ("semantic", "episodic")]

        if rules:
            sections.append("## Learned Rules\n")
            for i, mem in enumerate(rules, 1):
                sections.append(f"{i}. {mem.get('content', '')}")
            sections.append("")

        if context:
            sections.append("## Recent Allocation Lessons\n")
            for mem in context:
                sections.append(f"- {mem.get('content', '')}")
            sections.append("")

    # Allocation history (last 3 bars)
    if portfolio.history:
        sections.append("## Recent Allocation Results\n")
        for h in portfolio.history[-3:]:
            w = h.get("weights", {})
            top3 = sorted(w.items(), key=lambda x: -x[1])[:3]
            top3_str = ", ".join(f"{t}={v:.0%}" for t, v in top3)
            ret = h.get("port_return", 0)
            sections.append(f"- [{top3_str}] → return: {ret:+.2%}")
        sections.append("")

    # Instructions
    tickers_str = '", "'.join(tickers)
    sections.append(
        "## Task\n"
        "Allocate portfolio weights across all tickers + cash.\n"
        "Weights must sum to 1.0. Consider:\n"
        "- Cross-ticker correlations (don't overweight correlated stocks)\n"
        "- Sector diversification\n"
        "- Risk management (avoid >30% in one sector)\n"
        "- Recent performance (cut losers, ride winners — but don't chase)\n\n"
        "Respond in JSON:\n"
        "```json\n"
        "{\n"
        f'  "{tickers_str}": <weight 0.0-1.0>,\n'
        '  "cash": <weight 0.0-1.0>,\n'
        '  "reasoning": "brief explanation"\n'
        "}\n```"
    )

    return "\n".join(sections)


def parse_portfolio_response(
    response: str,
    tickers: list[str],
) -> dict[str, float]:
    """Parse LLM response into portfolio weights.

    Returns dict of ticker → weight. Falls back to equal-weight on failure.
    """
    import re

    # Try to extract JSON
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            return _equal_weight(tickers)

    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError:
        return _equal_weight(tickers)

    # Extract weights
    weights = {}
    for ticker in tickers:
        w = parsed.get(ticker, 0)
        if isinstance(w, (int, float)) and 0 <= w <= 1:
            weights[ticker] = float(w)
        else:
            weights[ticker] = 1.0 / len(tickers)

    # Normalize to sum <= 1.0
    total = sum(weights.values()) + parsed.get("cash", 0)
    if total > 1.5 or total < 0.5:
        return _equal_weight(tickers)
    if total > 1.0:
        scale = 1.0 / total
        weights = {t: w * scale for t, w in weights.items()}

    return weights


def _equal_weight(tickers: list[str]) -> dict[str, float]:
    """Equal-weight fallback allocation."""
    w = 1.0 / len(tickers)
    return {t: w for t in tickers}


def compute_portfolio_metrics(history: list[dict]) -> dict:
    """Compute portfolio performance metrics from history."""
    if not history:
        return {}

    returns = [h["port_return"] for h in history]
    values = [h["total_value"] for h in history]

    total_return = values[-1] / values[0] - 1 if values[0] > 0 else 0
    avg_return = np.mean(returns)
    vol = np.std(returns) if len(returns) > 1 else 0

    # Annualized (4 bars/day, 252 days/year)
    ann_factor = 252 * 4
    ann_return = avg_return * ann_factor
    ann_vol = vol * np.sqrt(ann_factor) if vol > 0 else 0
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0

    # Max drawdown
    drawdowns = [h.get("drawdown", 0) for h in history]
    max_dd = max(drawdowns) if drawdowns else 0

    # Turnover (if tracked)
    turnovers = [h.get("turnover", 0) for h in history if "turnover" in h]
    avg_turnover = np.mean(turnovers) if turnovers else 0

    return {
        "total_return": float(total_return),
        "avg_bar_return": float(avg_return),
        "volatility": float(vol),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(max_dd),
        "avg_turnover": float(avg_turnover),
        "n_bars": len(history),
        "final_value": float(values[-1]) if values else 1.0,
    }


def cost_adjusted_sharpe(history: list[dict], cost_bps: float) -> float:
    """Compute Sharpe after retroactively deducting transaction costs.

    Applies costs to each bar's return based on that bar's turnover, then
    recomputes the annualized Sharpe ratio.

    Args:
        history: portfolio history dicts, each with 'port_return' and 'turnover' keys.
        cost_bps: cost per trade in basis points (e.g., 5 = 5bp = 0.05%).

    Returns:
        Cost-adjusted annualized Sharpe ratio.
    """
    if not history:
        return 0.0

    cost_frac = cost_bps / 10_000.0  # convert bps to fraction
    adjusted_returns = []
    for h in history:
        ret = h["port_return"]
        turnover = h.get("turnover", 0.0)
        adjusted_returns.append(ret - turnover * cost_frac)

    arr = np.array(adjusted_returns)
    avg_return = arr.mean()
    vol = arr.std() if len(arr) > 1 else 0.0

    ann_factor = 252 * 4  # 4 bars/day, 252 days/year
    ann_return = avg_return * ann_factor
    ann_vol = vol * np.sqrt(ann_factor) if vol > 0 else 0.0
    return float(ann_return / ann_vol) if ann_vol > 0 else 0.0
