"""MCP-style finance tools for AEL framework.

Each tool reads from the pre-collected data cache first (data/ directory),
falling back to live API only if cache misses. Tools are designed as
independent, composable units that the agent can select among.

MCP Tool Catalog
================
Category 1 — Data Retrieval (read cached raw data):
  get_price_history       Read OHLCV daily prices from cache
  get_fundamentals        Read financial statements + key ratios
  get_analyst_data        Read analyst targets, recommendations, upgrades
  get_options_data        Read options chain summaries (IV, put/call)
  get_earnings_data       Read quarterly earnings + revenue

Category 2 — Computation (derived from raw data):
  compute_technicals      RSI, MACD, Bollinger, SMA from price history
  compute_quant_risk      GARCH vol, VaR, Sharpe, factor exposure
  compute_momentum        Multi-horizon momentum + trend strength
  compute_correlations    Cross-ticker correlation matrix

Category 3 — Analysis Models:
  run_dcf_model           DCF valuation (bull/base/bear scenarios)
  score_risk              Risk scoring across 5 categories (1-10)
  score_composite_signal  Weighted composite of all signals
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ael.tools.base import BaseTool

# Data cache directory (set via env var or default)
DATA_DIR = Path(
    os.environ.get(
        "AEL_DATA_DIR",
        Path(__file__).parent.parent.parent / "data",
    )
)

# Active frequency — set by the backtest runner before episodes start
_ACTIVE_FREQUENCY: str = "1d"  # "1d" or "1h"
_USE_PRICE_CACHE: bool = False
_PRICE_CACHE: dict[tuple[str, str], pd.DataFrame] = {}


def set_frequency(freq: str, use_cache: bool = False) -> None:
    """Set the active data frequency. Called by backtest runner at startup.

    Args:
        freq: "1d" or "1h".
        use_cache: If True, cache full CSVs in memory to avoid repeated
            disk reads (~25k reads → ~20 reads). Safe because price data
            is static during a backtest run.
    """
    global _ACTIVE_FREQUENCY, _USE_PRICE_CACHE
    _ACTIVE_FREQUENCY = freq
    _USE_PRICE_CACHE = use_cache
    if not use_cache:
        _PRICE_CACHE.clear()


def _prices_dir() -> Path:
    """Return the correct prices directory based on active frequency."""
    if _ACTIVE_FREQUENCY == "1h":
        return DATA_DIR / "prices_1h"
    return DATA_DIR / "prices"


def _read_prices(ticker: str, as_of_datetime: str | None = None) -> pd.DataFrame:
    """Read price data, optionally filtered to <= as_of_datetime.

    This is the central data access function. All price-based tools should
    call this instead of reading CSV directly.

    When use_cache=True (set via set_frequency), the full CSV is read once
    and cached in memory. Otherwise, reads from disk every call.
    """
    cache_key = (ticker, _ACTIVE_FREQUENCY)
    if _USE_PRICE_CACHE and cache_key in _PRICE_CACHE:
        df = _PRICE_CACHE[cache_key]
    else:
        path = _prices_dir() / f"{ticker}.csv"
        if not path.exists():
            raise FileNotFoundError(f"No price data for {ticker} at {path}")
        df = pd.read_csv(path, index_col=0)
        if _USE_PRICE_CACHE:
            _PRICE_CACHE[cache_key] = df

    # Filter to data available at prediction time
    if as_of_datetime is not None:
        df = df[df.index <= as_of_datetime]

    if df.empty:
        raise ValueError(f"No price data for {ticker} at or before {as_of_datetime}")

    return df


def _annualization_factor() -> float:
    """Return annualization factor based on frequency."""
    if _ACTIVE_FREQUENCY == "1h":
        return 252 * 4  # 4 bars/day × 252 trading days
    return 252

# usstock_analysis path for fallback
_USSTOCK_PATH = os.environ.get("USSTOCK_PATH", "")
if _USSTOCK_PATH and str(_USSTOCK_PATH) not in sys.path:
    sys.path.insert(0, str(_USSTOCK_PATH))


# =====================================================================
# Category 1: Data Retrieval Tools
# =====================================================================

class GetPriceHistory(BaseTool):
    """Read OHLCV prices from cache (daily or hourly based on active frequency)."""

    name = "get_price_history"
    description = "Read Open/High/Low/Close/Volume price data from cache."
    cost_per_call = 0.0

    def execute(self, ticker: str, as_of_datetime: str | None = None, **kwargs) -> Any:
        df = _read_prices(ticker, as_of_datetime)
        return {
            "ticker": ticker,
            "rows": len(df),
            "start": str(df.index[0]),
            "end": str(df.index[-1]),
            "latest_close": float(df["Close"].iloc[-1]),
            "prices": df["Close"].tolist()[-50:],  # last 50 bars max
            "volumes": df["Volume"].tolist()[-50:],
            "dates": list(df.index)[-50:],
            "high": df["High"].tolist()[-50:],
            "low": df["Low"].tolist()[-50:],
            "open": df["Open"].tolist()[-50:],
        }


class GetFundamentals(BaseTool):
    """Read financial statements and key ratios from cache."""

    name = "get_fundamentals"
    description = "Financial statements (income, balance sheet, cash flow) + 36 key ratios."
    cost_per_call = 0.0

    def execute(self, ticker: str, **kwargs) -> Any:
        path = DATA_DIR / "fundamentals" / f"{ticker}.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)
        # Fallback
        from lib.data_fetcher import fetch_stock_data
        return fetch_stock_data(ticker)


class GetAnalystData(BaseTool):
    """Read analyst targets, recommendations, and rating changes."""

    name = "get_analyst_data"
    description = "Analyst target prices, consensus recommendation, upgrades/downgrades history."
    cost_per_call = 0.0

    def execute(self, ticker: str, **kwargs) -> Any:
        path = DATA_DIR / "analyst" / f"{ticker}.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)
        # Fallback
        from lib.analyst_tracker import merge_analyst_activity
        return merge_analyst_activity(ticker)


class GetOptionsData(BaseTool):
    """Read options chain summaries: IV, volumes, put/call ratio."""

    name = "get_options_data"
    description = "Options chain: implied volatility, put/call ratio, open interest."
    cost_per_call = 0.0

    def execute(self, ticker: str, **kwargs) -> Any:
        path = DATA_DIR / "options" / f"{ticker}.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)
        # Fallback
        from lib.options_analyzer import fetch_options_summary
        return fetch_options_summary(ticker)


class GetEarningsData(BaseTool):
    """Read quarterly earnings, revenue, and earnings calendar."""

    name = "get_earnings_data"
    description = "Quarterly income, balance sheet, cash flow, and earnings dates."
    cost_per_call = 0.0

    def execute(self, ticker: str, **kwargs) -> Any:
        path = DATA_DIR / "earnings" / f"{ticker}.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return {"error": f"No earnings data for {ticker}"}


# =====================================================================
# Category 2: Computation Tools
# =====================================================================

class ComputeTechnicals(BaseTool):
    """Compute technical indicators from cached price data."""

    name = "compute_technicals"
    description = "RSI(14), MACD(12,26,9), Bollinger(20,2), SMA(50,200), support/resistance."
    cost_per_call = 0.0

    def execute(self, ticker: str, as_of_datetime: str | None = None, **kwargs) -> Any:
        df = _read_prices(ticker, as_of_datetime)
        close = df["Close"]
        high = df["High"]
        low = df["Low"]

        result = {"ticker": ticker}

        # RSI(14 bars) — same period count, adapts to frequency automatically
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        rsi_val = float(rsi.iloc[-1]) if not rsi.empty else 50
        result["rsi_14"] = {
            "value": rsi_val,
            "signal": "OVERBOUGHT" if rsi_val > 70 else ("OVERSOLD" if rsi_val < 30 else "NEUTRAL"),
        }

        # MACD(12, 26, 9)
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9).mean()
        histogram = macd_line - signal_line
        macd_val = float(macd_line.iloc[-1])
        sig_val = float(signal_line.iloc[-1])
        result["macd"] = {
            "macd": macd_val,
            "signal": sig_val,
            "histogram": float(histogram.iloc[-1]),
            "trend": "BULLISH" if macd_val > sig_val else "BEARISH",
        }

        # Bollinger Bands(20, 2)
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        upper = sma20 + 2 * std20
        lower = sma20 - 2 * std20
        latest = float(close.iloc[-1])
        result["bollinger"] = {
            "upper": float(upper.iloc[-1]),
            "middle": float(sma20.iloc[-1]),
            "lower": float(lower.iloc[-1]),
            "position": (
                "ABOVE_UPPER" if latest > float(upper.iloc[-1])
                else ("BELOW_LOWER" if latest < float(lower.iloc[-1]) else "WITHIN_BANDS")
            ),
        }

        # Moving averages
        sma50 = close.rolling(50).mean()
        sma200 = close.rolling(200).mean() if len(close) >= 200 else close.rolling(len(close)).mean()
        result["moving_averages"] = {
            "sma_50": float(sma50.iloc[-1]) if not sma50.isna().iloc[-1] else None,
            "sma_200": float(sma200.iloc[-1]) if not sma200.isna().iloc[-1] else None,
            "above_50": latest > float(sma50.iloc[-1]) if not sma50.isna().iloc[-1] else None,
            "above_200": latest > float(sma200.iloc[-1]) if not sma200.isna().iloc[-1] else None,
        }

        # Support/resistance
        result["support_resistance"] = {
            "52w_high": float(high.max()),
            "52w_low": float(low.min()),
            "pct_from_high": float((latest - high.max()) / high.max()),
            "pct_from_low": float((latest - low.min()) / low.min()),
        }

        # Composite score (0-5)
        bullish_signals = sum([
            rsi_val < 70,
            macd_val > sig_val,
            latest > float(sma50.iloc[-1]) if not sma50.isna().iloc[-1] else False,
            latest > float(sma200.iloc[-1]) if not sma200.isna().iloc[-1] else False,
            latest > float(lower.iloc[-1]),
        ])
        result["technical_score"] = bullish_signals
        result["technical_outlook"] = (
            "STRONG_BUY" if bullish_signals >= 5
            else "BUY" if bullish_signals >= 4
            else "NEUTRAL" if bullish_signals >= 3
            else "SELL" if bullish_signals >= 2
            else "STRONG_SELL"
        )

        return result


class ComputeQuantRisk(BaseTool):
    """Compute quantitative risk metrics from cached price data."""

    name = "compute_quant_risk"
    description = "Volatility (realized + GARCH-style), VaR/CVaR, Sharpe/Sortino, beta."
    cost_per_call = 0.0

    def execute(self, ticker: str, as_of_datetime: str | None = None, **kwargs) -> Any:
        df = _read_prices(ticker, as_of_datetime)
        spy_path = _prices_dir() / "SPY.csv"

        close = df["Close"]
        returns = close.pct_change().dropna()
        ann_factor = _annualization_factor()

        result = {"ticker": ticker}

        # Realized volatility
        vol_30 = float(returns.tail(30).std() * np.sqrt(ann_factor))
        vol_full = float(returns.std() * np.sqrt(ann_factor))
        result["volatility"] = {
            "realized_30bar_annualized": vol_30,
            "realized_full_annualized": vol_full,
            "trend": "INCREASING" if vol_30 > vol_full else "DECREASING",
        }

        # VaR and CVaR (95%)
        var_95 = float(np.percentile(returns, 5))
        cvar_95 = float(returns[returns <= var_95].mean()) if len(returns[returns <= var_95]) > 0 else var_95
        result["var"] = {
            "var_95": var_95,
            "cvar_95": cvar_95,
        }

        # Risk-return metrics
        ann_return = float(returns.mean() * ann_factor)
        ann_vol = float(returns.std() * np.sqrt(ann_factor))
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0
        downside = returns[returns < 0]
        downside_vol = float(downside.std() * np.sqrt(ann_factor)) if len(downside) > 0 else ann_vol
        sortino = ann_return / downside_vol if downside_vol > 0 else 0

        # Max drawdown
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns - running_max) / running_max
        max_dd = float(drawdown.min())

        result["risk_return"] = {
            "annual_return": ann_return,
            "annual_volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_dd,
            "calmar_ratio": ann_return / abs(max_dd) if max_dd != 0 else 0,
        }

        # Beta vs SPY
        if spy_path.exists():
            try:
                spy_df = _read_prices("SPY", as_of_datetime=as_of_datetime)
            except (FileNotFoundError, ValueError):
                spy_df = pd.read_csv(spy_path, index_col=0, parse_dates=True)
            spy_returns = spy_df["Close"].pct_change().dropna()
            # Align dates
            common = returns.index.intersection(spy_returns.index)
            if len(common) > 20:
                r = returns.loc[common]
                m = spy_returns.loc[common]
                cov = np.cov(r, m)
                beta = float(cov[0, 1] / cov[1, 1]) if cov[1, 1] != 0 else 1.0
                alpha = float(r.mean() - beta * m.mean()) * 252
                result["factor"] = {"beta": beta, "alpha": alpha}

        return result


class ComputeMomentum(BaseTool):
    """Compute multi-horizon momentum and trend strength."""

    name = "compute_momentum"
    description = "Returns over 5/10/20/60 days, trend strength, volume trend."
    cost_per_call = 0.0

    def execute(self, ticker: str, as_of_datetime: str | None = None, **kwargs) -> Any:
        df = _read_prices(ticker, as_of_datetime)
        close = df["Close"]
        volume = df["Volume"]

        latest = float(close.iloc[-1])
        result = {"ticker": ticker, "latest_price": latest}

        # Multi-horizon returns (bar-based, adapts to frequency)
        # Daily: [5, 10, 20, 60] days; Hourly: [4, 8, 20, 60] bars
        horizons = [4, 8, 20, 60] if _ACTIVE_FREQUENCY == "1h" else [5, 10, 20, 60]
        for horizon in horizons:
            if len(close) > horizon:
                ret = float((close.iloc[-1] / close.iloc[-horizon] - 1))
                result[f"return_{horizon}bar"] = ret
            else:
                result[f"return_{horizon}bar"] = None

        # Trend strength: slope of linear regression over last 20 bars
        if len(close) >= 20:
            y = close.tail(20).values
            x = np.arange(len(y))
            slope, intercept = np.polyfit(x, y, 1)
            r_squared = 1 - np.sum((y - (slope * x + intercept))**2) / np.sum((y - y.mean())**2)
            result["trend"] = {
                "slope_per_bar": float(slope),
                "r_squared": float(r_squared),
                "direction": "UP" if slope > 0 else "DOWN",
                "strength": "STRONG" if r_squared > 0.7 else ("MODERATE" if r_squared > 0.4 else "WEAK"),
            }

        # Volume trend
        if len(volume) >= 20:
            vol_20 = float(volume.tail(20).mean())
            vol_5 = float(volume.tail(5).mean())
            result["volume_trend"] = {
                "avg_20bar": vol_20,
                "avg_5bar": vol_5,
                "ratio": vol_5 / vol_20 if vol_20 > 0 else 1.0,
                "signal": "INCREASING" if vol_5 > vol_20 * 1.2 else (
                    "DECREASING" if vol_5 < vol_20 * 0.8 else "STABLE"
                ),
            }

        return result


class ComputeCorrelations(BaseTool):
    """Compute cross-ticker correlation matrix from cached prices."""

    name = "compute_correlations"
    description = "Pairwise correlation matrix across all cached tickers (30-day rolling)."
    cost_per_call = 0.0

    def execute(self, ticker: str, as_of_datetime: str | None = None, **kwargs) -> Any:
        pdir = _prices_dir()
        all_files = sorted(pdir.glob("*.csv"))

        # Load all tickers (use _read_prices for cache benefit)
        all_closes = {}
        for f in all_files:
            t = f.stem
            if t == "SPY":
                continue
            try:
                df = _read_prices(t, as_of_datetime=as_of_datetime)
            except (FileNotFoundError, ValueError):
                continue
            all_closes[t] = df["Close"]

        if len(all_closes) < 2:
            return {"error": "Need at least 2 tickers for correlation"}

        combined = pd.DataFrame(all_closes).dropna()
        returns = combined.pct_change().dropna()

        # Full-period correlation
        corr = returns.corr()
        # Rolling correlation (20 bars) with the target ticker
        window = 20
        rolling_corr = {}
        if ticker in returns.columns:
            target = returns[ticker]
            for other in returns.columns:
                if other != ticker:
                    rc = target.rolling(window).corr(returns[other])
                    rolling_corr[other] = float(rc.iloc[-1]) if not rc.isna().iloc[-1] else None

        return {
            "ticker": ticker,
            "full_period_correlation": {
                k: {k2: float(v2) for k2, v2 in corr[k].items()}
                for k in corr.columns
            },
            "rolling_correlation_vs_target": rolling_corr,
            "most_correlated": max(rolling_corr, key=lambda x: abs(rolling_corr[x] or 0)) if rolling_corr else None,
            "least_correlated": min(rolling_corr, key=lambda x: abs(rolling_corr[x] or 0)) if rolling_corr else None,
        }


# =====================================================================
# Category 3: Analysis Model Tools
# =====================================================================

class RunDCFModel(BaseTool):
    """Run DCF valuation model from cached fundamentals."""

    name = "run_dcf_model"
    description = "Discounted cash flow: bull/base/bear scenarios → probability-weighted target price."
    cost_per_call = 0.0

    def execute(self, ticker: str, **kwargs) -> Any:
        # Try cached fundamentals first, then pass to usstock DCF
        fund_path = DATA_DIR / "fundamentals" / f"{ticker}.json"
        if fund_path.exists():
            with open(fund_path) as f:
                fundamentals = json.load(f)
            # Build raw_data dict compatible with run_dcf
            info = fundamentals.get("info", {})
            raw_data = {
                "price_data": {
                    "current_price": info.get("fiftyDayAverage", 100),
                },
                "fundamentals": {
                    "pe_ratio": info.get("trailingPE"),
                    "peg_ratio": info.get("pegRatio"),
                    "ev_to_ebitda": info.get("enterpriseToEbitda"),
                },
                "margins": {
                    "gross_margin": info.get("grossMargins"),
                    "operating_margin": info.get("operatingMargins"),
                    "net_margin": info.get("profitMargins"),
                },
                "growth": {
                    "revenue_growth": info.get("revenueGrowth"),
                    "earnings_growth": info.get("earningsGrowth"),
                },
                "financial_health": {
                    "total_cash": info.get("totalCash"),
                    "total_debt": info.get("totalDebt"),
                    "free_cash_flow": info.get("freeCashflow"),
                    "current_ratio": info.get("currentRatio"),
                },
                "analyst_consensus": {
                    "target_mean": info.get("fiftyDayAverage"),
                },
            }
            try:
                from lib.analysis.dcf import run_dcf
                return run_dcf(raw_data)
            except Exception:
                pass

        # Simplified DCF when usstock not available
        return self._simple_dcf(ticker)

    def _simple_dcf(self, ticker: str) -> dict:
        """Simplified DCF using cached data only."""
        fund_path = DATA_DIR / "fundamentals" / f"{ticker}.json"
        if not fund_path.exists():
            return {"error": f"No fundamentals for {ticker}"}

        with open(fund_path) as f:
            info = json.load(f).get("info", {})

        fcf = info.get("freeCashflow", 0)
        mc = info.get("marketCap", 1)
        growth = info.get("revenueGrowth", 0.05) or 0.05
        pe = info.get("trailingPE", 20) or 20

        if fcf and mc:
            fcf_yield = fcf / mc
            # Simple Gordon growth: fair_value = FCF * (1+g) / (r - g)
            r = 0.10  # discount rate
            g = min(growth, 0.08)  # cap perpetual growth
            if r > g:
                fair_multiple = (1 + g) / (r - g)
                implied_upside = (fcf_yield * fair_multiple - 1)
            else:
                implied_upside = 0
        else:
            implied_upside = 0

        return {
            "ticker": ticker,
            "fcf": fcf,
            "market_cap": mc,
            "growth_rate": growth,
            "pe_ratio": pe,
            "implied_upside": implied_upside,
            "signal": "UNDERVALUED" if implied_upside > 0.15 else (
                "OVERVALUED" if implied_upside < -0.15 else "FAIR"
            ),
        }


class ScoreRisk(BaseTool):
    """Score overall risk from fundamentals (1-10 scale)."""

    name = "score_risk"
    description = "Risk score across valuation, financial health, growth, macro, technical (1-10)."
    cost_per_call = 0.0

    def execute(self, ticker: str, **kwargs) -> Any:
        fund_path = DATA_DIR / "fundamentals" / f"{ticker}.json"
        if not fund_path.exists():
            return {"error": f"No fundamentals for {ticker}", "overall": 5}

        with open(fund_path) as f:
            info = json.load(f).get("info", {})

        scores = {}

        # Valuation risk
        pe = info.get("trailingPE") or 20
        scores["valuation"] = min(10, max(1, pe / 5))

        # Financial health
        cr = info.get("currentRatio") or 1.5
        de = info.get("debtToEquity") or 50
        scores["financial"] = min(10, max(1, 5 - (cr - 1) * 2 + de / 50))

        # Growth
        rg = info.get("revenueGrowth") or 0.05
        scores["growth"] = min(10, max(1, 5 - rg * 20))

        # Macro (beta)
        beta = info.get("beta") or 1.0
        scores["macro"] = min(10, max(1, beta * 5))

        # Technical (use price position vs 52w range)
        hi = info.get("fiftyTwoWeekHigh") or 100
        lo = info.get("fiftyTwoWeekLow") or 50
        avg = info.get("fiftyDayAverage") or (hi + lo) / 2
        pct = (avg - lo) / (hi - lo) if hi != lo else 0.5
        scores["technical"] = min(10, max(1, pct * 10))

        # Weighted overall
        overall = (
            scores["valuation"] * 0.25
            + scores["financial"] * 0.25
            + scores["growth"] * 0.25
            + scores["macro"] * 0.15
            + scores["technical"] * 0.10
        )

        rating = (
            "LOW" if overall <= 3 else
            "MODERATE" if overall <= 5 else
            "HIGH" if overall <= 7 else
            "VERY HIGH"
        )

        return {
            "ticker": ticker,
            "scores": scores,
            "overall": round(overall, 1),
            "rating": rating,
        }


class ScoreCompositeSignal(BaseTool):
    """Compute weighted composite signal from all available data."""

    name = "score_composite_signal"
    description = "Weighted composite of momentum, valuation, risk, technicals → BUY/SELL/HOLD."
    cost_per_call = 0.0

    def execute(self, ticker: str, tool_outputs: dict | None = None, **kwargs) -> Any:
        """Score composite signal. If tool_outputs provided, uses them.
        Otherwise reads from cache directly."""
        if tool_outputs is None:
            tool_outputs = {}

        signals = []
        weights = []

        # Technical score (0-5 → -50 to +50)
        tech = tool_outputs.get("compute_technicals", {})
        if tech:
            tech_score = tech.get("technical_score", 2.5)
            signals.append((tech_score - 2.5) / 2.5 * 50)
            weights.append(0.20)

        # Momentum (multi-horizon return → signal)
        mom = tool_outputs.get("compute_momentum", {})
        if mom:
            ret_20 = mom.get("return_20d", 0) or 0
            signals.append(np.clip(ret_20 * 500, -50, 50))
            weights.append(0.15)

        # DCF valuation signal
        dcf = tool_outputs.get("run_dcf_model", {})
        if dcf:
            upside = dcf.get("implied_upside", 0) or 0
            signals.append(np.clip(upside * 100, -50, 50))
            weights.append(0.20)

        # Risk (inverse — high risk = negative)
        risk = tool_outputs.get("score_risk", {})
        if risk:
            risk_val = risk.get("overall", 5)
            signals.append((5 - risk_val) * 10)
            weights.append(0.10)

        # Analyst consensus
        analyst = tool_outputs.get("get_analyst_data", {})
        if analyst:
            rec = analyst.get("recommendation_mean")
            if rec:
                # yfinance: 1=Strong Buy, 5=Strong Sell
                signals.append((3 - rec) * 25)
                weights.append(0.15)

        # Options sentiment (put/call ratio)
        options = tool_outputs.get("get_options_data", {})
        if options:
            chains = options.get("chains", {})
            pcr_vals = [c.get("put_call_ratio") for c in chains.values() if c.get("put_call_ratio")]
            if pcr_vals:
                avg_pcr = np.mean(pcr_vals)
                signals.append((1.0 - avg_pcr) * 30)  # low PCR = bullish
                weights.append(0.10)

        # Quant risk (Sharpe → signal)
        quant = tool_outputs.get("compute_quant_risk", {})
        if quant:
            sharpe = quant.get("risk_return", {}).get("sharpe_ratio", 0)
            signals.append(np.clip(sharpe * 20, -50, 50))
            weights.append(0.10)

        if not signals:
            return {"ticker": ticker, "composite_score": 0, "signal": "HOLD", "confidence": 0}

        # Normalize weights
        total_w = sum(weights)
        composite = sum(s * w / total_w for s, w in zip(signals, weights))

        signal = (
            "STRONG_BUY" if composite > 40 else
            "BUY" if composite > 15 else
            "HOLD" if composite > -15 else
            "SELL" if composite > -40 else
            "STRONG_SELL"
        )
        confidence = min(0.95, abs(composite) / 50)

        return {
            "ticker": ticker,
            "composite_score": float(composite),
            "signal": signal,
            "confidence": float(confidence),
            "n_signals": len(signals),
        }


# =====================================================================
# Registry
# =====================================================================

ALL_FINANCE_TOOLS: list[type[BaseTool]] = [
    # Category 1: Data retrieval
    GetPriceHistory,
    GetFundamentals,
    GetAnalystData,
    GetOptionsData,
    GetEarningsData,
    # Category 2: Computation
    ComputeTechnicals,
    ComputeQuantRisk,
    ComputeMomentum,
    ComputeCorrelations,
    # Category 3: Analysis
    RunDCFModel,
    ScoreRisk,
    ScoreCompositeSignal,
]


def register_all_finance_tools(registry) -> None:
    """Register all finance tools with a ToolRegistry instance."""
    for tool_cls in ALL_FINANCE_TOOLS:
        registry.register(tool_cls())
