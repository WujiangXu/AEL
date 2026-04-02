"""Download and cache raw market data for AEL backtest experiments.

Collects OHLCV prices, fundamentals, analyst data, options, earnings,
and sector metadata for a set of tickers. Data is stored in data/ as
CSV (prices) and JSON (everything else).

Usage:
    # activate your virtual environment

    # Default (D1 Tech-10):
    python scripts/collect_data.py

    # Custom tickers + date range:
    python scripts/collect_data.py --tickers AAPL NVDA JNJ --start 2024-07-01

    # Preset datasets:
    python scripts/collect_data.py --dataset d2
    python scripts/collect_data.py --dataset d3
    python scripts/collect_data.py --dataset d4
    python scripts/collect_data.py --dataset all   # collect everything
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Default configuration (D1)
DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "NFLX", "AMD", "CRM"]
DEFAULT_BENCHMARK = "SPY"
DEFAULT_START = "2024-10-01"
DEFAULT_END = "2025-03-28"
DATA_DIR = Path(__file__).parent.parent / "data"

# Preset dataset definitions
DATASETS = {
    "d1": {
        "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "NFLX", "AMD", "CRM"],
        "benchmark": "SPY",
        "start": "2024-10-01",
        "end": "2025-03-28",
        "description": "Tech-10 (control)",
    },
    "d2": {
        "tickers": ["AAPL", "NVDA", "JNJ", "UNH", "JPM", "GS", "XOM", "PG", "CAT", "NEE"],
        "benchmark": "SPY",
        "start": "2024-10-01",
        "end": "2025-03-28",
        "description": "Sector-Diverse (7 GICS sectors)",
    },
    "d3": {
        "tickers": ["MSFT", "GOOGL", "PANW", "ABNB", "DKNG", "ROKU", "PATH", "UPST", "BILL", "SOFI"],
        "benchmark": "SPY",
        "start": "2024-10-01",
        "end": "2025-03-28",
        "description": "Market-Cap Diverse (mega to small cap)",
    },
    "d4": {
        "tickers": ["AAPL", "NVDA", "JNJ", "UNH", "JPM", "GS", "XOM", "PG", "CAT", "NEE"],
        "benchmark": "SPY",
        "start": "2024-07-01",
        "end": "2025-03-28",
        "description": "Regime-Shift (extended to Jul 2024)",
    },
}


def safe_json(obj):
    """Convert non-serializable types for JSON output."""
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="index")
    if hasattr(obj, "item"):  # numpy scalar
        return obj.item()
    if pd.isna(obj):
        return None
    return str(obj)


def collect_prices(ticker: str, start_date: str, end_date: str) -> bool:
    """Download OHLCV daily prices."""
    out_path = DATA_DIR / "prices" / f"{ticker}.csv"
    if out_path.exists():
        # Check if existing data covers the requested range
        existing = pd.read_csv(out_path, index_col=0, parse_dates=True)
        if len(existing) > 0:
            existing_start = existing.index.min().strftime("%Y-%m-%d")
            existing_end = existing.index.max().strftime("%Y-%m-%d")
            if existing_start <= start_date and existing_end >= end_date:
                logger.info(f"  [prices] {ticker}: already cached ({existing_start} to {existing_end})")
                return True
            # Need to extend range
            start_date = min(start_date, existing_start)
            end_date = max(end_date, existing_end)
            logger.info(f"  [prices] {ticker}: extending to {start_date}..{end_date}")

    try:
        data = yf.download(
            ticker, start=start_date, end=end_date, progress=False
        )
        if data.empty:
            logger.warning(f"  [prices] {ticker}: no data returned")
            return False
        # Flatten multi-level columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data.to_csv(out_path)
        logger.info(f"  [prices] {ticker}: {len(data)} rows saved")
        return True
    except Exception as e:
        logger.error(f"  [prices] {ticker}: {e}")
        return False


def collect_fundamentals(ticker: str) -> bool:
    """Download fundamental data: financials, balance sheet, cash flow, key stats."""
    out_path = DATA_DIR / "fundamentals" / f"{ticker}.json"
    if out_path.exists():
        logger.info(f"  [fundamentals] {ticker}: already cached")
        return True
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}

        # Financial statements — convert Timestamp columns to string keys
        def df_to_dict_safe(df):
            if df is None or df.empty:
                return {}
            d = {}
            for col in df.columns:
                key = col.isoformat() if hasattr(col, "isoformat") else str(col)
                d[key] = {k: (v.item() if hasattr(v, "item") else v) for k, v in df[col].items()}
            return d

        income_stmt = {}
        balance_sheet = {}
        cash_flow = {}

        try:
            income_stmt = df_to_dict_safe(stock.income_stmt)
        except Exception:
            pass

        try:
            balance_sheet = df_to_dict_safe(stock.balance_sheet)
        except Exception:
            pass

        try:
            cash_flow = df_to_dict_safe(stock.cashflow)
        except Exception:
            pass

        # Key metrics from info
        fundamentals = {
            "info": {
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "marketCap": info.get("marketCap"),
                "enterpriseValue": info.get("enterpriseValue"),
                "beta": info.get("beta"),
                "trailingPE": info.get("trailingPE"),
                "forwardPE": info.get("forwardPE"),
                "priceToBook": info.get("priceToBook"),
                "priceToSalesTrailing12Months": info.get("priceToSalesTrailing12Months"),
                "enterpriseToEbitda": info.get("enterpriseToEbitda"),
                "pegRatio": info.get("pegRatio"),
                "dividendYield": info.get("dividendYield"),
                "payoutRatio": info.get("payoutRatio"),
                "profitMargins": info.get("profitMargins"),
                "grossMargins": info.get("grossMargins"),
                "operatingMargins": info.get("operatingMargins"),
                "revenueGrowth": info.get("revenueGrowth"),
                "earningsGrowth": info.get("earningsGrowth"),
                "returnOnEquity": info.get("returnOnEquity"),
                "returnOnAssets": info.get("returnOnAssets"),
                "debtToEquity": info.get("debtToEquity"),
                "currentRatio": info.get("currentRatio"),
                "quickRatio": info.get("quickRatio"),
                "freeCashflow": info.get("freeCashflow"),
                "totalCash": info.get("totalCash"),
                "totalDebt": info.get("totalDebt"),
                "totalRevenue": info.get("totalRevenue"),
                "ebitda": info.get("ebitda"),
                "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
                "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow"),
                "fiftyDayAverage": info.get("fiftyDayAverage"),
                "twoHundredDayAverage": info.get("twoHundredDayAverage"),
                "averageVolume": info.get("averageVolume"),
                "sharesOutstanding": info.get("sharesOutstanding"),
                "shortRatio": info.get("shortRatio"),
                "shortPercentOfFloat": info.get("shortPercentOfFloat"),
            },
            "income_statement": income_stmt,
            "balance_sheet": balance_sheet,
            "cash_flow": cash_flow,
        }

        with open(out_path, "w") as f:
            json.dump(fundamentals, f, indent=2, default=safe_json)
        logger.info(f"  [fundamentals] {ticker}: saved")
        return True
    except Exception as e:
        logger.error(f"  [fundamentals] {ticker}: {e}")
        return False


def collect_analyst(ticker: str) -> bool:
    """Download analyst recommendations and target prices."""
    out_path = DATA_DIR / "analyst" / f"{ticker}.json"
    if out_path.exists():
        logger.info(f"  [analyst] {ticker}: already cached")
        return True
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}

        analyst_data = {
            "target_mean_price": info.get("targetMeanPrice"),
            "target_median_price": info.get("targetMedianPrice"),
            "target_high_price": info.get("targetHighPrice"),
            "target_low_price": info.get("targetLowPrice"),
            "number_of_analyst_opinions": info.get("numberOfAnalystOpinions"),
            "recommendation_key": info.get("recommendationKey"),
            "recommendation_mean": info.get("recommendationMean"),
        }

        # Recommendations history
        try:
            recs = stock.recommendations
            if recs is not None and not recs.empty:
                analyst_data["recommendations_history"] = recs.reset_index().to_dict(orient="records")
        except Exception:
            analyst_data["recommendations_history"] = []

        # Upgrades/downgrades
        try:
            upgrades = stock.upgrades_downgrades
            if upgrades is not None and not upgrades.empty:
                # Filter to recent (last 6 months)
                cutoff = datetime.now() - timedelta(days=180)
                if hasattr(upgrades.index, 'tz_localize'):
                    recent = upgrades[upgrades.index >= pd.Timestamp(cutoff, tz=upgrades.index.tz)]
                else:
                    recent = upgrades.tail(50)
                analyst_data["upgrades_downgrades"] = recent.reset_index().to_dict(orient="records")
        except Exception:
            analyst_data["upgrades_downgrades"] = []

        with open(out_path, "w") as f:
            json.dump(analyst_data, f, indent=2, default=safe_json)
        logger.info(f"  [analyst] {ticker}: saved")
        return True
    except Exception as e:
        logger.error(f"  [analyst] {ticker}: {e}")
        return False


def collect_options(ticker: str) -> bool:
    """Download options chain summary."""
    out_path = DATA_DIR / "options" / f"{ticker}.json"
    if out_path.exists():
        logger.info(f"  [options] {ticker}: already cached")
        return True
    try:
        stock = yf.Ticker(ticker)
        expirations = stock.options
        if not expirations:
            logger.warning(f"  [options] {ticker}: no options data")
            with open(out_path, "w") as f:
                json.dump({"expiration_dates": [], "chains": {}}, f)
            return True

        options_data = {
            "expiration_dates": list(expirations),
            "chains": {},
        }

        # Collect first 3 expirations
        for exp in expirations[:3]:
            try:
                chain = stock.option_chain(exp)
                calls = chain.calls
                puts = chain.puts

                options_data["chains"][exp] = {
                    "calls_count": len(calls),
                    "puts_count": len(puts),
                    "calls_summary": {
                        "avg_iv": float(calls["impliedVolatility"].mean()) if "impliedVolatility" in calls.columns else None,
                        "total_volume": int(calls["volume"].sum()) if "volume" in calls.columns and not calls["volume"].isna().all() else 0,
                        "total_open_interest": int(calls["openInterest"].sum()) if "openInterest" in calls.columns and not calls["openInterest"].isna().all() else 0,
                    },
                    "puts_summary": {
                        "avg_iv": float(puts["impliedVolatility"].mean()) if "impliedVolatility" in puts.columns else None,
                        "total_volume": int(puts["volume"].sum()) if "volume" in puts.columns and not puts["volume"].isna().all() else 0,
                        "total_open_interest": int(puts["openInterest"].sum()) if "openInterest" in puts.columns and not puts["openInterest"].isna().all() else 0,
                    },
                    "put_call_ratio": (
                        float(puts["volume"].sum() / calls["volume"].sum())
                        if "volume" in calls.columns
                        and calls["volume"].sum() > 0
                        and not calls["volume"].isna().all()
                        else None
                    ),
                }
            except Exception as e:
                logger.warning(f"  [options] {ticker} exp {exp}: {e}")

        with open(out_path, "w") as f:
            json.dump(options_data, f, indent=2, default=safe_json)
        logger.info(f"  [options] {ticker}: {len(options_data['chains'])} expirations saved")
        return True
    except Exception as e:
        logger.error(f"  [options] {ticker}: {e}")
        return False


def collect_earnings(ticker: str) -> bool:
    """Download earnings calendar and history."""
    out_path = DATA_DIR / "earnings" / f"{ticker}.json"
    if out_path.exists():
        logger.info(f"  [earnings] {ticker}: already cached")
        return True
    try:
        stock = yf.Ticker(ticker)

        earnings_data = {}

        # Earnings dates
        try:
            dates = stock.earnings_dates
            if dates is not None and not dates.empty:
                earnings_data["earnings_dates"] = dates.reset_index().to_dict(orient="records")
        except Exception:
            earnings_data["earnings_dates"] = []

        # Quarterly earnings
        try:
            qe = stock.quarterly_earnings
            if qe is not None and not qe.empty:
                earnings_data["quarterly_earnings"] = qe.reset_index().to_dict(orient="records")
        except Exception:
            earnings_data["quarterly_earnings"] = []

        # Quarterly revenue
        try:
            qr = stock.quarterly_revenue
            if qr is not None and not qr.empty:
                earnings_data["quarterly_revenue"] = qr.reset_index().to_dict(orient="records")
        except Exception:
            earnings_data["quarterly_revenue"] = []

        with open(out_path, "w") as f:
            json.dump(earnings_data, f, indent=2, default=safe_json)
        logger.info(f"  [earnings] {ticker}: saved")
        return True
    except Exception as e:
        logger.error(f"  [earnings] {ticker}: {e}")
        return False


def collect_metadata(tickers: list[str], force: bool = False) -> bool:
    """Collect sector/industry metadata for all tickers.

    If metadata file already exists, merges new tickers into it.
    """
    out_path = DATA_DIR / "metadata" / "tickers.json"

    existing = {}
    if out_path.exists() and not force:
        with open(out_path) as f:
            existing = json.load(f)
        # Check if all tickers are already in metadata
        missing = [t for t in tickers if t not in existing]
        if not missing:
            logger.info(f"  [metadata]: all {len(tickers)} tickers already cached")
            return True
        logger.info(f"  [metadata]: adding {len(missing)} new tickers")
        tickers = missing

    try:
        metadata = dict(existing)
        for ticker in tickers:
            stock = yf.Ticker(ticker)
            info = stock.info or {}
            metadata[ticker] = {
                "name": info.get("shortName") or info.get("longName"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "market_cap": info.get("marketCap"),
                "beta": info.get("beta"),
                "currency": info.get("currency"),
                "exchange": info.get("exchange"),
                "country": info.get("country"),
            }
            time.sleep(0.3)  # rate limit

        with open(out_path, "w") as f:
            json.dump(metadata, f, indent=2, default=safe_json)
        logger.info(f"  [metadata]: {len(metadata)} tickers saved")
        return True
    except Exception as e:
        logger.error(f"  [metadata]: {e}")
        return False


def collect_for_tickers(
    tickers: list[str],
    benchmark: str,
    start_date: str,
    end_date: str,
) -> dict:
    """Collect all data types for a set of tickers."""
    # Ensure directories exist
    for subdir in ["prices", "fundamentals", "analyst", "options", "earnings", "metadata"]:
        (DATA_DIR / subdir).mkdir(parents=True, exist_ok=True)

    all_tickers = list(dict.fromkeys(tickers + [benchmark]))  # dedupe preserving order
    results = {"success": [], "failed": []}

    for ticker in all_tickers:
        logger.info(f"\n{'='*40} {ticker} {'='*40}")

        ok = True
        ok &= collect_prices(ticker, start_date, end_date)
        time.sleep(0.5)

        if ticker != benchmark:
            ok &= collect_fundamentals(ticker)
            time.sleep(0.5)
            ok &= collect_analyst(ticker)
            time.sleep(0.5)
            ok &= collect_options(ticker)
            time.sleep(0.5)
            ok &= collect_earnings(ticker)
            time.sleep(0.5)

        if ok:
            results["success"].append(ticker)
        else:
            results["failed"].append(ticker)

    # Metadata (all non-benchmark tickers)
    collect_metadata([t for t in all_tickers if t != benchmark])

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Download and cache market data for AEL experiments"
    )
    parser.add_argument(
        "--tickers", nargs="+", default=None,
        help="Tickers to collect (overrides --dataset)"
    )
    parser.add_argument(
        "--benchmark", default="SPY",
        help="Benchmark ticker (default: SPY)"
    )
    parser.add_argument(
        "--start", default=None,
        help="Start date YYYY-MM-DD (default: 2024-10-01)"
    )
    parser.add_argument(
        "--end", default=None,
        help="End date YYYY-MM-DD (default: 2025-03-28)"
    )
    parser.add_argument(
        "--dataset", choices=list(DATASETS.keys()) + ["all"], default=None,
        help="Preset dataset (d1/d2/d3/d4/all)"
    )
    parser.add_argument(
        "--data-dir", default=None,
        help="Output directory (default: data/)"
    )
    args = parser.parse_args()

    global DATA_DIR
    if args.data_dir:
        DATA_DIR = Path(args.data_dir)

    if args.dataset == "all":
        # Collect all datasets
        all_tickers = set()
        earliest_start = "2025-01-01"
        latest_end = "2024-01-01"
        for ds in DATASETS.values():
            all_tickers.update(ds["tickers"])
            all_tickers.add(ds["benchmark"])
            earliest_start = min(earliest_start, ds["start"])
            latest_end = max(latest_end, ds["end"])

        logger.info(f"Collecting ALL datasets: {len(all_tickers)} unique tickers")
        logger.info(f"Date range: {earliest_start} to {latest_end}")
        results = collect_for_tickers(
            list(all_tickers), "SPY", earliest_start, latest_end
        )
    elif args.dataset:
        ds = DATASETS[args.dataset]
        logger.info(f"Dataset {args.dataset}: {ds['description']}")
        results = collect_for_tickers(
            ds["tickers"], ds["benchmark"],
            args.start or ds["start"], args.end or ds["end"],
        )
    else:
        tickers = args.tickers or DEFAULT_TICKERS
        benchmark = args.benchmark
        start = args.start or DEFAULT_START
        end = args.end or DEFAULT_END

        logger.info(f"Collecting data for {len(tickers)} tickers + {benchmark}")
        logger.info(f"Date range: {start} to {end}")
        results = collect_for_tickers(tickers, benchmark, start, end)

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Collection complete:")
    logger.info(f"  Success: {len(results['success'])}/{len(results['success']) + len(results['failed'])}")
    if results["failed"]:
        logger.info(f"  Failed:  {results['failed']}")

    # Verify data sizes
    logger.info(f"\nData directory sizes:")
    for subdir in ["prices", "fundamentals", "analyst", "options", "earnings", "metadata"]:
        path = DATA_DIR / subdir
        if path.exists():
            files = list(path.glob("*"))
            total_size = sum(f.stat().st_size for f in files if f.is_file())
            logger.info(f"  {subdir}/: {len(files)} files, {total_size/1024:.1f} KB")


if __name__ == "__main__":
    main()
