"""Compute diverse evaluation metrics across all portfolio runs.

Usage: python -m experiments.summarize_metrics

Computes per-method (averaged over seeds):
- Sharpe ratio (risk-adjusted return)
- Sortino ratio (downside-risk-adjusted return)
- Calmar ratio (return / max drawdown)
- Total return (%)
- Max drawdown (%)
- Win rate (% of bars with positive return)
- Average turnover (portfolio churn)
- Volatility (annualized std dev)
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

LOGS_DIR = Path(__file__).parent.parent / "logs"
ANN_FACTOR = 252 * 4  # 4 bars/day * 252 trading days


def compute_metrics_from_history(score_history: list[dict], phase: str = "test") -> dict:
    """Compute diverse metrics from per-bar score history for a given phase."""
    bars = [b for b in score_history if b.get("phase") == phase]
    if not bars:
        return {}

    returns = [b["port_return"] for b in bars]
    arr = np.array(returns)
    n = len(arr)

    if n < 2:
        return {"n_bars": n}

    # Basic stats
    mean_ret = float(np.mean(arr))
    vol = float(np.std(arr, ddof=1))
    total_ret = float(np.prod(1 + arr) - 1)

    # Sharpe ratio (annualized)
    ann_ret = mean_ret * ANN_FACTOR
    ann_vol = vol * np.sqrt(ANN_FACTOR) if vol > 0 else 0
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    # Sortino ratio (only penalizes downside deviation)
    downside = arr[arr < 0]
    down_vol = float(np.std(downside, ddof=1)) if len(downside) > 1 else vol
    ann_down_vol = down_vol * np.sqrt(ANN_FACTOR) if down_vol > 0 else 0
    sortino = ann_ret / ann_down_vol if ann_down_vol > 0 else 0

    # Max drawdown
    cum = np.cumprod(1 + arr)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    max_dd = float(np.min(dd))

    # Calmar ratio (annualized return / |max drawdown|)
    ann_total = (1 + total_ret) ** (ANN_FACTOR / n) - 1
    calmar = ann_total / abs(max_dd) if max_dd != 0 else 0

    # Win rate
    win_rate = float(np.mean(arr > 0))

    # Average turnover
    turnovers = [b.get("turnover", 0) for b in bars]
    avg_turnover = float(np.mean(turnovers)) if turnovers else 0

    # Tail ratio (95th percentile gain / |5th percentile loss|)
    p95 = float(np.percentile(arr, 95))
    p5 = float(np.percentile(arr, 5))
    tail_ratio = p95 / abs(p5) if p5 != 0 else float("inf")

    return {
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "calmar": round(calmar, 3),
        "total_return_pct": round(total_ret * 100, 3),
        "max_drawdown_pct": round(max_dd * 100, 3),
        "win_rate": round(win_rate, 3),
        "volatility_ann": round(ann_vol, 4),
        "avg_turnover": round(avg_turnover, 4),
        "tail_ratio": round(tail_ratio, 3),
        "n_bars": n,
    }


def main():
    method_seeds: dict[str, list[dict]] = defaultdict(list)

    # Scan all pf_* dirs (both logs/ and logs/baselines/)
    scan_dirs = [LOGS_DIR]
    baselines_dir = LOGS_DIR / "baselines"
    if baselines_dir.exists():
        scan_dirs.append(baselines_dir)

    for parent in scan_dirs:
        for d in sorted(parent.iterdir()):
            if not d.is_dir() or not d.name.startswith("pf_"):
                continue
            bt = list(d.glob("backtest_*.json"))
            if not bt:
                continue

            data = json.loads(bt[0].read_text())
            score_history = data.get("score_history", [])
            if not score_history:
                continue

            # Extract method name
            m = re.match(r"^pf_(.+?)_(d_\w+)_s(\d+)(?:_.*)?$", d.name)
            method = m.group(1) if m else d.name

            metrics = compute_metrics_from_history(score_history, phase="test")
            if metrics and metrics.get("n_bars", 0) > 0:
                method_seeds[method].append(metrics)

    if not method_seeds:
        print("No portfolio results found")
        return

    # Key methods to report (ordered)
    key_methods = [
        "stateless", "tool_only", "tool_memory", "tool_memory_reflect",
        "tool_memory_reflect_skills", "eael_incremental",
        "step5_uniform_credit", "step5_llm_credit",
        "eael_no_reflect", "eael_no_plannerevolve", "eael_no_skills",
        "credit_ablation", "step5_curriculum", "step5_curriculum_llm",
        "step5_fcc_tuned", "step5_no_pertool", "step5_no_warmup",
        # Baselines
        "reflexion", "expel", "factorminer", "meta_reflexion", "evotool",
    ]

    metric_keys = ["sharpe", "sortino", "calmar", "total_return_pct",
                   "max_drawdown_pct", "win_rate", "volatility_ann", "avg_turnover", "tail_ratio"]

    # Print header
    print(f"{'Method':<28} {'N':>3} ", end="")
    for mk in metric_keys:
        short = mk.replace("_pct", "%").replace("_ann", "").replace("total_return", "Ret")
        short = short.replace("max_drawdown", "MaxDD").replace("win_rate", "WinR")
        short = short.replace("avg_turnover", "Turnov").replace("tail_ratio", "TailR")
        short = short.replace("volatility", "Vol").replace("sharpe", "Sharpe")
        short = short.replace("sortino", "Sortino").replace("calmar", "Calmar")
        print(f"{short:>10}", end="")
    print()
    print("-" * (28 + 3 + 10 * len(metric_keys) + 2))

    # Print each method
    for method in key_methods:
        seeds = method_seeds.get(method, [])
        if not seeds:
            continue

        n = len(seeds)
        row = f"{method:<28} {n:>3} "
        for mk in metric_keys:
            vals = [s.get(mk, 0) for s in seeds]
            mean = np.mean(vals)
            if n >= 3:
                std = np.std(vals, ddof=1)
                row += f"{mean:>6.2f}±{std:<3.1f}"
            else:
                row += f"{mean:>10.2f}"
        print(row)

    # Print JSON for easy consumption
    print(f"\n=== JSON (mean ± std) ===")
    summary = {}
    for method in key_methods:
        seeds = method_seeds.get(method, [])
        if not seeds:
            continue
        entry = {"n_seeds": len(seeds)}
        for mk in metric_keys:
            vals = [s.get(mk, 0) for s in seeds]
            entry[f"{mk}_mean"] = round(float(np.mean(vals)), 3)
            if len(vals) >= 2:
                entry[f"{mk}_std"] = round(float(np.std(vals, ddof=1)), 3)
        summary[method] = entry
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
