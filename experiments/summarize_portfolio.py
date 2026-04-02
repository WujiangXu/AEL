"""Summarize portfolio experiment results. Usage: python -m experiments.summarize_portfolio"""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict

logs_dir = Path(__file__).parent.parent / "logs"

results = {}
for d in sorted(logs_dir.iterdir()):
    if not d.is_dir() or not d.name.startswith("pf_"):
        continue
    bt = list(d.glob("backtest_*.json"))
    if not bt:
        continue
    data = json.loads(bt[0].read_text())
    results[d.name] = data

if not results:
    print("No portfolio results found (looking for pf_* dirs)")
    exit()

print(f"Found {len(results)} portfolio runs\n")
print(f"{'Run':<50} {'Return%':>8} {'Sharpe':>8} {'MaxDD%':>8} {'Bars':>6}")
print("-" * 85)

# Group by method
import re
method_results = defaultdict(list)
for name, data in results.items():
    m = re.match(r"^pf_(.+?)_(d_small|d_large|d[1-4])_s(\d+)(?:_.*)?$", name)
    method = m.group(1) if m else name
    metrics = data.get("metrics", {})
    phase = data.get("phase_metrics", {}).get("test", {})

    total_ret = metrics.get("total_return", 0)
    sharpe = metrics.get("sharpe_ratio", 0)
    mdd = metrics.get("max_drawdown", 0)
    n_bars = metrics.get("n_bars", 0)
    test_ret = phase.get("total_return", 0)
    test_sharpe = phase.get("sharpe", 0)

    print(f"{name:<50} {total_ret:>+7.2%} {sharpe:>8.2f} {mdd:>+7.2%} {n_bars:>6}")
    method_results[method].append({
        "total_return": total_ret, "sharpe": sharpe, "mdd": mdd,
        "test_return": test_ret, "test_sharpe": test_sharpe,
    })

print(f"\n{'='*85}")
print(f"AVERAGES BY METHOD (over seeds)")
print(f"{'='*85}")
print(f"{'Method':<25} {'Return%':>8} {'Sharpe':>8} {'MaxDD%':>8} {'TestRet%':>9} {'TestSharpe':>11} {'N':>3}")
print("-" * 85)

import numpy as np
for method in sorted(method_results.keys()):
    runs = method_results[method]
    n = len(runs)
    avg_ret = np.mean([r["total_return"] for r in runs])
    avg_sharpe = np.mean([r["sharpe"] for r in runs])
    avg_mdd = np.mean([r["mdd"] for r in runs])
    avg_test_ret = np.mean([r["test_return"] for r in runs])
    avg_test_sharpe = np.mean([r["test_sharpe"] for r in runs])
    print(f"{method:<25} {avg_ret:>+7.2%} {avg_sharpe:>8.2f} {avg_mdd:>+7.2%} {avg_test_ret:>+8.2%} {avg_test_sharpe:>11.2f} {n:>3}")

print(f"\nBenchmark: equal_weight = +1.04% test return, 4.85 test Sharpe, -1.0% max DD")
