"""Quick summary of all backtest results. Run: python experiments/summarize.py"""
from __future__ import annotations
import json, sys
from pathlib import Path
from collections import defaultdict

logs_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent.parent / "logs"

# Collect all backtest results
all_results = {}
for run_dir in sorted(logs_dir.iterdir()):
    if not run_dir.is_dir():
        continue
    bt_files = list(run_dir.glob("backtest_*.json"))
    if not bt_files:
        continue
    with open(bt_files[0]) as f:
        all_results[run_dir.name] = json.load(f)

print(f"Found {len(all_results)} completed runs\n")

# Parse method, dataset, seed
parsed = []
for name, data in all_results.items():
    parts = name.rsplit("_", 2)
    if len(parts) >= 3 and parts[2].startswith("s"):
        method, dataset, seed = parts[0], parts[1], parts[2]
    else:
        method, dataset, seed = name, "?", "?"

    metrics = data.get("metrics", {})
    pm = data.get("phase_metrics", {})
    tm = pm.get("test", metrics)

    n_episodes = data.get("num_episodes", len(data.get("score_history", [])))
    scores = data.get("score_history", [])
    n_llm = sum(1 for s in scores if s.get("used_llm", True))

    parsed.append({
        "name": name, "method": method, "dataset": dataset, "seed": seed,
        "d1": tm.get("1d", {}).get("directional_accuracy", 0),
        "d7": tm.get("7d", {}).get("directional_accuracy", 0),
        "d14": tm.get("14d", {}).get("directional_accuracy", 0),
        "overall": tm.get("overall", {}).get("directional_accuracy", 0),
        "mape": tm.get("overall", {}).get("mape", 0),
        "n_episodes": n_episodes,
        "n_llm": n_llm,
    })

# ── Full table: every run ──
print("=" * 105)
print("FULL RESULTS TABLE — Every run (test-phase metrics)")
print("=" * 105)
print(f"{'Run Name':<40} {'DA-1d':>7} {'DA-7d':>7} {'DA-14d':>7} {'Overall':>8} {'MAPE':>8} {'Episodes':>9}")
print("-" * 105)
for r in sorted(parsed, key=lambda x: (x["method"], x["dataset"], x["seed"])):
    print(f"{r['name']:<40} {r['d1']:>7.3f} {r['d7']:>7.3f} {r['d14']:>7.3f} "
          f"{r['overall']:>8.3f} {r['mape']:>8.4f} {r['n_episodes']:>9}")

print()

# ── E1: Main comparison ──
methods_e1 = ["stateless", "memory_only", "tool_only", "ael_full", "credit_ablation", "eael_full"]
datasets = ["d1", "d2", "d3", "d4"]

print("=" * 85)
print("E1: MAIN COMPARISON — Test Directional Accuracy (avg over 3 seeds)")
print("=" * 85)
print(f"{'Method':<22} {'D1':>10} {'D2':>10} {'D3':>10} {'D4':>10} {'Grand':>10}")
print("-" * 85)

grand = {}
for method in methods_e1:
    row = []
    for ds in datasets:
        runs = [r for r in parsed if r["method"] == method and r["dataset"] == ds]
        if runs:
            avg = sum(r["overall"] for r in runs) / len(runs)
            row.append(avg)
        else:
            row.append(None)
    grand_avg = sum(v for v in row if v is not None) / max(sum(1 for v in row if v is not None), 1)
    grand[method] = grand_avg
    cells = [f"{v:.3f}" if v is not None else "  —  " for v in row]
    print(f"{method:<22} {'  '.join(f'{c:>8}' for c in cells)}   {grand_avg:>7.3f}")

# Highlight best
best_method = max(grand, key=grand.get)
print(f"\nBest method: {best_method} ({grand[best_method]:.3f})")

# ── E1 detailed: per-horizon ──
print(f"\n{'Method':<22} {'DA-1d':>8} {'DA-7d':>8} {'DA-14d':>8} {'MAPE':>8}")
print("-" * 55)
for method in methods_e1:
    runs = [r for r in parsed if r["method"] == method]
    if not runs:
        continue
    n = len(runs)
    print(f"{method:<22} {sum(r['d1'] for r in runs)/n:>7.3f} "
          f"{sum(r['d7'] for r in runs)/n:>7.3f} "
          f"{sum(r['d14'] for r in runs)/n:>7.3f} "
          f"{sum(r['mape'] for r in runs)/n:>8.4f}")

# ── E2: EAEL ablation ──
methods_e2 = ["eael_full", "eael_no_reflect", "eael_no_coldstart",
              "eael_no_plannerevolve", "eael_preset_tools", "eael_no_skills"]

print("\n" + "=" * 85)
print("E2: EAEL ABLATION on D2 — Test Directional Accuracy (avg ± std over 3 seeds)")
print("=" * 85)
print(f"{'Config':<28} {'DA-1d':>8} {'DA-7d':>8} {'DA-14d':>8} {'Overall':>12} {'MAPE':>8}")
print("-" * 85)

e2_grand = {}
for method in methods_e2:
    runs = [r for r in parsed if r["method"] == method and r["dataset"] == "d2"]
    if not runs:
        continue
    n = len(runs)
    avg_ov = sum(r["overall"] for r in runs) / n
    std_ov = (sum((r["overall"] - avg_ov)**2 for r in runs) / max(n - 1, 1))**0.5
    e2_grand[method] = avg_ov
    print(f"{method:<28} {sum(r['d1'] for r in runs)/n:>7.3f} "
          f"{sum(r['d7'] for r in runs)/n:>7.3f} "
          f"{sum(r['d14'] for r in runs)/n:>7.3f} "
          f"{avg_ov:>7.3f}±{std_ov:.3f} "
          f"{sum(r['mape'] for r in runs)/n:>7.4f}")

if e2_grand:
    best_e2 = max(e2_grand, key=e2_grand.get)
    worst_e2 = min(e2_grand, key=e2_grand.get)
    print(f"\nBest: {best_e2} ({e2_grand[best_e2]:.3f})")
    print(f"Biggest drop: {worst_e2} ({e2_grand[worst_e2]:.3f}, "
          f"Δ={e2_grand[worst_e2] - e2_grand.get('eael_full', 0):+.3f} vs full)")

# ── Learning gain ──
print("\n" + "=" * 85)
print("LEARNING GAIN: First-quarter vs Last-quarter scores (D2, seed=42)")
print("=" * 85)
for method in methods_e1 + [m for m in methods_e2 if m not in methods_e1]:
    name = f"{method}_d2_s42"
    if name not in all_results:
        continue
    scores = [s.get("score", 0) for s in all_results[name].get("score_history", [])]
    if len(scores) >= 8:
        q = len(scores) // 4
        first_q = sum(scores[:q]) / q
        last_q = sum(scores[-q:]) / q
        gain = last_q - first_q
        print(f"  {method:<28} first_q={first_q:.3f}  last_q={last_q:.3f}  gain={gain:+.3f}")
