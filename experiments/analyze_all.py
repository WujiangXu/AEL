"""Unified analysis: combine simple baselines + paper baselines + AEL methods.

Usage: python -m experiments.analyze_all --dataset d_small
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
BASELINE_LOGS = LOGS_DIR / "baselines"


def load_simple_baseline_results(dataset: str) -> dict[str, dict]:
    """Load simple baseline results from baselines/evaluate output."""
    results = {}
    for f in BASELINE_LOGS.glob(f"baseline_*_{dataset}.json"):
        data = json.loads(f.read_text())
        method = data["method"]
        avg = data.get("avg", {})
        results[method] = avg
    return results


def load_paper_baseline_results(dataset: str) -> dict[str, dict]:
    """Load paper baseline results from baselines/run_baselines output."""
    results = {}
    for d in BASELINE_LOGS.iterdir():
        if not d.is_dir() or dataset not in d.name:
            continue
        bt_files = list(d.glob("backtest_*.json"))
        if not bt_files:
            continue
        data = json.loads(bt_files[0].read_text())
        method = data.get("method", d.name.split("_" + dataset)[0])
        seed = data.get("seed", "?")
        key = f"{method}_s{seed}"
        results[key] = data
    return results


def load_ael_results() -> dict[str, dict]:
    """Load AEL method results from run_all output."""
    results = {}
    for d in sorted(LOGS_DIR.iterdir()):
        if not d.is_dir() or d.name == "baselines" or d.name == "__pycache__":
            continue
        bt_files = list(d.glob("backtest_*.json"))
        if not bt_files:
            continue
        data = json.loads(bt_files[0].read_text())
        results[d.name] = data
    return results


def extract_test_da(data: dict, horizons: list[int] | None = None) -> dict:
    """Extract test-phase directional accuracy from a result dict."""
    pm = data.get("phase_metrics", {})
    test = pm.get("test", data.get("metrics", {}))
    if not test:
        test = data.get("metrics", {})

    result = {}
    # Try all possible key formats
    for h in (horizons or [1, 2, 4]):
        for suffix in [f"{h}bar", f"{h}d", str(h)]:
            if suffix in test:
                result[f"{h}"] = test[suffix].get("directional_accuracy", 0)
                break

    # If no per-horizon keys found, try scanning all keys
    if not any(k.isdigit() for k in result):
        for key, val in test.items():
            if isinstance(val, dict) and "directional_accuracy" in val and key != "overall":
                # Extract horizon number from key (e.g., "1bar" -> "1", "7d" -> "7")
                h_str = "".join(c for c in key if c.isdigit())
                if h_str:
                    result[h_str] = val["directional_accuracy"]

    result["overall"] = test.get("overall", {}).get("directional_accuracy", 0)
    result["mape"] = test.get("overall", {}).get("mape", 0)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="d_small")
    args = parser.parse_args()

    horizons = [1, 2, 4]  # hourly

    # ── Simple baselines ──
    simple = load_simple_baseline_results(args.dataset)

    # ── Paper baselines ──
    paper_raw = load_paper_baseline_results(args.dataset)

    # ── AEL methods ──
    ael_raw = load_ael_results()
    # Debug info
    if ael_raw:
        print(f"(loaded {len(ael_raw)} AEL runs)")

    # ── Aggregate paper baselines by method (avg over seeds) ──
    paper_by_method = defaultdict(list)
    for key, data in paper_raw.items():
        method = data.get("method", key.rsplit("_s", 1)[0])
        paper_by_method[method].append(extract_test_da(data, horizons))

    paper_avg = {}
    for method, runs in paper_by_method.items():
        paper_avg[method] = {
            k: float(np.mean([r.get(k, 0) for r in runs]))
            for k in runs[0] if k != "mape"
        }
        paper_avg[method]["mape"] = float(np.mean([r.get("mape", 0) for r in runs]))
        paper_avg[method]["n_seeds"] = len(runs)

    # ── Aggregate AEL by method (avg over seeds) ──
    ael_by_method = defaultdict(list)
    for name, data in ael_raw.items():
        # Parse method from name: method_dataset_sSEED
        # Dataset names: d_small, d_large, d1, d2, d3, d4
        # Split off _sSEED first, then _dataset
        import re
        # Handle version-tagged names: method_dataset_sSEED_vTAG
        m = re.match(r"^(.+?)_(d_small|d_large|d[1-4])_s(\d+)(?:_.*)?$", name)
        if m:
            method = m.group(1)
        else:
            method = name
        ael_by_method[method].append(extract_test_da(data, horizons))

    ael_avg = {}
    for method, runs in ael_by_method.items():
        ael_avg[method] = {
            k: float(np.mean([r.get(k, 0) for r in runs]))
            for k in runs[0] if k != "mape"
        }
        ael_avg[method]["mape"] = float(np.mean([r.get("mape", 0) for r in runs]))
        ael_avg[method]["n_seeds"] = len(runs)

    # ── Print unified table ──
    print("=" * 85)
    print(f"UNIFIED RESULTS — {args.dataset} (test phase, avg over seeds)")
    print("=" * 85)

    h_labels = [f"DA-{h}bar" for h in horizons]
    header = f"{'Method':<24} {'Type':<12}" + "".join(f" {l:>9}" for l in h_labels) + f" {'Overall':>9} {'MAPE':>8} {'N':>3}"
    print(header)
    print("-" * 85)

    all_methods = []

    # Simple baselines
    for method in ["random", "buy_and_hold", "momentum", "ma_crossover"]:
        if method not in simple:
            continue
        test = simple[method].get("phase_test", simple[method].get("metrics", {}))
        row = {"method": method, "type": "simple"}
        for h in horizons:
            for suffix in [f"{h}bar", f"{h}d"]:
                if suffix in test:
                    row[str(h)] = test[suffix].get("directional_accuracy", 0)
                    break
        row["overall"] = test.get("overall", {}).get("directional_accuracy", 0)
        row["mape"] = test.get("overall", {}).get("mape", 0)
        row["n"] = 3
        all_methods.append(row)

    # Paper baselines
    for method in ["reflexion", "expel", "factorminer", "meta_reflexion", "evotool"]:
        if method not in paper_avg:
            continue
        r = paper_avg[method]
        row = {"method": method, "type": "paper", **r, "n": r.get("n_seeds", 0)}
        all_methods.append(row)

    # AEL methods
    ael_order = ["stateless", "memory_only", "tool_only", "ael_full", "credit_ablation", "eael_full"]
    for method in ael_order:
        if method not in ael_avg:
            continue
        r = ael_avg[method]
        row = {"method": method, "type": "AEL", **r, "n": r.get("n_seeds", 0)}
        all_methods.append(row)

    # EAEL ablations
    ablation_order = ["eael_no_reflect", "eael_no_coldstart", "eael_no_plannerevolve",
                       "eael_preset_tools", "eael_no_skills"]
    for method in ablation_order:
        if method not in ael_avg:
            continue
        r = ael_avg[method]
        row = {"method": method, "type": "ablation", **r, "n": r.get("n_seeds", 0)}
        all_methods.append(row)

    # Print rows
    prev_type = ""
    for row in all_methods:
        if row["type"] != prev_type:
            if prev_type:
                print("-" * 85)
            prev_type = row["type"]

        h_vals = "".join(f" {row.get(str(h), 0):>9.3f}" for h in horizons)
        overall = row.get("overall", 0)
        mape = row.get("mape", 0)
        n = row.get("n", 0)
        print(f"{row['method']:<24} {row['type']:<12}{h_vals} {overall:>9.3f} {mape:>8.4f} {n:>3}")

    # Best method
    if all_methods:
        best = max(all_methods, key=lambda r: r.get("overall", 0))
        print(f"\nBest: {best['method']} ({best['type']}) — Overall DA = {best['overall']:.3f}")

    # Save as JSON
    out_path = LOGS_DIR / f"unified_results_{args.dataset}.json"
    with open(out_path, "w") as f:
        json.dump(all_methods, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
