"""Collect all paper 5-seed results and print summary table.

Usage:
    python -m experiments.collect_results
"""
import json
import re
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).parent.parent
LOGS = PROJECT / "logs"
BASELINES_DIR = LOGS / "baselines"

# Map from config key → display name used in the paper
METHOD_DISPLAY = {
    "stateless": "Stateless",
    "tool_only": "+Tools",
    "tool_memory": "+Memory",
    "tool_memory_reflect": "+Reflect",
    "tool_memory_reflect_skills": "+Skills",
    "eael_incremental": "AEL (FCC)",
    "step5_uniform_credit": "Uniform credit",
    "step5_no_pertool": "- per_tool",
    "step5_no_warmup": "- warm-up",
    "step5_fcc_tuned": "FCC tuned",
    "step5_llm_credit": "AEL (LLM-FCC)",
    "eael_no_reflect": "w/o reflection",
    "eael_no_plannerevolve": "w/o planner evolve",
    # Baselines
    "reflexion": "Reflexion",
    "expel": "ExpeL",
    "factorminer": "FactorMiner",
    "meta_reflexion": "Meta-Reflexion",
    "evotool": "EvoTool",
    "hyperagent": "HyperAgent",
}

SEEDS = [42, 123, 456, 789, 1024]


def extract_test_sharpe(result_file: Path) -> float | None:
    """Extract test-phase Sharpe from a backtest result JSON."""
    try:
        data = json.loads(result_file.read_text())
        # AEL portfolio format
        pm = data.get("phase_metrics", {})
        if "test" in pm:
            # Different key names in different runners
            test = pm["test"]
            return test.get("sharpe_ratio", test.get("sharpe", None))
        # Baseline portfolio format (seed_metrics list)
        sm = data.get("seed_metrics", [])
        if sm and isinstance(sm, list) and "phase_test" in sm[0]:
            return sm[0]["phase_test"].get("sharpe_ratio", None)
    except Exception as e:
        print(f"  Error reading {result_file}: {e}")
    return None


def find_result(method: str, seed: int) -> float | None:
    """Find and extract test Sharpe for a (method, seed) pair."""
    # AEL variants: logs/pf_{method}_d_full_s{seed}_{hash}/backtest_*.json
    pattern = f"pf_{method}_d_full_s{seed}_*"
    for d in sorted(LOGS.glob(pattern)):
        for f in d.glob("backtest_*.json"):
            val = extract_test_sharpe(f)
            if val is not None:
                return val

    # Baselines: logs/baselines/pf_{method}_d_full_s{seed}/backtest_*.json
    base_dir = BASELINES_DIR / f"pf_{method}_d_full_s{seed}"
    if base_dir.exists():
        for f in base_dir.glob("backtest_*.json"):
            val = extract_test_sharpe(f)
            if val is not None:
                return val

    return None


def main():
    print("=" * 90)
    print("PAPER 5-SEED RESULTS COLLECTION")
    print("=" * 90)

    all_results = {}

    for method_key, display_name in METHOD_DISPLAY.items():
        seed_values = {}
        for seed in SEEDS:
            val = find_result(method_key, seed)
            seed_values[seed] = val

        found = [v for v in seed_values.values() if v is not None]
        all_results[method_key] = seed_values

        # Print row
        vals_str = "  ".join(
            f"s{s}={v:+.3f}" if v is not None else f"s{s}=  ???"
            for s, v in seed_values.items()
        )
        if found:
            mean = np.mean(found)
            std = np.std(found)
            print(f"{display_name:25s} | {vals_str} | mean={mean:+.3f} std={std:.3f} (N={len(found)})")
        else:
            print(f"{display_name:25s} | {vals_str} | NO RESULTS")

    # Print Python-ready data for generate_paper_figures.py
    print("\n" + "=" * 90)
    print("COPY-PASTE DATA FOR generate_paper_figures.py")
    print("=" * 90)

    print("\n# FIG3_METHODS (credit comparison, 5 seeds)")
    for key in ["step5_uniform_credit", "eael_incremental", "step5_fcc_tuned",
                 "step5_llm_credit"]:
        vals = all_results.get(key, {})
        found = [vals[s] for s in SEEDS if vals.get(s) is not None]
        if found:
            arr_str = ", ".join(f"{v:.2f}" for v in found)
            print(f'    ("{METHOD_DISPLAY[key]}", [{arr_str}], ...),')

    print("\n# FIG2_STEPS (synergy, means)")
    for key in ["stateless", "tool_memory", "tool_memory_reflect",
                 "tool_memory_reflect_skills", "eael_incremental", "step5_llm_credit"]:
        vals = all_results.get(key, {})
        found = [vals[s] for s in SEEDS if vals.get(s) is not None]
        if found:
            print(f'    ("{METHOD_DISPLAY[key]}", {np.mean(found):.2f}, ...),')

    print("\n# Ablation table (per-seed)")
    for key in ["eael_incremental", "eael_no_reflect", "eael_no_plannerevolve"]:
        vals = all_results.get(key, {})
        found = [vals[s] for s in SEEDS if vals.get(s) is not None]
        if found:
            per_seed = "  ".join(f"{vals.get(s, 0):.2f}" for s in SEEDS)
            print(f"  {METHOD_DISPLAY[key]:25s}: {per_seed}  mean={np.mean(found):.2f}")


if __name__ == "__main__":
    main()
