"""Recompute full Table 3 metrics from score_history in backtest files."""
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

import sys

LOGS = Path(__file__).parent.parent / "logs"
BASELINES = LOGS / "baselines"
SEEDS = {42, 123, 456, 789, 1024}
# Optional: filter to a specific git hash
HASH_FILTER = sys.argv[1] if len(sys.argv) > 1 else None

METHOD_MAP = {
    "pf_reflexion_": "Reflexion",
    "pf_expel_": "ExpeL",
    "pf_factorminer_": "FactorMiner",
    "pf_meta_reflexion_": "Meta-Reflexion",
    "pf_evotool_": "EvoTool",
    "pf_hyperagent_": "HyperAgent",
    "pf_stateless_": "Stateless",
    "pf_tool_only_": "+Tools",
    "pf_tool_memory_d_full": "+Memory",
    "pf_tool_memory_reflect_d_full": "+Reflect",
    "pf_tool_memory_reflect_skills_": "+Skills",
    "pf_eael_incremental_": "AEL (FCC)",
    "pf_step5_uniform_credit_": "Uniform credit",
    "pf_step5_llm_credit_": "AEL (LLM-FCC)",
    "pf_step5_no_pertool_": "- per_tool",
    "pf_step5_no_warmup_": "- warm-up",
    "pf_step5_fcc_tuned_": "FCC tuned",
    "pf_step5_ablation_no_reflect_": "w/o reflection",
    "pf_step5_ablation_no_planner_": "w/o planner evolve",
    "pf_eael_no_reflect_": "w/o reflection",
    "pf_eael_no_planner_": "w/o planner evolve",
    "pf_step5a_credit_only_": "step5a credit-only",
    "pf_step5b_credit_planner_": "step5b credit+planner",
    "pf_tool_memory_diffmem_": "+Memory (diff)",
    "pf_step5_llm_uni_a03_": "LLM+Uni α=0.3",
    "pf_step5_llm_uni_a05_": "LLM+Uni α=0.5",
    "pf_step5_llm_uni_a07_": "LLM+Uni α=0.7",
    # Ablation from +Reflect (step3 base)
    "pf_step3_no_warmup_": "R: -warmup",
    "pf_step3_coldstart_": "R: +coldstart",
    "pf_step3_fcc_": "R: +FCC credit",
    "pf_step3_llmfcc_": "R: +LLM-FCC credit",
    "pf_step3_planner_evolve_": "R: +planner evolve",
    "pf_step3_pertool_": "R: +per-tool",
    "pf_step3_skills_": "R: +skills",
    "pf_eael_no_plannerevolve_": "w/o planner evolve",
}

def get_method(dirname):
    for prefix, name in METHOD_MAP.items():
        if dirname.startswith(prefix):
            return name
    return None

def get_seed(dirname):
    """Extract seed number from dir name like pf_method_d_full_s42_hash."""
    import re
    m = re.search(r'_s(\d+)', dirname)
    return int(m.group(1)) if m else None

def matches_hash(dirname):
    """Check if dir matches the hash filter."""
    if not HASH_FILTER:
        return True
    return HASH_FILTER in dirname

def compute_metrics(returns):
    """Compute all 7 metrics from a list of per-bar returns."""
    arr = np.array(returns, dtype=float)
    if len(arr) < 2:
        return None
    mean_r = np.mean(arr)
    std_r = np.std(arr)
    ann = np.sqrt(252 * 4)

    sharpe = mean_r / std_r * ann if std_r > 1e-10 else 0

    down = arr[arr < 0]
    down_std = np.std(down) if len(down) > 1 else std_r
    sortino = mean_r / down_std * ann if down_std > 1e-10 else 0

    cum = np.cumprod(1 + arr)
    peak = np.maximum.accumulate(cum)
    dd = (peak - cum) / peak
    max_dd = float(np.max(dd)) if len(dd) > 0 else 0

    ann_ret = mean_r * 252 * 4
    calmar = ann_ret / max_dd if max_dd > 0.001 else 0

    win_rate = float(np.mean(arr > 0))

    p95 = np.percentile(arr, 95)
    p5 = np.percentile(arr, 5)
    tail_ratio = abs(p95 / p5) if abs(p5) > 1e-8 else 1.0

    total_ret = float(np.prod(1 + arr) - 1) * 100

    return {
        "sharpe": sharpe, "sortino": sortino, "calmar": calmar,
        "return_pct": total_ret, "max_dd_pct": max_dd * 100,
        "win_rate": win_rate, "tail_ratio": tail_ratio,
    }

# Collect — scan both logs/ and logs/baselines/, keep latest per method+seed
def scan_dir(base, results_dict):
    seen = {}  # (method, seed) -> dir mtime
    step3_dirs = [d.name for d in base.iterdir() if d.is_dir() and 'step3_' in d.name]
    if step3_dirs:
        print(f"DEBUG: Found {len(step3_dirs)} step3 dirs in {base}:")
        for sd in step3_dirs[:10]:
            d = base / sd
            bt = list(d.glob("backtest_*.json"))
            m = get_method(sd)
            print(f"  {sd} -> method={m}, backtest={len(bt)}")
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        method = get_method(d.name)
        if not method:
            continue
        seed = get_seed(d.name)
        if seed not in SEEDS:
            continue
        if not matches_hash(d.name):
            continue
        bt = list(d.glob("backtest_*.json"))
        if not bt:
            continue
        # Keep latest run per (method, seed)
        mtime = bt[0].stat().st_mtime
        key = (method, seed)
        if key in seen and seen[key] >= mtime:
            continue
        seen[key] = mtime
        data = json.loads(bt[0].read_text())
        sh = data.get("score_history", [])
        test_rets = [e["port_return"] for e in sh if e.get("phase") == "test"]
        if not test_rets:
            continue
        m = compute_metrics(test_rets)
        if m:
            m["_seed"] = seed
            results_dict[key] = m

raw = {}
scan_dir(LOGS, raw)
if BASELINES.exists():
    scan_dir(BASELINES, raw)

# Group by method
results = defaultdict(list)
for (method, seed), m in sorted(raw.items()):
    results[method].append(m)

# Print
METRICS = ["sharpe", "sortino", "calmar", "return_pct", "max_dd_pct", "win_rate", "tail_ratio"]
HEADERS = ["Sharpe", "Sortino", "Calmar", "Ret%", "MaxDD%", "WinR", "TailR"]
ORDER = [
    "Reflexion", "ExpeL", "FactorMiner", "Meta-Reflexion", "EvoTool", "HyperAgent",
    "Stateless", "+Tools", "+Memory", "+Reflect", "+Skills", "AEL (FCC)",
    "Uniform credit", "- per_tool", "- warm-up", "FCC tuned", "AEL (LLM-FCC)",
    "w/o reflection", "w/o planner evolve",
    # New isolation experiments
    "step5a credit-only", "step5b credit+planner",
    # Ablation from +Reflect
    "R: -warmup", "R: +coldstart", "R: +FCC credit", "R: +LLM-FCC credit",
    "R: +planner evolve", "R: +per-tool", "R: +skills",
]

print(f"{'Method':<22} {'N':>2}  " + "  ".join(f"{h:>10}" for h in HEADERS))
print("-" * 105)

for method in ORDER:
    if method not in results:
        continue
    runs = results[method]
    n = len(runs)
    print(f"{method:<22} {n:>2}  ", end="")
    for m in METRICS:
        vals = [r[m] for r in runs]
        mean = np.mean(vals)
        std = np.std(vals)
        if m in ("win_rate", "tail_ratio"):
            print(f"  {mean:8.2f}", end="")
        elif m == "max_dd_pct":
            if std > 0.01:
                print(f" -{abs(mean):5.2f}±{std:.2f}", end="")
            else:
                print(f" -{abs(mean):8.2f}", end="")
        else:
            if std > 0.01 and n > 1:
                print(f" {mean:+6.2f}±{std:.2f}", end="")
            else:
                print(f" {mean:+9.2f}", end="")
    print()

# LaTeX
print("\n=== LATEX ===\n")
for method in ORDER:
    if method not in results:
        continue
    runs = results[method]
    n = len(runs)
    parts = [f" & {method} & {n}"]
    for m in METRICS:
        vals = [r[m] for r in runs]
        mean = np.mean(vals)
        std = np.std(vals)
        if m in ("win_rate", "tail_ratio"):
            parts.append(f" & {mean:.2f}")
        elif m == "max_dd_pct":
            parts.append(f" & $-${abs(mean):.2f}" + (f"$\\pm${std:.2f}" if std > 0.01 and n > 1 else ""))
        elif m == "return_pct":
            parts.append(f" & {mean:+.2f}" + (f"$\\pm${std:.2f}" if std > 0.01 and n > 1 else ""))
        else:
            parts.append(f" & {mean:.2f}" + (f"$\\pm${std:.2f}" if std > 0.01 and n > 1 else ""))
    parts.append(" \\\\")
    print("".join(parts))
