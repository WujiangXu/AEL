"""Statistical significance tests for AEL paper (COLM 2026).

Computes bootstrap confidence intervals, paired bootstrap tests, Welch's t-test,
and Wilcoxon signed-rank tests for the main results (Table 1 and controlled experiments).

Usage:
    python experiments/statistical_tests.py
"""

from __future__ import annotations

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Seed results from paper Table 1 and controlled experiments
# (docs/incremental_results.md, Sections 2.1 and 2.5)
# ---------------------------------------------------------------------------

RESULTS: dict[str, list[float]] = {
    # AEL variants
    "LLM-FCC": [3.10, 1.11, 3.48, 2.85, 2.40],  # 5 seeds
    "AEL-FCC": [2.77, 1.32, 1.95, 0.68, 1.72, 0.35, 2.14, 0.19],  # 8 seeds
    "Uniform": [3.02, 1.58, 1.88],  # 3 seeds (controlled)
    # Prior work
    "EvoTool": [2.09, 0.45, 4.54, 1.30, -0.71],  # 5 seeds
    "Reflexion": [-0.76, -0.41, -0.62],  # 3 seeds
    "ExpeL": [-0.53, 4.58, 0.24],  # 3 seeds
    "FactorMiner": [0.91, -0.22, 2.04],  # 3 seeds
    "Meta-Reflexion": [0.40, -0.72, 1.53],  # 3 seeds
    # AEL build (stateless uses first 5 of 9 seeds for pairing)
    "Stateless": [0.81, 0.77, 1.02, -0.30, 0.52, 1.44, 0.71, 1.21, 0.14],  # 9 seeds
    # Non-LLM (deterministic, single values)
    "momentum": [1.44],
    "equal_weight": [0.70],
}

# Matched 3-seed incremental build (seeds 42, 123, 456)
INCREMENTAL_BUILD: dict[str, list[float]] = {
    "Stateless": [0.77],  # single matched run
    "+Memory": [0.36],
    "+Reflect": [0.42],
    "+Skills": [0.51],
    "AEL-FCC (incr)": [2.01],
}


# ---------------------------------------------------------------------------
# Bootstrap utilities
# ---------------------------------------------------------------------------

def bootstrap_ci(
    data: list[float] | np.ndarray,
    n_resamples: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap confidence interval for the mean.

    Args:
        data: observed values.
        n_resamples: number of bootstrap resamples.
        ci: confidence level (e.g. 0.95 for 95% CI).
        seed: random seed for reproducibility.

    Returns:
        (lower, upper) bounds of the CI.
    """
    rng = np.random.default_rng(seed)
    arr = np.asarray(data, dtype=float)
    n = len(arr)
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        means[i] = rng.choice(arr, size=n, replace=True).mean()
    alpha = (1 - ci) / 2
    return float(np.percentile(means, 100 * alpha)), float(np.percentile(means, 100 * (1 - alpha)))


def paired_bootstrap_diff(
    a: list[float] | np.ndarray,
    b: list[float] | np.ndarray,
    n_resamples: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap CI for the difference mean(a) - mean(b).

    When a and b have different lengths, uses unpaired bootstrap (resample each
    independently). When they have the same length, uses paired bootstrap
    (resample indices jointly).

    Returns:
        (mean_diff, ci_lower, ci_upper)
    """
    rng = np.random.default_rng(seed)
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    diffs = np.empty(n_resamples)

    if len(a_arr) == len(b_arr):
        # Paired bootstrap
        n = len(a_arr)
        for i in range(n_resamples):
            idx = rng.integers(0, n, size=n)
            diffs[i] = a_arr[idx].mean() - b_arr[idx].mean()
    else:
        # Unpaired bootstrap
        na, nb = len(a_arr), len(b_arr)
        for i in range(n_resamples):
            diffs[i] = (
                rng.choice(a_arr, size=na, replace=True).mean()
                - rng.choice(b_arr, size=nb, replace=True).mean()
            )

    alpha = (1 - ci) / 2
    mean_diff = float(a_arr.mean() - b_arr.mean())
    return mean_diff, float(np.percentile(diffs, 100 * alpha)), float(np.percentile(diffs, 100 * (1 - alpha)))


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def compute_all_tests(results: dict[str, list[float]] | None = None) -> str:
    """Run all statistical tests and return LaTeX-ready footnote text.

    Args:
        results: {method_name: [sharpe_per_seed]}. Uses paper defaults if None.

    Returns:
        Formatted string with all test results.
    """
    if results is None:
        results = RESULTS

    lines: list[str] = []
    sep = "=" * 72

    # --- 1. Bootstrap 95% CI for each method's mean Sharpe ---
    lines.append(sep)
    lines.append("1. Bootstrap 95% CI for mean Sharpe (10,000 resamples)")
    lines.append(sep)
    for name, vals in results.items():
        if len(vals) < 2:
            lines.append(f"  {name:20s}  mean={np.mean(vals):.2f}  (N={len(vals)}, CI not applicable)")
            continue
        lo, hi = bootstrap_ci(vals)
        lines.append(f"  {name:20s}  mean={np.mean(vals):.2f}  95% CI=[{lo:.2f}, {hi:.2f}]  (N={len(vals)})")
    lines.append("")

    # --- 2. Paired bootstrap: LLM-FCC vs each baseline ---
    llm_fcc = results["LLM-FCC"]
    lines.append(sep)
    lines.append("2. Paired bootstrap CI for LLM-FCC minus each method")
    lines.append(sep)
    for name, vals in results.items():
        if name == "LLM-FCC":
            continue
        if len(vals) < 2:
            continue
        diff, lo, hi = paired_bootstrap_diff(llm_fcc, vals)
        sig = "*" if lo > 0 else ""
        lines.append(f"  LLM-FCC - {name:20s}  Δ={diff:+.2f}  95% CI=[{lo:+.2f}, {hi:+.2f}] {sig}")
    lines.append("")

    # --- 3. Welch's t-test: LLM-FCC vs top-3 ---
    top3 = ["EvoTool", "AEL-FCC", "momentum"]
    lines.append(sep)
    lines.append("3. Welch's t-test: LLM-FCC vs top-3 baselines")
    lines.append(sep)
    for name in top3:
        vals = results.get(name, [])
        if len(vals) < 2:
            lines.append(f"  vs {name:20s}  N={len(vals)}, t-test not applicable (need N≥2)")
            continue
        t_stat, p_val = stats.ttest_ind(llm_fcc, vals, equal_var=False)
        sig = "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        lines.append(f"  vs {name:20s}  t={t_stat:.3f}  p={p_val:.4f} {sig}")
    lines.append("")

    # --- 4. Wilcoxon signed-rank (paired, N≥5) ---
    lines.append(sep)
    lines.append("4. Wilcoxon signed-rank test (where N≥5 paired observations exist)")
    lines.append(sep)
    for name, vals in results.items():
        if name == "LLM-FCC":
            continue
        n_pair = min(len(llm_fcc), len(vals))
        if n_pair < 5:
            lines.append(f"  vs {name:20s}  N_paired={n_pair}, skipped (need N≥5)")
            continue
        a_paired = llm_fcc[:n_pair]
        b_paired = vals[:n_pair]
        try:
            stat, p_val = stats.wilcoxon(a_paired, b_paired)
            sig = "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            lines.append(f"  vs {name:20s}  W={stat:.1f}  p={p_val:.4f} {sig}  (N={n_pair})")
        except ValueError as e:
            lines.append(f"  vs {name:20s}  Wilcoxon error: {e}")
    lines.append("")

    # --- 5. LaTeX-ready footnote text ---
    lines.append(sep)
    lines.append("5. LaTeX-ready footnote text")
    lines.append(sep)

    # Compute key values for footnote
    llm_lo, llm_hi = bootstrap_ci(llm_fcc)
    evo_vals = results["EvoTool"]
    diff_evo, diff_lo, diff_hi = paired_bootstrap_diff(llm_fcc, evo_vals)
    t_evo, p_evo = stats.ttest_ind(llm_fcc, evo_vals, equal_var=False)
    fcc_vals = results["AEL-FCC"]
    t_fcc, p_fcc = stats.ttest_ind(llm_fcc, fcc_vals, equal_var=False)

    footnote = (
        f"Bootstrap 95\\% CI for \\ael{{}} (\\llmfcc{{}}) mean Sharpe: "
        f"[{llm_lo:.2f}, {llm_hi:.2f}] ($N$=5, 10{{,}}000 resamples). "
        f"Paired bootstrap difference vs.\\ EvoTool: $\\Delta$={diff_evo:+.2f}, "
        f"95\\% CI [{diff_lo:+.2f}, {diff_hi:+.2f}]. "
        f"Welch's $t$-test: vs.\\ EvoTool $p$={p_evo:.3f}, "
        f"vs.\\ AEL-\\fcc{{}} $p$={p_fcc:.3f}."
    )
    lines.append(footnote)
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    print(compute_all_tests())
