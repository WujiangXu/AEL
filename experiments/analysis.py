"""Plotting and statistical analysis for AEL/EAEL experiments.

Includes both original analysis functions and EAEL-specific extensions:
- Per-tool Thompson convergence (E5)
- Planner evolution genealogy (E4)
- Regime analysis (E7)
- Cost analysis (E8)
- Cold-start advantage (E6)
- Paired bootstrap p-values with Holm-Bonferroni correction (E1/E2)
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ──────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────

def load_results(log_dir: str) -> dict:
    """Load backtest results from a log directory."""
    path = Path(log_dir)
    files = list(path.glob("backtest_*.json"))
    if not files:
        raise FileNotFoundError(f"No backtest results in {log_dir}")
    with open(files[0]) as f:
        return json.load(f)


def load_trajectories(log_dir: str) -> list[dict]:
    """Load JSONL trajectory file from a log directory."""
    path = Path(log_dir) / "trajectories.jsonl"
    if not path.exists():
        return []
    trajs = []
    with open(path) as f:
        for line in f:
            trajs.append(json.loads(line))
    return trajs


# ──────────────────────────────────────────────────────────────
# Original analysis functions
# ──────────────────────────────────────────────────────────────

def plot_learning_curves(
    results_dict: dict[str, dict],
    metric: str = "mean_score",
    output_path: str = "figures/learning_curves.pdf",
    title: str = "Learning Curves (Directional Accuracy)",
):
    """Plot learning curves for multiple methods."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for i, (method, results) in enumerate(results_dict.items()):
        curve = results.get("learning_curve", [])
        if not curve:
            continue
        episodes = [c["episode_end"] for c in curve]
        values = [c[metric] for c in curve]

        color = colors[i % len(colors)]
        ax.plot(episodes, values, label=method, color=color, linewidth=2)

        if "std_score" in curve[0]:
            stds = [c["std_score"] for c in curve]
            lower = [v - s for v, s in zip(values, stds)]
            upper = [v + s for v, s in zip(values, stds)]
            ax.fill_between(episodes, lower, upper, alpha=0.15, color=color)

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_credit_entropy(
    results: dict,
    output_path: str = "figures/credit_entropy.pdf",
):
    """Plot credit assignment entropy over episodes."""
    trajectories_path = Path(results.get("log_dir", "logs")) / "trajectories.jsonl"
    if not trajectories_path.exists():
        print(f"No trajectories file at {trajectories_path}")
        return

    credits_over_time = []
    with open(trajectories_path) as f:
        for line in f:
            traj = json.loads(line)
            mc = traj.get("module_credits", {})
            if mc:
                vals = np.array(list(mc.values()))
                vals = np.clip(vals, 1e-10, None)
                vals = vals / vals.sum()
                entropy = -np.sum(vals * np.log(vals))
                credits_over_time.append(entropy)

    if not credits_over_time:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    window = 10
    if len(credits_over_time) >= window:
        smoothed = np.convolve(credits_over_time, np.ones(window)/window, mode="valid")
        ax.plot(range(window-1, len(credits_over_time)), smoothed, linewidth=2)
    else:
        ax.plot(credits_over_time, linewidth=2)

    ax.axhline(y=np.log(3), color="gray", linestyle="--", label="Max entropy (uniform)")
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Credit Entropy", fontsize=12)
    ax.set_title("Module Credit Specialization Over Time", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_planner_selection(
    results: dict,
    output_path: str = "figures/planner_selection.pdf",
):
    """Plot planner selection distribution over time."""
    config_history = results.get("config_history", [])
    if not config_history:
        return

    planners = set(c["planner"] for c in config_history if "planner" in c)
    window = 20

    fig, ax = plt.subplots(figsize=(10, 5))
    for planner in sorted(planners):
        selections = [1 if c.get("planner") == planner else 0 for c in config_history]
        if len(selections) >= window:
            smoothed = np.convolve(selections, np.ones(window)/window, mode="valid")
            ax.plot(range(window-1, len(selections)), smoothed, label=planner, linewidth=2)

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Selection Frequency", fontsize=12)
    ax.set_title("Planner Selection Distribution Over Time", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def compute_aulc(scores: list[float]) -> float:
    """Compute Area Under Learning Curve (AULC), normalized by length."""
    if not scores:
        return 0.0
    return float(np.trapz(scores) / len(scores))


def compute_learning_gain(scores: list[float], window: int = 20) -> float:
    """Learning gain = mean(last window) - mean(first window)."""
    if len(scores) < 2 * window:
        return 0.0
    return float(np.mean(scores[-window:]) - np.mean(scores[:window]))


def bootstrap_ci(
    values: list[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval. Returns (mean, ci_lower, ci_upper)."""
    arr = np.array(values)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(arr, size=len(arr), replace=True)
        boot_means.append(np.mean(sample))
    boot_means = np.array(boot_means)

    alpha = (1 - ci) / 2
    return (
        float(np.mean(arr)),
        float(np.percentile(boot_means, alpha * 100)),
        float(np.percentile(boot_means, (1 - alpha) * 100)),
    )


def comparison_table(results_dict: dict[str, dict], phase: str = "test") -> str:
    """Generate a markdown comparison table.

    Args:
        results_dict: method_name -> backtest results dict.
        phase: Which phase to report ('test', 'train', 'val', or 'overall').
    """
    header = "| Method | Dir Acc (1d) | Dir Acc (7d) | Dir Acc (14d) | Overall | MAPE | AULC | LG |"
    sep = "|---|---|---|---|---|---|---|---|"
    rows = [header, sep]

    for method, results in results_dict.items():
        # Use phase-specific metrics if available, fall back to overall
        if phase != "overall" and "phase_metrics" in results:
            metrics = results.get("phase_metrics", {}).get(phase, {})
        else:
            metrics = results.get("metrics", {})

        if not metrics:
            metrics = results.get("metrics", {})

        d1 = metrics.get("1d", {}).get("directional_accuracy", 0)
        d7 = metrics.get("7d", {}).get("directional_accuracy", 0)
        d14 = metrics.get("14d", {}).get("directional_accuracy", 0)
        overall = metrics.get("overall", {}).get("directional_accuracy", 0)
        mape = metrics.get("overall", {}).get("mape", 0)

        # Filter scores to phase if available
        score_history = results.get("score_history", [])
        if phase != "overall":
            scores = [s["score"] for s in score_history if s.get("phase", "train") == phase]
        else:
            scores = [s["score"] for s in score_history]

        aulc = compute_aulc(scores) if scores else 0
        lg = compute_learning_gain(scores) if scores else 0

        row = (
            f"| {method} | {d1:.3f} | {d7:.3f} | {d14:.3f} | "
            f"{overall:.3f} | {mape:.4f} | {aulc:.3f} | {lg:+.3f} |"
        )
        rows.append(row)

    return "\n".join(rows)


# ──────────────────────────────────────────────────────────────
# EAEL-specific analysis extensions
# ──────────────────────────────────────────────────────────────

def plot_per_tool_convergence(
    log_dir: str,
    output_path: str = "figures/per_tool_convergence.pdf",
    title: str = "Per-Tool Thompson Beta Convergence",
):
    """E5: Plot Thompson Beta(alpha, beta) posterior mean for each tool over episodes.

    Reads evolution_state.json or trajectories to reconstruct per-tool hit/miss history.
    """
    state_path = Path(log_dir) / "evolution_state.json"
    if not state_path.exists():
        print(f"No evolution state at {state_path}")
        return

    with open(state_path) as f:
        state = json.load(f)

    tool_profiles = state.get("tool_registry", {})
    if not tool_profiles:
        return

    # Also try to reconstruct per-episode trajectory for time series
    trajs = load_trajectories(log_dir)

    if trajs:
        # Reconstruct running hit/miss counts per tool
        tool_history: dict[str, list[float]] = defaultdict(list)
        tool_hits: dict[str, int] = defaultdict(int)
        tool_misses: dict[str, int] = defaultdict(int)

        for traj in trajs:
            tools_invoked = traj.get("tools_invoked", [])
            for tc in tools_invoked:
                name = tc.get("tool_name", "")
                if not name:
                    continue
                # Use outcome_score as proxy (positive = hit)
                score = traj.get("outcome_score", 0)
                if score > 0:
                    tool_hits[name] += 1
                elif score < 0:
                    tool_misses[name] += 1
                alpha = tool_hits[name] + 1
                beta = tool_misses[name] + 1
                posterior_mean = alpha / (alpha + beta)
                tool_history[name].append(posterior_mean)

        fig, ax = plt.subplots(figsize=(12, 6))
        for tool_name, history in sorted(tool_history.items()):
            if len(history) > 5:
                ax.plot(history, label=tool_name, linewidth=1.5)

        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Prior (0.5)")
        ax.set_xlabel("Tool Invocation Count", fontsize=12)
        ax.set_ylabel("Posterior Mean P(correct)", fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Saved: {output_path}")
    else:
        # Static bar chart from final state
        fig, ax = plt.subplots(figsize=(10, 5))
        names = []
        means = []
        for name, profile in sorted(tool_profiles.items()):
            alpha = profile.get("directional_hits", 0) + 1
            beta = profile.get("directional_misses", 0) + 1
            names.append(name.replace("compute_", "").replace("get_", ""))
            means.append(alpha / (alpha + beta))

        ax.barh(names, means, color="#2ca02c")
        ax.axvline(x=0.5, color="gray", linestyle="--", label="Prior")
        ax.set_xlabel("Posterior Mean P(correct)", fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend()

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Saved: {output_path}")


def plot_planner_genealogy(
    results: dict,
    output_path: str = "figures/planner_genealogy.pdf",
):
    """E4: Plot planner evolution timeline — when new planners are generated,
    their names, survival, and performance vs predecessor."""
    reflections = results.get("reflections", [])
    planner_pool = results.get("planner_pool", {})
    config_history = results.get("config_history", [])

    if not reflections:
        return

    # Track evolution events from reflections
    evolution_events = []
    for i, r in enumerate(reflections):
        structural = r.get("structural_problem", {})
        if structural.get("detected", False):
            evolution_events.append({
                "day": i,
                "date": r.get("date", f"day_{i}"),
                "failing_planner": r.get("planner_critique", {}).get("failing_planner", "?"),
                "description": structural.get("description", ""),
            })

    if not evolution_events:
        print("No planner evolution events found.")
        return

    # Build planner lifetime from config_history
    planner_first_seen: dict[str, int] = {}
    planner_last_seen: dict[str, int] = {}
    for c in config_history:
        p = c.get("planner", "")
        ep = c.get("episode", 0)
        if p not in planner_first_seen:
            planner_first_seen[p] = ep
        planner_last_seen[p] = ep

    fig, ax = plt.subplots(figsize=(14, 4 + len(planner_first_seen) * 0.3))

    # Timeline bars for each planner
    planners = sorted(planner_first_seen.keys())
    for i, p in enumerate(planners):
        start = planner_first_seen[p]
        end = planner_last_seen[p]
        is_evolved = p not in {
            "sequential", "decompose", "adaptive",
            "cot_reasoning", "reflexion", "hypothesis_test",
        }
        color = "#d62728" if is_evolved else "#1f77b4"
        ax.barh(i, end - start + 1, left=start, height=0.6, color=color, alpha=0.7)
        ax.text(start + 1, i, p, va="center", fontsize=8)

    # Mark evolution events
    for ev in evolution_events:
        day_episode = ev["day"] * 10  # approximate
        ax.axvline(x=day_episode, color="red", linestyle=":", alpha=0.5)

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_title("Planner Evolution Timeline", fontsize=14)
    ax.set_yticks(range(len(planners)))
    ax.set_yticklabels(planners, fontsize=8)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_regime_analysis(
    results: dict,
    output_path: str = "figures/regime_analysis.pdf",
):
    """E7: Per-regime performance breakdown on D4.

    Splits score_history by SPY volatility/return regime segments.
    """
    score_history = results.get("score_history", [])
    if not score_history:
        return

    # Define regime boundaries by date
    regimes = {
        "A: Yen Carry Unwind": ("2024-07-01", "2024-08-31"),
        "B: Recovery Rally": ("2024-09-01", "2024-11-30"),
        "C: Post-Election": ("2024-12-01", "2025-01-31"),
        "D: Tariff Correction": ("2025-02-01", "2025-03-28"),
    }

    regime_scores: dict[str, list[float]] = defaultdict(list)
    for entry in score_history:
        date = entry.get("date", "")
        for regime_name, (start, end) in regimes.items():
            if start <= date <= end:
                regime_scores[regime_name].append(entry["score"])
                break

    if not regime_scores:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart: mean score per regime
    ax = axes[0]
    names = list(regime_scores.keys())
    means = [np.mean(scores) for scores in regime_scores.values()]
    stds = [np.std(scores) / np.sqrt(len(scores)) for scores in regime_scores.values()]
    colors = ["#ff7f0e", "#2ca02c", "#1f77b4", "#d62728"]
    ax.bar(range(len(names)), means, yerr=stds, color=colors[:len(names)], alpha=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.split(":")[0] for n in names], fontsize=10)
    ax.set_ylabel("Mean Score", fontsize=12)
    ax.set_title("Performance by Market Regime", fontsize=13)
    ax.grid(True, alpha=0.3, axis="y")

    # Box plot: score distribution per regime
    ax2 = axes[1]
    data = [scores for scores in regime_scores.values()]
    bp = ax2.boxplot(data, labels=[n.split(":")[0] for n in names], patch_artist=True)
    for patch, color in zip(bp["boxes"], colors[:len(names)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax2.set_ylabel("Score Distribution", fontsize=12)
    ax2.set_title("Score Distribution by Regime", fontsize=13)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def generate_cost_table(
    results_dict: dict[str, dict],
) -> str:
    """E8: Generate LLM cost analysis table.

    Estimates calls/episode, tokens, and approximate cost per method.
    """
    # Cost model per call type (approximate tokens)
    CALL_COSTS = {
        "predictor": {"calls_per_episode": 1, "tokens": 2000},
        "morning_planning": {"calls_per_day": 1, "tokens": 1500},
        "evening_reflection": {"calls_per_day": 1, "tokens": 2000},
        "cold_start": {"calls_total": 1, "tokens": 3000},
        "planner_evolution": {"calls_total": 2, "tokens": 4000},  # average
    }

    # Per-method active components
    METHOD_COMPONENTS = {
        "M1-Stateless": ["predictor"],
        "M2-Memory": ["predictor"],
        "M3-Tools": ["predictor"],
        "M4-AEL-Full": ["predictor"],
        "M5-AEL-NoCred": ["predictor"],
        "M6-EAEL-Full": ["predictor", "morning_planning", "evening_reflection", "cold_start", "planner_evolution"],
        "M7-NoReflect": ["predictor", "cold_start"],
        "M8-NoColdStart": ["predictor", "morning_planning", "evening_reflection", "planner_evolution"],
        "M9-NoPlannerEvolve": ["predictor", "morning_planning", "evening_reflection", "cold_start"],
        "M10-PresetTools": ["predictor", "morning_planning", "evening_reflection", "cold_start", "planner_evolution"],
    }

    header = "| Method | Calls/Episode | Tokens/Episode | Est. $/1000 ep |"
    sep = "|---|---|---|---|"
    rows = [header, sep]

    for method, components in METHOD_COMPONENTS.items():
        total_calls = 0
        total_tokens = 0

        for comp in components:
            cost = CALL_COSTS[comp]
            if "calls_per_episode" in cost:
                total_calls += cost["calls_per_episode"]
                total_tokens += cost["tokens"]
            elif "calls_per_day" in cost:
                # ~10 episodes per day (10 tickers)
                total_calls += cost["calls_per_day"] / 10
                total_tokens += cost["tokens"] / 10
            elif "calls_total" in cost:
                # Amortize over ~600 episodes
                total_calls += cost["calls_total"] / 600
                total_tokens += cost["tokens"] / 600

        # Approximate cost at $0.10/1M tokens (gpt5-nano estimate)
        cost_per_1k = total_tokens * 1000 * 0.10 / 1_000_000

        row = f"| {method} | {total_calls:.2f} | {total_tokens:.0f} | ${cost_per_1k:.2f} |"
        rows.append(row)

    return "\n".join(rows)


def plot_cold_start_advantage(
    results_with_cs: dict,
    results_without_cs: dict,
    n_episodes: int = 50,
    output_path: str = "figures/cold_start_advantage.pdf",
):
    """E6: Plot first-N episodes of M6 vs M8 to quantify cold-start savings."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for label, results, color in [
        ("EAEL-Full (with cold-start)", results_with_cs, "#1f77b4"),
        ("EAEL-NoColdStart", results_without_cs, "#d62728"),
    ]:
        scores = [s["dir_acc"] for s in results.get("score_history", [])[:n_episodes]]
        if not scores:
            continue
        cumulative = np.cumsum(scores) / np.arange(1, len(scores) + 1)
        ax.plot(cumulative, label=label, color=color, linewidth=2)

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Cumulative Directional Accuracy", fontsize=12)
    ax.set_title(f"Cold-Start Advantage (First {n_episodes} Episodes)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Compute savings: episode where no-CS reaches CS's 10-episode accuracy
    cs_scores = [s["dir_acc"] for s in results_with_cs.get("score_history", [])[:n_episodes]]
    no_cs_scores = [s["dir_acc"] for s in results_without_cs.get("score_history", [])[:n_episodes]]
    if cs_scores and no_cs_scores:
        cs_10 = np.mean(cs_scores[:10]) if len(cs_scores) >= 10 else np.mean(cs_scores)
        cs_cum = np.cumsum(no_cs_scores) / np.arange(1, len(no_cs_scores) + 1)
        catch_up = np.argmax(cs_cum >= cs_10) if np.any(cs_cum >= cs_10) else len(no_cs_scores)
        ax.annotate(
            f"Cold-start saves ~{catch_up} episodes",
            xy=(catch_up, cs_10), fontsize=10,
            arrowprops=dict(arrowstyle="->"),
            xytext=(catch_up + 5, cs_10 - 0.05),
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_memory_dynamics(
    results_dict: dict[str, dict],
    output_path: str = "figures/memory_dynamics.pdf",
):
    """E9: Memory tier counts over episodes for multiple methods."""
    # This requires trajectory-level data with memory size snapshots
    # For now, plot from evolution_summary if available
    fig, ax = plt.subplots(figsize=(10, 5))

    for method, results in results_dict.items():
        summary = results.get("evolution_summary", {})
        mem_size = summary.get("memory_size", 0)
        if mem_size:
            ax.bar(method, mem_size, alpha=0.7)

    ax.set_ylabel("Memory Entries", fontsize=12)
    ax.set_title("Final Memory Size by Method", fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


# ──────────────────────────────────────────────────────────────
# Statistical testing
# ──────────────────────────────────────────────────────────────

def compute_paired_bootstrap_pvalue(
    scores_a: list[float],
    scores_b: list[float],
    n_bootstrap: int = 10000,
    alternative: str = "greater",
) -> float:
    """Paired bootstrap test: is method A significantly better than method B?

    Args:
        scores_a: Per-episode scores for method A.
        scores_b: Per-episode scores for method B (same episodes).
        n_bootstrap: Number of bootstrap samples.
        alternative: 'greater' (A > B), 'less' (A < B), or 'two-sided'.

    Returns:
        p-value.
    """
    n = min(len(scores_a), len(scores_b))
    a = np.array(scores_a[:n])
    b = np.array(scores_b[:n])
    observed_diff = np.mean(a) - np.mean(b)

    diffs = a - b
    boot_diffs = []
    for _ in range(n_bootstrap):
        idx = np.random.randint(0, n, size=n)
        boot_diffs.append(np.mean(diffs[idx]))
    boot_diffs = np.array(boot_diffs)

    if alternative == "greater":
        p = float(np.mean(boot_diffs <= 0))
    elif alternative == "less":
        p = float(np.mean(boot_diffs >= 0))
    else:
        p = float(np.mean(np.abs(boot_diffs) >= abs(observed_diff)))

    return p


def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Holm-Bonferroni correction for multiple hypothesis testing.

    Returns list of booleans: True if the corresponding hypothesis is rejected.
    """
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    rejected = [False] * n

    for rank, (orig_idx, p) in enumerate(indexed):
        adjusted_alpha = alpha / (n - rank)
        if p <= adjusted_alpha:
            rejected[orig_idx] = True
        else:
            break  # stop rejecting once one fails

    return rejected


def significance_table(
    results_dict: dict[str, dict],
    baseline_method: str,
    phase: str = "test",
) -> str:
    """Generate a significance table comparing all methods to a baseline.

    Uses paired bootstrap with Holm-Bonferroni correction.
    """
    if baseline_method not in results_dict:
        return f"Baseline '{baseline_method}' not found in results."

    baseline_scores = [
        s["score"]
        for s in results_dict[baseline_method].get("score_history", [])
        if s.get("phase", "train") == phase or phase == "overall"
    ]

    comparisons = []
    for method, results in results_dict.items():
        if method == baseline_method:
            continue
        method_scores = [
            s["score"]
            for s in results.get("score_history", [])
            if s.get("phase", "train") == phase or phase == "overall"
        ]
        if method_scores:
            p = compute_paired_bootstrap_pvalue(method_scores, baseline_scores)
            diff = np.mean(method_scores) - np.mean(baseline_scores)
            comparisons.append((method, p, diff))

    if not comparisons:
        return "No comparisons available."

    # Holm-Bonferroni
    p_values = [c[1] for c in comparisons]
    rejected = holm_bonferroni(p_values)

    header = f"| Method | vs {baseline_method} (delta) | p-value | Significant |"
    sep = "|---|---|---|---|"
    rows = [header, sep]
    for (method, p, diff), is_sig in zip(comparisons, rejected):
        sig_str = "Yes *" if is_sig else "No"
        rows.append(f"| {method} | {diff:+.4f} | {p:.4f} | {sig_str} |")

    return "\n".join(rows)


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze AEL/EAEL experiment results")
    parser.add_argument("--log-dirs", nargs="+", required=True,
                        help="Log directories to compare")
    parser.add_argument("--names", nargs="+", default=None,
                        help="Method names (same order as log-dirs)")
    parser.add_argument("--output-dir", default="figures",
                        help="Output directory for figures")
    parser.add_argument("--phase", default="test",
                        choices=["train", "val", "test", "overall"],
                        help="Which phase to report metrics for")
    args = parser.parse_args()

    names = args.names or [Path(d).name for d in args.log_dirs]
    results_dict = {}
    for name, log_dir in zip(names, args.log_dirs):
        try:
            results_dict[name] = load_results(log_dir)
        except FileNotFoundError as e:
            print(f"Warning: {e}")

    if results_dict:
        print(f"\n{'='*60}")
        print(f"Comparison Table ({args.phase} phase)")
        print(f"{'='*60}")
        print(comparison_table(results_dict, phase=args.phase))

        print(f"\n{'='*60}")
        print("Cost Analysis")
        print(f"{'='*60}")
        print(generate_cost_table(results_dict))

        plot_learning_curves(
            results_dict,
            output_path=f"{args.output_dir}/learning_curves.pdf",
        )

        # Per-method plots
        for name, log_dir in zip(names, args.log_dirs):
            results = results_dict.get(name)
            if results:
                plot_per_tool_convergence(
                    log_dir,
                    output_path=f"{args.output_dir}/per_tool_{name}.pdf",
                )
                plot_planner_genealogy(
                    results,
                    output_path=f"{args.output_dir}/genealogy_{name}.pdf",
                )
