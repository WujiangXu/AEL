"""Experiment orchestrator: run all (method, dataset, seed) combinations.

Usage:
    # Run E1 main comparison (priority 1):
    python experiments/run_all.py --experiment e1 --parallel 4

    # Run E2 ablation (priority 2):
    python experiments/run_all.py --experiment e2 --parallel 4

    # Dry run to see what would be executed:
    python experiments/run_all.py --experiment e1 --dry-run

    # Single specific run:
    python experiments/run_all.py --method eael_full --dataset d2 --seed 42

    # Smoke test (20 episodes):
    python experiments/run_all.py --experiment smoke --parallel 2
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
CONFIGS_DIR = PROJECT_ROOT / "experiments" / "configs"

# Method config files (M1-M10 + incremental steps)
INCREMENTAL_DIR = CONFIGS_DIR / "incremental"
METHOD_CONFIGS = {
    "stateless":              CONFIGS_DIR / "stateless.yaml",           # M1
    "memory_only":            CONFIGS_DIR / "memory_only.yaml",         # M2
    "tool_only":              CONFIGS_DIR / "tool_only.yaml",           # M3
    "ael_full":               CONFIGS_DIR / "ael_full.yaml",            # M4
    "credit_ablation":        CONFIGS_DIR / "credit_ablation.yaml",     # M5
    "eael_full":              CONFIGS_DIR / "eael_full.yaml",           # M6
    "eael_no_reflect":        CONFIGS_DIR / "eael_no_reflect.yaml",     # M7
    "eael_no_coldstart":      CONFIGS_DIR / "eael_no_coldstart.yaml",   # M8
    "eael_no_plannerevolve":  CONFIGS_DIR / "eael_no_plannerevolve.yaml",  # M9
    "eael_preset_tools":      CONFIGS_DIR / "eael_preset_tools.yaml",   # M10
    "eael_no_skills":         CONFIGS_DIR / "eael_no_skills.yaml",      # M11
    # Incremental build steps (tool_only → full EAEL)
    "tool_memory":            INCREMENTAL_DIR / "step2_tool_memory.yaml",
    "tool_memory_reflect":    INCREMENTAL_DIR / "step3_tool_memory_reflect.yaml",
    "tool_memory_reflect_skills": INCREMENTAL_DIR / "step4_tool_memory_reflect_skills.yaml",
    "eael_incremental":       INCREMENTAL_DIR / "step5_eael_full.yaml",
    # Credit assignment controlled experiments
    "step5_uniform_credit":   INCREMENTAL_DIR / "step5_uniform_credit.yaml",
    "step5_no_pertool":       INCREMENTAL_DIR / "step5_no_pertool.yaml",
    "step5_no_warmup":        INCREMENTAL_DIR / "step5_no_warmup.yaml",
    "step5_fcc_tuned":        INCREMENTAL_DIR / "step5_fcc_tuned.yaml",
    "step5_curriculum":       INCREMENTAL_DIR / "step5_curriculum.yaml",
    "step5_llm_credit":       INCREMENTAL_DIR / "step5_llm_credit.yaml",
    "step5_curriculum_llm":   INCREMENTAL_DIR / "step5_curriculum_llm.yaml",
    # Incremental credit isolation experiments
    "step5a_credit_only":     INCREMENTAL_DIR / "step5a_credit_only.yaml",
    "step5b_credit_planner":  INCREMENTAL_DIR / "step5b_credit_planner.yaml",
    "tool_memory_diffmem":    INCREMENTAL_DIR / "step2b_tool_memory_diffmem.yaml",
    "step5_llm_uni_a03":      INCREMENTAL_DIR / "step5_llm_uni_a03.yaml",
    "step5_llm_uni_a05":      INCREMENTAL_DIR / "step5_llm_uni_a05.yaml",
    "step5_llm_uni_a07":      INCREMENTAL_DIR / "step5_llm_uni_a07.yaml",
}

# Dataset override configs
DATASET_CONFIGS = {
    "d1": CONFIGS_DIR / "dataset_d1.yaml",
    "d2": CONFIGS_DIR / "dataset_d2.yaml",
    "d3": CONFIGS_DIR / "dataset_d3.yaml",
    "d4": CONFIGS_DIR / "dataset_d4.yaml",
    "d_small": CONFIGS_DIR / "dataset_d_small.yaml",
    "d_large": CONFIGS_DIR / "dataset_d_large.yaml",
    "d_full": CONFIGS_DIR / "dataset_d_full.yaml",
}

# Canonical 5-seed set for all paper results
PAPER_SEEDS = [42, 123, 456, 789, 1024]

# Run lists (extracted for composition)
_E1_RUNS = [
    (method, dataset, seed)
    for method in ["stateless", "memory_only", "tool_only", "ael_full", "credit_ablation", "eael_full"]
    for dataset in ["d1", "d2", "d3", "d4"]
    for seed in [42, 123, 456]
    if not (method == "credit_ablation" and dataset in ("d3", "d4"))
]

_E2_RUNS = [
    (method, "d2", seed)
    for method in ["eael_full", "eael_no_reflect", "eael_no_coldstart",
                   "eael_no_plannerevolve", "eael_preset_tools", "eael_no_skills"]
    for seed in [42, 123, 456]
]

# Paper 5-seed runs: all methods on D-full with 5 seeds for consistent N
_PAPER_AEL_RUNS = [
    (method, "d_full", seed)
    for method in [
        "stateless", "tool_only",
        "tool_memory", "tool_memory_reflect",
        "tool_memory_reflect_skills", "eael_incremental",
    ]
    for seed in PAPER_SEEDS
]

_PAPER_CREDIT_RUNS = [
    (method, "d_full", seed)
    for method in [
        "step5_uniform_credit", "step5_no_pertool",
        "step5_no_warmup", "step5_fcc_tuned",
        "step5_llm_credit",
    ]
    for seed in PAPER_SEEDS
]

_PAPER_ABLATION_RUNS = [
    (method, "d_full", seed)
    for method in [
        "eael_no_reflect", "eael_no_plannerevolve",
    ]
    for seed in PAPER_SEEDS
]

_ALL_RUNS = list(dict.fromkeys(_E1_RUNS + _E2_RUNS))  # dedup, preserve order

# Experiment definitions
EXPERIMENTS = {
    "e1": {
        "description": "E1: Main comparison (6 methods x 4 datasets x 3 seeds = 66 runs)",
        "runs": _E1_RUNS,
    },
    "e2": {
        "description": "E2: EAEL ablation on D2 (6 configs x 3 seeds = 18 runs)",
        "runs": _E2_RUNS,
    },
    "all": {
        "description": "E1+E2 combined: 81 unique runs (3 shared eael_full_d2 deduplicated)",
        "runs": _ALL_RUNS,
    },
    "smoke": {
        "description": "Smoke test: M1 and M6 on D2 with 20 episodes",
        "runs": [
            ("stateless", "d2", 42),
            ("eael_full", "d2", 42),
        ],
        "max_episodes": 20,
    },
    "smoke_1h": {
        "description": "Hourly smoke test: M1 on D-small with 20 episodes",
        "runs": [
            ("stateless", "d_small", 42),
        ],
        "max_episodes": 20,
    },
    "e1_1h": {
        "description": "E1 hourly: 6 methods on D-small × 3 seeds = 18 runs",
        "runs": [
            (method, "d_small", seed)
            for method in ["stateless", "memory_only", "tool_only", "ael_full", "credit_ablation", "eael_full"]
            for seed in [42, 123, 456]
        ],
    },
    # Incremental build: tool_only → +memory → +reflect → +skills → +credit
    "incremental_small": {
        "description": "Incremental build on D-small: 4 steps × 3 seeds = 12 runs",
        "runs": [
            (method, "d_small", seed)
            for method in ["tool_memory", "tool_memory_reflect", "tool_memory_reflect_skills", "eael_incremental"]
            for seed in [42, 123, 456]
        ],
    },
    "incremental_large": {
        "description": "Incremental build on D-large: 4 steps × 3 seeds = 12 runs",
        "runs": [
            (method, "d_large", seed)
            for method in ["tool_memory", "tool_memory_reflect", "tool_memory_reflect_skills", "eael_incremental"]
            for seed in [42, 123, 456]
        ],
    },
    # Portfolio allocation experiments (1 LLM call per bar, allocates across all tickers)
    "portfolio_small": {
        "description": "Portfolio allocation on D-small: 4 incremental + 2 baselines × 3 seeds = 18 runs",
        "runner": "portfolio",
        "runs": [
            (method, "d_small", seed)
            for method in [
                "stateless", "tool_only",
                "tool_memory", "tool_memory_reflect",
                "tool_memory_reflect_skills", "eael_incremental",
            ]
            for seed in [42, 123, 456]
        ],
    },
    "portfolio_large": {
        "description": "Portfolio allocation on D-large: 4 incremental + 2 baselines × 3 seeds = 18 runs",
        "runner": "portfolio",
        "runs": [
            (method, "d_large", seed)
            for method in [
                "stateless", "tool_only",
                "tool_memory", "tool_memory_reflect",
                "tool_memory_reflect_skills", "eael_incremental",
            ]
            for seed in [42, 123, 456]
        ],
    },
    "portfolio_full": {
        "description": "Portfolio on D-full: diverse train (bull+bear+flat), split test (bear W11 + bull W12)",
        "runner": "portfolio",
        "runs": [
            (method, "d_full", seed)
            for method in [
                "stateless", "tool_only",
                "tool_memory", "tool_memory_reflect",
                "tool_memory_reflect_skills", "eael_incremental",
            ]
            for seed in [42, 123, 456]
        ],
    },
    "credit_study": {
        "description": "Credit assignment controlled study: 4 single-variable configs x 3 seeds = 12 runs",
        "runner": "portfolio",
        "runs": [
            (method, "d_full", seed)
            for method in [
                "step5_uniform_credit", "step5_no_pertool",
                "step5_no_warmup", "step5_fcc_tuned",
                "step5_curriculum", "step5_llm_credit",
                "step5_curriculum_llm",
            ]
            for seed in [42, 123, 456]
        ],
    },
    # ---- Paper 5-seed experiments: consistent N=5 across all methods ----
    "paper_ael": {
        "description": "Paper AEL incremental build on D-full: 6 methods × 5 seeds = 30 runs",
        "runner": "portfolio",
        "runs": _PAPER_AEL_RUNS,
    },
    "paper_credit": {
        "description": "Paper credit study on D-full: 5 configs × 5 seeds = 25 runs",
        "runner": "portfolio",
        "runs": _PAPER_CREDIT_RUNS,
    },
    "paper_ablation": {
        "description": "Paper ablation on D-full: 2 ablation configs × 5 seeds = 10 runs",
        "runner": "portfolio",
        "runs": _PAPER_ABLATION_RUNS,
    },
    "paper_all": {
        "description": "All paper runs: AEL + credit + ablation = 65 portfolio runs",
        "runner": "portfolio",
        "runs": list(dict.fromkeys(_PAPER_AEL_RUNS + _PAPER_CREDIT_RUNS + _PAPER_ABLATION_RUNS)),
    },
    # Only new seeds (789, 1024) — trust existing 42/123/456 results
    "paper_new_seeds": {
        "description": "New seeds only: AEL + credit + ablation × 2 seeds = 26 portfolio runs",
        "runner": "portfolio",
        "runs": [
            (method, "d_full", seed)
            for method in [
                "stateless", "tool_only",
                "tool_memory", "tool_memory_reflect",
                "tool_memory_reflect_skills", "eael_incremental",
                "step5_uniform_credit", "step5_no_pertool",
                "step5_no_warmup", "step5_fcc_tuned",
                "step5_llm_credit",
                "eael_no_reflect", "eael_no_plannerevolve",
            ]
            for seed in [789, 1024]
        ],
    },
}


def _get_git_version() -> str:
    """Get current git tag or short commit hash for log versioning."""
    try:
        tag = subprocess.run(["git", "describe", "--tags", "--exact-match"],
                             capture_output=True, text=True, timeout=5)
        if tag.returncode == 0:
            return tag.stdout.strip()
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, timeout=5)
        return commit.stdout.strip() if commit.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


_GIT_VERSION = _get_git_version()


def merge_configs(method_path: Path, dataset_path: Path) -> dict:
    """Load method config and overlay dataset benchmark settings."""
    with open(method_path) as f:
        config = yaml.safe_load(f)

    with open(dataset_path) as f:
        dataset = yaml.safe_load(f)

    # Overlay benchmark section
    if "benchmark" in dataset:
        config["benchmark"] = {**config.get("benchmark", {}), **dataset["benchmark"]}

    return config


def make_run_config(
    method: str,
    dataset: str,
    seed: int,
    max_episodes: int | None = None,
) -> dict:
    """Create a merged config dict for a specific (method, dataset, seed) triple."""
    method_path = METHOD_CONFIGS[method]
    dataset_path = DATASET_CONFIGS[dataset]

    config = merge_configs(method_path, dataset_path)
    config["seed"] = seed
    config["name"] = f"{method}_{dataset}_s{seed}_{_GIT_VERSION}"
    config["log_dir"] = str(PROJECT_ROOT / "logs" / config["name"])

    if max_episodes is not None:
        config["num_episodes"] = max_episodes

    return config


def run_single(
    method: str,
    dataset: str,
    seed: int,
    max_episodes: int | None = None,
    runner_type: str | None = None,
) -> dict:
    """Execute a single experiment run.

    Writes merged config to a temp YAML, then calls the appropriate runner script.
    Uses run_portfolio.py for runner_type="portfolio", otherwise run_experiment.py.
    Returns a result summary dict.
    """
    config = make_run_config(method, dataset, seed, max_episodes)
    run_name = config["name"]
    if runner_type == "portfolio":
        run_name = f"pf_{run_name}"
        config["name"] = run_name
        config["log_dir"] = str(PROJECT_ROOT / "logs" / run_name)
    log_dir = Path(config["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    # Check if already completed
    result_files = list(log_dir.glob("backtest_*.json"))
    if result_files:
        logger.info(f"[SKIP] {run_name}: already completed")
        return {"name": run_name, "status": "skipped"}

    # Write merged config
    config_path = log_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    logger.info(f"[START] {run_name}")
    start_time = time.time()

    try:
        # Select runner script based on runner_type
        if runner_type == "portfolio":
            runner_script = PROJECT_ROOT / "experiments" / "run_portfolio.py"
        else:
            runner_script = PROJECT_ROOT / "experiments" / "run_experiment.py"

        # Run via subprocess to isolate each experiment
        cmd = [
            sys.executable,
            str(runner_script),
            "--config", str(config_path),
        ]
        if max_episodes:
            cmd.extend(["--max-episodes", str(max_episodes)])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=28800,  # 8 hour timeout per run (Bedrock + high parallelism)
            env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
        )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            logger.info(f"[DONE] {run_name} ({elapsed:.0f}s)")
            return {"name": run_name, "status": "success", "elapsed": elapsed}
        else:
            logger.error(f"[FAIL] {run_name}: {result.stderr[-500:]}")
            # Save error log
            (log_dir / "error.log").write_text(result.stderr)
            return {"name": run_name, "status": "failed", "error": result.stderr[-200:]}

    except subprocess.TimeoutExpired:
        logger.error(f"[TIMEOUT] {run_name}")
        return {"name": run_name, "status": "timeout"}
    except Exception as e:
        logger.error(f"[ERROR] {run_name}: {e}")
        return {"name": run_name, "status": "error", "error": str(e)}


def _summarize_results():
    """Print portfolio results table from existing log dirs."""
    import re
    from collections import defaultdict
    import numpy as np

    logs_dir = PROJECT_ROOT / "logs"
    method_results = defaultdict(list)

    for d in sorted(logs_dir.iterdir()):
        if not d.is_dir() or not d.name.startswith("pf_"):
            continue
        bt = list(d.glob("backtest_*.json"))
        if not bt:
            continue
        data = json.loads(bt[0].read_text())
        m = re.match(r"^pf_(.+?)_(d_\w+)_s(\d+)(?:_.*)?$", d.name)
        method = m.group(1) if m else d.name
        metrics = data.get("metrics", {})
        phase = data.get("phase_metrics", {})
        method_results[method].append({
            "name": d.name,
            "sharpe": metrics.get("sharpe_ratio", 0),
            "total_return": metrics.get("total_return", 0),
            "max_dd": metrics.get("max_drawdown", 0),
            "test_sharpe": phase.get("test", {}).get("sharpe", 0),
            "test_return": phase.get("test", {}).get("total_return", 0),
            "train_sharpe": phase.get("train", {}).get("sharpe", 0),
            "val_sharpe": phase.get("val", {}).get("sharpe", 0),
        })

    if not method_results:
        print("No portfolio results found")
        return

    # Print individual runs
    print(f"{'Run':<55} {'Sharpe':>7} {'TestSR':>7} {'TestRet%':>9} {'MaxDD%':>8}")
    print("-" * 90)
    for method in sorted(method_results.keys()):
        for r in method_results[method]:
            print(f"{r['name']:<55} {r['sharpe']:>7.2f} {r['test_sharpe']:>7.2f} "
                  f"{r['test_return']:>+8.2%} {r['max_dd']:>+7.2%}")

    # Print averages
    print(f"\n{'='*90}")
    print(f"{'Method':<30} {'Sharpe':>7} {'TrainSR':>8} {'ValSR':>7} {'TestSR':>7} {'TestRet%':>9} {'MaxDD%':>8} {'N':>3}")
    print("-" * 90)
    for method in ["stateless", "tool_only", "tool_memory", "tool_memory_reflect",
                    "tool_memory_reflect_skills", "eael_incremental"]:
        runs = method_results.get(method, [])
        if not runs:
            continue
        n = len(runs)
        print(f"{method:<30} "
              f"{np.mean([r['sharpe'] for r in runs]):>7.2f} "
              f"{np.mean([r['train_sharpe'] for r in runs]):>8.2f} "
              f"{np.mean([r['val_sharpe'] for r in runs]):>7.2f} "
              f"{np.mean([r['test_sharpe'] for r in runs]):>7.2f} "
              f"{np.mean([r['test_return'] for r in runs]):>+8.2%} "
              f"{np.mean([r['max_dd'] for r in runs]):>+7.2%} "
              f"{n:>3}")

    # Also dump as JSON for easy consumption
    summary = {}
    for method, runs in method_results.items():
        summary[method] = {
            "avg_sharpe": float(np.mean([r["sharpe"] for r in runs])),
            "avg_test_sharpe": float(np.mean([r["test_sharpe"] for r in runs])),
            "avg_test_return": float(np.mean([r["test_return"] for r in runs])),
            "avg_max_dd": float(np.mean([r["max_dd"] for r in runs])),
            "n_seeds": len(runs),
        }
    print(f"\n=== JSON ===")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Run EAEL experiment suite")
    parser.add_argument(
        "--experiment", choices=list(EXPERIMENTS.keys()),
        help="Predefined experiment set to run"
    )
    parser.add_argument("--method", choices=list(METHOD_CONFIGS.keys()),
                        help="Single method to run")
    parser.add_argument("--dataset", choices=list(DATASET_CONFIGS.keys()),
                        help="Single dataset to use")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (for single runs)")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Multiple seeds (for single method+dataset)")
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="Cap on episodes per run")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of parallel workers")
    parser.add_argument("--runner", choices=["experiment", "portfolio"], default=None,
                        help="Runner type for single runs (default: experiment)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be run without executing")
    parser.add_argument("--summarize", action="store_true",
                        help="Summarize existing portfolio results and exit")
    args = parser.parse_args()

    if args.summarize:
        _summarize_results()
        return

    # Build run list: (method, dataset, seed, max_ep, runner_type)
    runs = []

    if args.experiment:
        exp = EXPERIMENTS[args.experiment]
        logger.info(f"Experiment: {exp['description']}")
        max_ep = args.max_episodes or exp.get("max_episodes")
        runner_type = exp.get("runner")  # None for default, "portfolio" for portfolio
        for method, dataset, seed in exp["runs"]:
            runs.append((method, dataset, seed, max_ep, runner_type))
    elif args.method and args.dataset:
        seeds = args.seeds or [args.seed]
        runner_type = args.runner
        for seed in seeds:
            runs.append((args.method, args.dataset, seed, args.max_episodes, runner_type))
    else:
        parser.error("Specify --experiment or both --method and --dataset")

    logger.info(f"Total runs: {len(runs)}, parallel workers: {args.parallel}")

    if args.dry_run:
        for method, dataset, seed, max_ep, rt in runs:
            prefix = "pf_" if rt == "portfolio" else ""
            name = f"{prefix}{method}_{dataset}_s{seed}"
            ep_str = f" (max_episodes={max_ep})" if max_ep else ""
            runner_str = f" [runner={rt}]" if rt else ""
            print(f"  {name}{ep_str}{runner_str}")
        print(f"\nTotal: {len(runs)} runs")
        return

    # Progress tracking
    PROGRESS_FILE = PROJECT_ROOT / "logs" / "progress.json"
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)

    def _write_progress(data: dict):
        tmp = PROGRESS_FILE.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, default=str)
        tmp.rename(PROGRESS_FILE)

    progress = {
        "experiment": args.experiment,
        "total": len(runs),
        "completed": 0, "skipped": 0, "failed": 0,
        "active": 0, "pending": len(runs),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "last_updated": None, "finished_at": None,
        "results": [],
    }

    def _update_progress(result: dict):
        status = result.get("status", "unknown")
        if status == "success":
            progress["completed"] += 1
        elif status == "skipped":
            progress["skipped"] += 1
        else:
            progress["failed"] += 1
        progress["results"].append(result)
        done = progress["completed"] + progress["skipped"] + progress["failed"]
        progress["active"] = min(args.parallel, progress["total"] - done)
        progress["pending"] = max(0, progress["total"] - done - progress["active"])
        progress["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        _write_progress(progress)

    # Execute
    all_results = []
    start_time = time.time()

    if args.parallel <= 1:
        progress["active"] = 1
        progress["pending"] = len(runs) - 1
        _write_progress(progress)
        for method, dataset, seed, max_ep, rt in runs:
            result = run_single(method, dataset, seed, max_ep, runner_type=rt)
            all_results.append(result)
            _update_progress(result)
    else:
        progress["active"] = min(args.parallel, len(runs))
        progress["pending"] = max(0, len(runs) - args.parallel)
        _write_progress(progress)
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            futures = {
                executor.submit(run_single, m, d, s, ep, rt): f"{m}_{d}_s{s}"
                for m, d, s, ep, rt in runs
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    result = future.result()
                    all_results.append(result)
                except Exception as e:
                    logger.error(f"[CRASH] {name}: {e}")
                    result = {"name": name, "status": "crash", "error": str(e)}
                    all_results.append(result)
                _update_progress(result)

    total_time = time.time() - start_time

    # Final progress update
    progress["active"] = 0
    progress["pending"] = 0
    progress["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    progress["elapsed_seconds"] = int(total_time)
    _write_progress(progress)

    # Summary
    success = sum(1 for r in all_results if r["status"] == "success")
    skipped = sum(1 for r in all_results if r["status"] == "skipped")
    failed = sum(1 for r in all_results if r["status"] in ("failed", "error", "crash", "timeout"))

    logger.info(f"\n{'='*60}")
    logger.info(f"Experiment suite complete in {total_time:.0f}s")
    logger.info(f"  Success: {success}, Skipped: {skipped}, Failed: {failed}")

    if failed > 0:
        logger.info("Failed runs:")
        for r in all_results:
            if r["status"] in ("failed", "error", "crash", "timeout"):
                logger.info(f"  {r['name']}: {r['status']} — {r.get('error', '')[:100]}")

    # Save run manifest
    manifest_path = PROJECT_ROOT / "logs" / "run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump({
            "experiment": args.experiment,
            "total_time": total_time,
            "results": all_results,
        }, f, indent=2, default=str)
    logger.info(f"Manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
