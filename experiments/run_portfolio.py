"""Config-driven portfolio experiment runner for AEL framework.

Runs the portfolio allocation task: one LLM call per bar allocates weights
across all tickers simultaneously. Uses the same config system and
build_components() as run_experiment.py but swaps BacktestRunner for
PortfolioRunner.

Usage:
    python experiments/run_portfolio.py --config experiments/configs/ael_full.yaml
    python experiments/run_portfolio.py --config configs/eael_full.yaml --seed 42
    python experiments/run_portfolio.py --config configs/eael_full.yaml --seeds 42,123,456
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ael.config import ExperimentConfig
from ael.benchmarks.portfolio_runner import PortfolioRunner
from experiments.run_experiment import build_components

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_portfolio_experiment(
    config_path: str,
    seed: int | None = None,
    model: str | None = None,
    max_episodes: int | None = None,
):
    """Load config and run a full portfolio allocation backtest."""
    config = ExperimentConfig.from_yaml(config_path)

    if seed is not None:
        config.seed = seed
        # Per-seed log directory to avoid overwrites
        config.log_dir = f"{config.log_dir}/seed_{seed}"

    if max_episodes is not None:
        config.num_episodes = max_episodes

    np.random.seed(config.seed)

    effective_model = model or config.llm_model
    logger.info(f"Running portfolio experiment: {config.name} (seed={config.seed})")
    logger.info(f"  Tickers:        {config.benchmark.tickers}")
    logger.info(f"  Planner evolve: {config.planner.evolve}")
    logger.info(f"  Tools evolve:   {config.tools.evolve}")
    logger.info(f"  Memory evolve:  {config.memory.evolve}")
    logger.info(f"  Memory backend: {config.memory.backend}")
    logger.info(f"  Credit method:  {config.credit.method}")
    logger.info(f"  LLM model:      {effective_model}")

    # Build components (reuse from run_experiment)
    loop = build_components(config, model=model)

    # Determine predict_fn: pass through to signal LLM usage vs equal-weight
    # predict_fn=True signals "use LLM", predict_fn=None for "equal-weight baseline"
    predict_fn = loop.predict_fn if (effective_model and effective_model != "heuristic") else None

    # Run portfolio backtest
    ep_cap = max_episodes or config.num_episodes
    runner = PortfolioRunner(config=config, evolution_loop=loop)
    results = runner.run(
        predict_fn=predict_fn,
        max_episodes=ep_cap,
    )

    # Print summary
    metrics = results.get("metrics", {})
    phase_metrics = results.get("phase_metrics", {})
    portfolio_final = results.get("portfolio_final", {})

    logger.info(f"\n{'='*60}")
    logger.info(f"Portfolio Experiment: {config.name}")
    logger.info(f"Episodes:       {results['total_episodes']}")
    logger.info(f"Total Return:   {metrics.get('total_return', 0):+.4f}")
    logger.info(f"Sharpe Ratio:   {metrics.get('sharpe_ratio', 0):.3f}")
    logger.info(f"Max Drawdown:   {metrics.get('max_drawdown', 0):.4f}")
    logger.info(f"Avg Turnover:   {metrics.get('avg_turnover', 0):.4f}")
    logger.info(f"Final Value:    {portfolio_final.get('total_value', 1.0):.4f}")

    if phase_metrics:
        logger.info(f"\nPhase breakdown:")
        for phase, pm in phase_metrics.items():
            logger.info(
                f"  {phase:5s}: return={pm.get('total_return', 0):+.4f} "
                f"sharpe={pm.get('sharpe', 0):.2f} "
                f"bars={pm.get('n_bars', 0)}"
            )

    logger.info(f"{'='*60}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run AEL portfolio experiment")
    parser.add_argument("config", nargs="?", type=str, default=None,
                        help="Path to YAML config file")
    parser.add_argument("--config", dest="config_flag", type=str, default=None,
                        help="Path to YAML config file (alternative to positional)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated seeds for multi-run (e.g., 42,123,456)")
    parser.add_argument("--model", type=str, default=None,
                        help="litellm model string (e.g., gpt5-nano, gpt-4o-mini). "
                             "Default: config llm_model or heuristic")
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="Override max episodes (for debugging)")
    args = parser.parse_args()

    # Resolve config path from either positional or --config flag
    args.config = args.config or args.config_flag
    if not args.config:
        parser.error("Config path required (positional or --config)")

    max_ep = args.max_episodes

    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
        all_results = []
        for seed in seeds:
            results = run_portfolio_experiment(
                args.config, seed=seed, model=args.model, max_episodes=max_ep,
            )
            all_results.append(results)
        # Print aggregate
        sharpes = [
            r["metrics"]["sharpe_ratio"]
            for r in all_results if "sharpe_ratio" in r.get("metrics", {})
        ]
        total_returns = [
            r["metrics"]["total_return"]
            for r in all_results if "total_return" in r.get("metrics", {})
        ]
        if sharpes:
            logger.info(f"\nAggregate ({len(seeds)} seeds):")
            logger.info(f"  Total Return: {np.mean(total_returns):+.4f} +/- {np.std(total_returns):.4f}")
            logger.info(f"  Sharpe Ratio: {np.mean(sharpes):.3f} +/- {np.std(sharpes):.3f}")
    else:
        run_portfolio_experiment(
            args.config, seed=args.seed, model=args.model, max_episodes=max_ep,
        )


if __name__ == "__main__":
    main()
