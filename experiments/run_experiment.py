"""Config-driven experiment runner for AEL framework.

V3: Supports XMem memory backend and unified litellm predictor.
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
from ael.evolution.loop import EvolutionLoop
from ael.evolution.meta_controller import MetaController
from ael.llm_predictor import create_llm_predictor, create_heuristic_predictor
from ael.memory import create_memory_backend
from ael.planner.pool import PlannerPool
from ael.tools.finance_tools import register_all_finance_tools
from ael.tools.registry import ToolRegistry
from ael.benchmarks.backtest import BacktestRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def build_components(config: ExperimentConfig, model: str | None = None):
    """Build all AEL components from experiment config.

    Args:
        model: litellm model string (e.g., "gpt-4o-mini", "claude-sonnet-4-20250514").
               If None or "heuristic", uses heuristic predictor (no LLM).
    """

    # Tool registry
    tool_registry = ToolRegistry()
    register_all_finance_tools(tool_registry)

    # Planner pool (only register enabled planners)
    planner_pool = PlannerPool(
        alpha=config.planner.linucb_alpha,
        enabled_planners=config.planner.enabled_planners,
    )

    # Memory store (local or XMem backend)
    memory_store = create_memory_backend(config.memory)

    # Meta-controller
    meta = MetaController(
        planner_names=config.planner.enabled_planners,
        evolve_planner=config.planner.evolve,
        evolve_tools=config.tools.evolve,
        evolve_memory=config.memory.evolve,
        per_tool_selection=config.tools.per_tool_selection,
        all_tool_names=tool_registry.all_names if config.tools.per_tool_selection else None,
    )

    # Predictor
    effective_model = model or config.llm_model
    if effective_model and effective_model != "heuristic":
        predict_fn = create_llm_predictor(
            model=effective_model,
            temperature=config.llm_temperature,
            max_tokens=config.max_tokens,
        )
    else:
        predict_fn = create_heuristic_predictor()

    # Evolution loop
    loop = EvolutionLoop(
        config=config,
        tool_registry=tool_registry,
        planner_pool=planner_pool,
        memory_store=memory_store,
        meta_controller=meta,
        predict_fn=predict_fn,
    )

    return loop


def run_experiment(
    config_path: str,
    seed: int | None = None,
    model: str | None = None,
    max_episodes: int | None = None,
):
    """Load config and run a full backtest experiment."""
    config = ExperimentConfig.from_yaml(config_path)

    if seed is not None:
        config.seed = seed
        # Per-seed log directory to avoid overwrites
        config.log_dir = f"{config.log_dir}/seed_{seed}"

    if max_episodes is not None:
        config.num_episodes = max_episodes

    np.random.seed(config.seed)

    effective_model = model or config.llm_model
    logger.info(f"Running experiment: {config.name} (seed={config.seed})")
    logger.info(f"  Planner evolve: {config.planner.evolve}")
    logger.info(f"  Tools evolve:   {config.tools.evolve}")
    logger.info(f"  Memory evolve:  {config.memory.evolve}")
    logger.info(f"  Memory backend: {config.memory.backend}")
    logger.info(f"  Credit method:  {config.credit.method}")
    logger.info(f"  LLM model:      {effective_model}")

    # Build components
    loop = build_components(config, model=model)

    # Run backtest
    ep_cap = max_episodes or config.num_episodes
    runner = BacktestRunner(config=config, evolution_loop=loop)
    results = runner.run(
        predict_fn=loop.predict_fn,
        max_episodes=ep_cap,
    )

    # Print summary
    metrics = results.get("metrics", {})
    overall = metrics.get("overall", {})
    logger.info(f"\n{'='*60}")
    logger.info(f"Experiment: {config.name}")
    logger.info(f"Episodes:   {results['total_episodes']}")
    logger.info(f"Dir Acc:    {overall.get('directional_accuracy', 0):.3f}")
    logger.info(f"MAPE:       {overall.get('mape', 0):.4f}")
    logger.info(f"W-Acc:      {overall.get('weighted_accuracy', 0):.3f}")
    logger.info(f"{'='*60}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run AEL experiment")
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
            results = run_experiment(args.config, seed=seed, model=args.model, max_episodes=max_ep)
            all_results.append(results)
        # Print aggregate
        overall_accs = [
            r["metrics"]["overall"]["directional_accuracy"]
            for r in all_results if "overall" in r.get("metrics", {})
        ]
        if overall_accs:
            logger.info(f"\nAggregate ({len(seeds)} seeds):")
            logger.info(f"  Dir Acc: {np.mean(overall_accs):.3f} ± {np.std(overall_accs):.3f}")
    else:
        run_experiment(args.config, seed=args.seed, model=args.model, max_episodes=max_ep)


if __name__ == "__main__":
    main()
