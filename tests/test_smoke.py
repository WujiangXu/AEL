"""End-to-end smoke test for the AEL framework.

Runs 3 episodes on 1 ticker using the heuristic predictor (no LLM needed).
Verifies: tool execution, planning, memory, credit assignment, and metrics.

Usage:
    # activate your virtual environment
    python tests/test_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from ael.config import ExperimentConfig
from ael.evolution.credit import structural_credit, shapley_credit, uniform_credit
from ael.evolution.loop import EvolutionLoop
from ael.evolution.meta_controller import MetaController, TaskFeatures
from ael.llm_predictor import create_heuristic_predictor
from ael.memory.store import MemoryStore
from ael.planner.pool import PlannerPool
from ael.tools.finance_tools import register_all_finance_tools
from ael.tools.registry import ToolRegistry
from ael.benchmarks.finance import (
    PredictionTask,
    get_realized_returns,
    evaluate_prediction,
    compute_outcome_score,
    compute_metrics,
)


def test_tool_execution():
    """Test that all tools can execute on cached data."""
    print("=" * 60)
    print("TEST 1: Tool Execution")
    print("=" * 60)

    registry = ToolRegistry()
    register_all_finance_tools(registry)

    ticker = "AAPL"
    results = {}

    for tool_name in registry.all_names:
        if tool_name == "score_composite_signal":
            # Needs prior outputs
            result = registry.call(tool_name, ticker=ticker, tool_outputs=results)
        else:
            result = registry.call(tool_name, ticker=ticker)

        status = "OK" if result.success else f"FAIL: {result.error}"
        print(f"  {tool_name:30s} {status} ({result.latency:.2f}s)")
        if result.success:
            results[tool_name] = result.output

    n_ok = sum(1 for r in [registry.call(t, ticker=ticker) for t in registry.all_names[:5]] if r.success)
    assert n_ok >= 3, f"Too many tool failures: only {n_ok}/5 succeeded"
    print(f"\n  PASSED: {len(results)}/{len(registry.all_names)} tools succeeded\n")
    return results


def test_planner_pool():
    """Test all planners generate valid plans."""
    print("=" * 60)
    print("TEST 2: Planner Pool")
    print("=" * 60)

    pool = PlannerPool()
    registry = ToolRegistry()
    register_all_finance_tools(registry)
    tools = registry.all_names

    task = {"ticker": "AAPL", "date": "2025-01-15", "task_type": "price_prediction"}

    for name in pool.all_names:
        planner = pool.get(name)
        steps = planner.plan(task, tools)
        print(f"  {name:20s} → {len(steps)} steps")
        assert len(steps) > 0, f"Planner {name} produced no steps"

    # Test LinUCB selection
    features = np.array([0.5, 0.3, 0.8, 0.9, 0.6, 1.0, 1.0])
    selected = pool.select(features)
    print(f"  LinUCB selected: {selected}")
    assert selected in pool.all_names

    print("  PASSED\n")


def test_memory_store():
    """Test memory write, retrieve, and eviction."""
    print("=" * 60)
    print("TEST 3: Memory Store")
    print("=" * 60)

    store = MemoryStore(retrieval_top_k=3, max_entries_per_tier=5)

    # Add entries
    store.add_episodic("AAPL went up 5% after strong earnings", {"ticker": "AAPL"}, quality=0.8)
    store.add_episodic("MSFT declined on cloud growth concerns", {"ticker": "MSFT"}, quality=0.6)
    store.add_semantic("Tech stocks respond strongly to earnings beats", {"sector": "Technology"}, quality=0.9)
    store.add_procedural("For high-vol stocks, weight technicals over DCF", {"task_type": "price_prediction"}, quality=0.7)

    print(f"  Store size: {store.size}")
    print(f"  Tier counts: {store.tier_counts()}")
    assert store.size == 4

    # Retrieve
    hits = store.retrieve({"ticker": "AAPL", "task_type": "price_prediction"})
    print(f"  Retrieved for AAPL: {len(hits)} hits")
    assert len(hits) > 0

    # Feedback
    store.mark_useful(hits[0].memory_id, True)
    print(f"  Marked {hits[0].memory_id} as useful")

    print("  PASSED\n")


def test_credit_assignment():
    """Test FCC credit computation."""
    print("=" * 60)
    print("TEST 4: Credit Assignment")
    print("=" * 60)

    from ael.types import TrajectoryRecord, ToolCall, MemoryHit, PlanStep

    traj = TrajectoryRecord(
        plan_steps=[
            PlanStep(step_id=0, action="call:get_price_history", rationale="test", success=True),
            PlanStep(step_id=1, action="call:compute_technicals", rationale="test", success=True),
            PlanStep(step_id=2, action="call:run_dcf_model", rationale="test", success=False),
            PlanStep(step_id=3, action="synthesize", rationale="test", success=True),
        ],
        tools_invoked=[
            ToolCall(tool_name="get_price_history", inputs={}, output={}, success=True, latency=0.1, cost=0),
            ToolCall(tool_name="compute_technicals", inputs={}, output={}, success=True, latency=0.2, cost=0),
            ToolCall(tool_name="run_dcf_model", inputs={}, output=None, success=False, latency=0.1, cost=0, error="no data"),
        ],
        memory_retrievals=[
            MemoryHit(memory_id="m1", tier="episodic", content="AAPL went up", relevance_score=0.8, was_useful=True),
        ],
    )

    # Structural credit
    struct = structural_credit(traj)
    print(f"  Structural: {struct}")
    assert "planner" in struct and "tools" in struct and "memory" in struct

    # Uniform credit
    uni = uniform_credit()
    print(f"  Uniform: {uni}")
    assert abs(sum(uni.values()) - 1.0) < 0.01

    # Shapley (with mock outcome_fn)
    def mock_outcome_fn(coalition):
        score = 0.0
        if "planner" in coalition:
            score += 0.3
        if "tools" in coalition:
            score += 0.4
        if "memory" in coalition:
            score += 0.1
        return score

    shap = shapley_credit(mock_outcome_fn)
    print(f"  Shapley: {shap}")
    assert abs(sum(shap.values()) - mock_outcome_fn(frozenset(["planner", "tools", "memory"]))) < 0.01

    print("  PASSED\n")


def test_evolution_loop():
    """Test full evolution loop: episode → credit → memory update."""
    print("=" * 60)
    print("TEST 5: Evolution Loop (3 episodes)")
    print("=" * 60)

    config = ExperimentConfig(
        name="smoke_test",
        num_episodes=3,
        log_dir="logs/smoke_test",
    )
    config.credit.method = "structural"

    registry = ToolRegistry()
    register_all_finance_tools(registry)
    pool = PlannerPool()
    memory = MemoryStore()
    meta = MetaController(planner_names=pool.all_names)
    predict_fn = create_heuristic_predictor()

    loop = EvolutionLoop(
        config=config,
        tool_registry=registry,
        planner_pool=pool,
        memory_store=memory,
        meta_controller=meta,
        predict_fn=predict_fn,
    )

    tickers = ["AAPL", "MSFT", "NVDA"]
    date = "2025-01-15"

    for i, ticker in enumerate(tickers):
        features = TaskFeatures(sector_id=0, volatility=0.3, market_cap_log=12.0)
        task = {"ticker": ticker, "date": date, "task_type": "price_prediction",
                "horizons": [1, 7, 14]}

        # Run episode
        traj = loop.run_episode(task, features)
        print(f"  Episode {i}: {ticker}, planner={traj.planner_id}, "
              f"tools_ok={sum(1 for t in traj.tools_invoked if t.success)}, "
              f"time={traj.wall_time:.2f}s")

        # Get realized returns (from cache)
        realized = get_realized_returns(ticker, date, [1, 7, 14])
        if realized:
            # Make prediction
            tool_outputs = {tc.tool_name: tc.output for tc in traj.tools_invoked if tc.success}
            prediction = predict_fn(task, tool_outputs, traj.memory_retrievals)

            # Evaluate
            evals = evaluate_prediction(prediction, realized)
            score = compute_outcome_score(evals)
            print(f"    Score={score:.3f}, DirAcc={np.mean([e.direction_correct for e in evals]):.2f}")

            # Credit assignment
            credits = loop.assign_credit(traj, score, realized)
            print(f"    Credits: {credits}")
        else:
            print(f"    No realized returns available for evaluation")

    # Check state
    summary = loop.summary()
    print(f"\n  Summary: {summary}")
    assert summary["episodes"] == 3
    assert memory.size > 0, "Memory should have entries after episodes"

    print("  PASSED\n")


def test_prediction_evaluation():
    """Test prediction evaluation pipeline."""
    print("=" * 60)
    print("TEST 6: Prediction Evaluation")
    print("=" * 60)

    realized = get_realized_returns("AAPL", "2025-01-15", [1, 7, 14])
    if not realized:
        print("  SKIPPED: No realized returns data available")
        return

    from ael.benchmarks.finance import PredictionResult
    prediction = PredictionResult(
        ticker="AAPL",
        date="2025-01-15",
        predictions={
            1: {"direction": "up", "magnitude": 0.01, "confidence": 0.6},
            7: {"direction": "up", "magnitude": 0.03, "confidence": 0.5},
            14: {"direction": "down", "magnitude": 0.02, "confidence": 0.4},
        },
    )

    evals = evaluate_prediction(prediction, realized)
    print(f"  Evaluated {len(evals)} horizons:")
    for ev in evals:
        print(f"    {ev.horizon}d: predicted={ev.predicted_direction}, "
              f"actual={ev.actual_direction}, correct={ev.direction_correct}")

    metrics = compute_metrics(evals)
    print(f"  Overall: {metrics.get('overall', {})}")

    score = compute_outcome_score(evals)
    print(f"  Outcome score: {score:.3f}")

    print("  PASSED\n")


def main():
    np.random.seed(42)
    print("\nAEL Framework Smoke Test")
    print("=" * 60)
    print()

    test_tool_execution()
    test_planner_pool()
    test_memory_store()
    test_credit_assignment()
    test_evolution_loop()
    test_prediction_evaluation()

    print("=" * 60)
    print("ALL SMOKE TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
