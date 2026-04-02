"""Portfolio backtest runner for AEL/EAEL.

Runs the portfolio allocation task: each bar, the agent allocates weights
across all tickers. One LLM call per bar (vs 10 in per-ticker mode).

This runner wires in the FULL AEL evolution loop:
- Meta-controller selects planner + memory policy per bar
- Credit assignment after each bar outcome
- Daily reflection at end of day
- Nightly evolution (skill extraction, planner evolution, memory policy evolution)

Usage:
    python -m ael.benchmarks.portfolio_runner --config experiments/configs/...yaml
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from ael.benchmarks.backtest import get_trading_bars, get_trading_days
from ael.benchmarks.finance import get_realized_returns
from ael.benchmarks.portfolio import (
    PortfolioState,
    build_portfolio_prompt,
    compute_portfolio_metrics,
    parse_portfolio_response,
    _equal_weight,
)
from ael.config import ExperimentConfig
from ael.evolution.loop import EvolutionLoop
from ael.evolution.meta_controller import TaskFeatures
from ael.evolution.reflection import compute_daily_side_info
from ael.tools.finance_tools import set_frequency
from ael.types import TrajectoryRecord, ToolCall

logger = logging.getLogger(__name__)


# ── Strategy descriptions for portfolio prompt injection ──────────────
PLANNER_PORTFOLIO_GUIDANCE = {
    "sequential": "Systematic: analyze all signals in order, then allocate based on consensus.",
    "decompose": "Decompose by dimension: first valuation, then momentum, then risk — allocate based on each sub-analysis.",
    "adaptive": "Quick scan first; deep-dive only on ambiguous tickers. Be decisive on clear signals.",
    "cot_reasoning": "Step-by-step: analyze trend → valuation → sentiment → risk for each ticker, then construct portfolio.",
    "reflexion": "Predict first, then self-critique your allocation. Adjust if your critique reveals a flaw.",
    "hypothesis_test": "Form a market hypothesis, test it against the signals, then allocate accordingly.",
}


class PortfolioRunner:
    """Runs portfolio allocation backtest with full AEL evolution loop."""

    def __init__(
        self,
        config: ExperimentConfig,
        evolution_loop: EvolutionLoop,
    ):
        self.config = config
        self.loop = evolution_loop
        self.tickers = config.benchmark.tickers
        self.portfolio = PortfolioState()

        # Initialize equal-weight
        self.portfolio.set_weights(_equal_weight(self.tickers))

        # Rolling buffer for z-score normalization of outcome scores
        self._return_buffer: list[float] = []
        self._alpha_buffer: list[float] = []

    def run(
        self,
        predict_fn=None,
        max_episodes: int | None = None,
    ) -> dict:
        """Run portfolio backtest with full AEL evolution loop.

        Each bar = 1 episode:
        1. Meta-controller selects planner + memory policy
        2. Run tools for ALL tickers (portfolio needs cross-ticker signals)
        3. Retrieve memory with selected policy
        4. Build prompt with planner strategy + memory + signals
        5. LLM allocates weights
        6. Evaluate portfolio return
        7. Credit assignment → bandit updates
        8. End of day: reflection + nightly evolution
        """
        frequency = self.config.benchmark.frequency
        bars_per_day = self.config.benchmark.bars_per_day
        flat_threshold = self.config.benchmark.flat_threshold
        warm_up = self.config.warm_up_episodes

        set_frequency(frequency, use_cache=True)

        # Build bar schedule
        if frequency == "1h":
            all_bars = get_trading_bars(
                self.config.benchmark.start_date,
                self.config.benchmark.end_date,
                self.tickers,
            )
            bars_by_day = {}
            for bar in all_bars:
                day = bar.split(" ")[0]
                bars_by_day.setdefault(day, []).append(bar)
            trading_days = sorted(bars_by_day.keys())
        else:
            trading_days = get_trading_days(
                self.config.benchmark.start_date,
                self.config.benchmark.end_date,
            )
            bars_by_day = {d: [d] for d in trading_days}

        train_end = self.config.benchmark.train_end_date
        val_end = self.config.benchmark.val_end_date
        has_split = bool(train_end and val_end)

        total_bars = sum(len(v) for v in bars_by_day.values())
        logger.info(
            f"Portfolio backtest: {len(trading_days)} days × {bars_per_day} bars "
            f"= {total_bars} episodes (full AEL loop)"
        )

        episode_count = 0
        score_history = []
        allocation_history = []
        config_history = []
        current_phase = "train"
        current_regime = "unknown"
        turnovers = []
        reflections = []
        yesterday_reflection = {}

        # Phase tracking
        phase_returns = {"train": [], "val": [], "test": []}

        for day_idx, day in enumerate(trading_days):
            if max_episodes and episode_count >= max_episodes:
                break

            # Determine phase
            if has_split:
                if day <= train_end:
                    phase = "train"
                elif day <= val_end:
                    phase = "val"
                else:
                    phase = "test"
                if phase != current_phase:
                    logger.info(f"=== Phase: {current_phase} → {phase} (day={day}) ===")
                    if phase == "test" and not self.loop.frozen:
                        self.loop.freeze()
                    current_phase = phase

            # ═══════════════════════════════════════════════════
            # MORNING: Plan the Day
            # ═══════════════════════════════════════════════════
            in_warmup = warm_up > 0 and episode_count < warm_up
            side_info = compute_daily_side_info(day, self.tickers)

            # NOTE: morning_planning removed — its output was never used in
            # the allocation prompt (wasted LLM call). Yesterday's reflection
            # insight is now injected directly into the prompt instead.

            # ═══════════════════════════════════════════════════
            # EXECUTION: Run All Bars
            # ═══════════════════════════════════════════════════
            day_results = []
            day_bars = bars_by_day.get(day, [day])

            for bar_idx, bar_time in enumerate(day_bars):
                if max_episodes and episode_count >= max_episodes:
                    break

                in_warmup = warm_up > 0 and episode_count < warm_up

                # 1. Run ALL tools for all tickers (portfolio needs full signal coverage)
                all_tool_outputs = {}
                for ticker in self.tickers:
                    ticker_outputs = {}
                    for tool_name in self.loop.tools.all_names:
                        try:
                            kwargs = {"ticker": ticker}
                            if frequency == "1h" and tool_name in (
                                "get_price_history", "compute_technicals",
                                "compute_quant_risk", "compute_momentum",
                                "compute_correlations",
                            ):
                                kwargs["as_of_datetime"] = bar_time
                            result = self.loop.tools.call(tool_name, **kwargs)
                            if result.success:
                                ticker_outputs[tool_name] = result.output
                        except Exception:
                            pass
                    all_tool_outputs[ticker] = ticker_outputs

                # 2. Compute current market regime
                recent_rets = [h["port_return"] for h in self.portfolio.history[-20:]]
                if recent_rets:
                    avg_ret = sum(recent_rets) / len(recent_rets)
                    current_regime = "bullish" if avg_ret > 0.001 else ("bearish" if avg_ret < -0.001 else "mixed")
                else:
                    avg_ret = 0
                    current_regime = "unknown"
                regime_context = {"regime": current_regime, "avg_return": avg_ret}

                # 3. META-CONTROLLER: select planner + memory policy
                features = self._build_portfolio_features(day, all_tool_outputs)
                if not in_warmup:
                    selection = self.loop.meta.select(features, frozen=self.loop.frozen)
                else:
                    selection = {"planner": "sequential", "tool_config": "__all__",
                                 "memory_policy": "none"}
                planner_name = selection["planner"]
                memory_policy_name = selection["memory_policy"]

                # 4. Retrieve memory with SELECTED policy
                memory_hits = []
                memory_context = None
                if not in_warmup:
                    try:
                        memory_policy = self.loop.meta.get_memory_policy(memory_policy_name)
                        mem_query = {
                            "task_type": "portfolio_allocation",
                            "market_regime": current_regime,
                        }
                        if hasattr(self.loop.memory, "retrieve_with_strategy"):
                            memory_hits = self.loop.memory.retrieve_with_strategy(
                                mem_query, memory_policy,
                            )
                        if not memory_hits:
                            memory_hits = self.loop.memory.retrieve(
                                query=mem_query,
                                tiers=["semantic", "procedural"],
                                top_k=5,
                            ) or []
                        if memory_hits:
                            matched = [h for h in memory_hits if h.relevance_score > 0.2]
                            if matched:
                                memory_context = [
                                    {"tier": h.tier, "content": h.content}
                                    for h in matched[:3]
                                ]
                    except Exception:
                        pass

                # 5. Build prompt WITH planner strategy injection
                planner_desc = PLANNER_PORTFOLIO_GUIDANCE.get(planner_name, "")
                strategy_context = {"planner": planner_name, "guidance": planner_desc}

                prompt = build_portfolio_prompt(
                    bar_time, self.tickers, all_tool_outputs,
                    self.portfolio, memory_context,
                    regime_context=regime_context,
                    strategy_context=strategy_context if not in_warmup else None,
                    yesterday_insight=yesterday_reflection if not in_warmup else None,
                )

                # 6. LLM allocation
                if predict_fn is not None:
                    try:
                        from ael.llm_utils import llm_call
                        response = llm_call(
                            model=self.config.llm_model,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=self.config.llm_temperature,
                            max_tokens=512,
                        )
                        text = response.choices[0].message.content
                        new_weights = parse_portfolio_response(text, self.tickers)
                    except Exception as e:
                        logger.warning(f"LLM allocation failed: {e}")
                        new_weights = _equal_weight(self.tickers)
                else:
                    new_weights = _equal_weight(self.tickers)

                # 7. Apply allocation
                turnover = self.portfolio.set_weights(new_weights)
                turnovers.append(turnover)

                # 8. Get realized returns for all tickers
                ticker_returns = {}
                for ticker in self.tickers:
                    realized = get_realized_returns(
                        ticker, bar_time, [1],
                        frequency=frequency, flat_threshold=flat_threshold,
                    )
                    if 1 in realized:
                        ticker_returns[ticker] = realized[1]["return_pct"]

                # 9. Update portfolio
                port_return = self.portfolio.update(ticker_returns)
                self.portfolio.history[-1]["turnover"] = turnover
                phase_returns[current_phase].append(port_return)

                # Outcome score: portfolio return scaled to [-1, 1] via z-score
                self._return_buffer.append(port_return)
                if len(self._return_buffer) > 10:
                    buf = np.array(self._return_buffer)
                    mu, sigma = float(np.mean(buf)), float(np.std(buf))
                    sigma = max(sigma, 1e-5)  # avoid division by zero
                    z = (port_return - mu) / sigma
                    outcome_score = float(np.clip(z / 2, -1, 1))
                else:
                    # Bootstrap: use generous default scale for hourly returns
                    outcome_score = float(np.clip(port_return / 0.005, -1, 1))

                # Compute planner alpha: value-add over equal-weight baseline
                equal_weight = 1.0 / len(self.tickers)
                equal_return = sum(ticker_returns.get(t, 0) for t in self.tickers) * equal_weight
                planner_return = sum(
                    new_weights.get(t, equal_weight) * ticker_returns.get(t, 0)
                    for t in self.tickers
                )
                planner_alpha = planner_return - equal_return

                # Z-scored alpha for meta-controller
                self._alpha_buffer.append(planner_alpha)
                if len(self._alpha_buffer) > 10:
                    abuf = np.array(self._alpha_buffer)
                    amu, asigma = float(np.mean(abuf)), float(np.std(abuf))
                    asigma = max(asigma, 1e-6)
                    alpha_z = float(np.clip((planner_alpha - amu) / asigma / 2, -1, 1))
                else:
                    alpha_z = float(np.clip(planner_alpha / 0.002, -1, 1))

                # 10. Derive PREDICTED direction from allocation intent (BEFORE seeing return)
                # The implicit prediction is: overweighting bullish-momentum stocks = "up"
                bullish_weight = 0.0
                for t in self.tickers:
                    mom = all_tool_outputs.get(t, {}).get("compute_momentum", {})
                    ret4 = mom.get("return_4bar", 0) or 0
                    if ret4 > 0:
                        bullish_weight += new_weights.get(t, 0)
                predicted_dir = "up" if bullish_weight > 0.5 else "down"
                actual_dir = "up" if port_return > 0 else "down"

                # 11. BUILD TRAJECTORY for credit assignment
                tool_calls = []
                for ticker in self.tickers:
                    for tool_name, output in all_tool_outputs.get(ticker, {}).items():
                        tool_calls.append(ToolCall(
                            tool_name=tool_name,
                            inputs={"ticker": ticker},
                            output=output,
                            success=True,
                        ))

                trajectory = TrajectoryRecord(
                    task_type="portfolio_allocation",
                    task_features={
                        "tickers": self.tickers,
                        "date": bar_time,
                        "regime": current_regime,
                    },
                    timestamp=bar_time,
                    planner_id=planner_name,
                    tool_config=list(self.loop.tools.all_names),
                    memory_policy=self.loop.meta.get_memory_policy(memory_policy_name)
                        if not in_warmup else {},
                    tools_invoked=tool_calls,
                    memory_retrievals=memory_hits,
                    prediction={
                        "direction": predicted_dir,  # from allocation intent, NOT hindsight
                        "weights": dict(new_weights),
                    },
                )
                trajectory._selection = selection
                trajectory._task_features = features
                trajectory._tool_outputs = all_tool_outputs
                trajectory._planner_alpha = planner_alpha
                trajectory._alpha_z = alpha_z

                # 12. CREDIT ASSIGNMENT (no-op when frozen or warmup)
                external_signal = {
                    1: {
                        "direction": actual_dir,
                        "return_pct": port_return,
                    },
                    "per_ticker": {
                        t: {"return": ticker_returns.get(t, 0)}
                        for t in self.tickers
                    },
                }
                credits = self.loop.assign_credit(
                    trajectory, outcome_score,
                    external_signal=external_signal,
                )

                # 13. EPISODIC MEMORY: store per-bar experience
                if not in_warmup and not self.loop.frozen:
                    try:
                        self.loop._store_episode_memory_v2(
                            trajectory,
                            tool_evals={},  # portfolio mode: no per-tool ground truth
                            actual_direction=actual_dir,
                        )
                    except Exception:
                        pass

                # Record selection for analysis
                config_history.append({
                    "episode": episode_count,
                    "date": bar_time,
                    "phase": current_phase,
                    "planner": planner_name,
                    "memory_policy": memory_policy_name,
                    "credits": credits,
                })

                score_history.append({
                    "episode": episode_count,
                    "bar_time": bar_time,
                    "phase": current_phase,
                    "port_return": port_return,
                    "total_value": self.portfolio.total_value,
                    "drawdown": self.portfolio.drawdown,
                    "turnover": turnover,
                    "score": outcome_score,
                    "planner": planner_name,
                })

                allocation_history.append({
                    "bar_time": bar_time,
                    "weights": dict(new_weights),
                    "port_return": port_return,
                })

                # 12. Build episode summary for day_results
                episode_summary = {
                    "ticker": "PORTFOLIO",
                    "date": day,
                    "phase": current_phase,
                    "planner": planner_name,
                    "tools_used": list(self.loop.tools.all_names),
                    "score": outcome_score,
                    "dir_acc": 1.0 if predicted_dir == actual_dir else 0.0,
                    "tool_evals": {},
                }
                day_results.append(episode_summary)

                logger.info(
                    f"Bar {episode_count}: {bar_time} [{current_phase}] "
                    f"planner={planner_name} "
                    f"return={port_return:+.3%} total={self.portfolio.total_value:.4f} "
                    f"turnover={turnover:.3f}"
                    f"{'[warmup]' if in_warmup else ''}"
                )

                episode_count += 1

                # NOTE: Session summaries disabled for portfolio mode —
                # they write noisy LLM narratives to semantic memory that
                # pollute allocation memory retrieval.

            if max_episodes and episode_count >= max_episodes:
                break

            # ═══════════════════════════════════════════════════
            # EVENING: Reflect on the Day
            # ═══════════════════════════════════════════════════
            in_warmup = warm_up > 0 and episode_count < warm_up

            # Track empirical accuracy of yesterday's regime prediction
            if yesterday_reflection and day_results:
                pred_regime = yesterday_reflection.get("market_regime", "")
                avg_day_ret = np.mean([r.get("score", 0) for r in day_results])
                if pred_regime in ("bullish", "bearish"):
                    actual_regime = "bullish" if avg_day_ret > 0 else "bearish"
                    if not hasattr(self, "_reflection_acc_buf"):
                        self._reflection_acc_buf: list[bool] = []
                    self._reflection_acc_buf.append(pred_regime == actual_regime)

            if day_results and not in_warmup:
                yesterday_reflection = self.loop.do_daily_reflection(
                    day_results, side_info,
                )
                # Inject empirical accuracy so the prompt builder can use it
                if hasattr(self, "_reflection_acc_buf") and len(self._reflection_acc_buf) >= 3:
                    window = self._reflection_acc_buf[-10:]
                    yesterday_reflection["_empirical_accuracy"] = sum(window) / len(window)
                reflections.append(yesterday_reflection)

                # ═══════════════════════════════════════════════════
                # NIGHT: Evolve Components
                # ═══════════════════════════════════════════════════
                self.loop.do_nightly_evolution(
                    yesterday_reflection, day_results=day_results,
                )

        # Unfreeze
        if self.loop.frozen:
            self.loop.unfreeze()

        # Compute metrics
        overall_metrics = compute_portfolio_metrics(self.portfolio.history)

        phase_metrics = {}
        for phase, rets in phase_returns.items():
            if rets:
                arr = np.array(rets)
                mean_r = float(np.mean(arr))
                std_r = float(np.std(arr))
                ann = np.sqrt(252 * 4)
                sharpe = mean_r / std_r * ann if std_r > 0 else 0

                # Sortino: downside deviation only
                downside = arr[arr < 0]
                down_std = float(np.std(downside)) if len(downside) > 1 else std_r
                sortino = mean_r / down_std * ann if down_std > 0 else 0

                # Max drawdown from cumulative returns
                cum = np.cumprod(1 + arr)
                peak = np.maximum.accumulate(cum)
                dd = (peak - cum) / peak
                max_dd = float(np.max(dd)) if len(dd) > 0 else 0

                # Calmar: annualized return / max drawdown
                ann_ret = mean_r * 252 * 4
                calmar = ann_ret / max_dd if max_dd > 0.001 else 0

                # Win rate
                win_rate = float(np.mean(arr > 0))

                # Tail ratio: 95th percentile gain / abs(5th percentile loss)
                p95 = float(np.percentile(arr, 95))
                p5 = float(np.percentile(arr, 5))
                tail_ratio = abs(p95 / p5) if abs(p5) > 1e-8 else 1.0

                total_ret = float(np.prod(1 + arr) - 1)

                phase_metrics[phase] = {
                    "avg_return": mean_r,
                    "total_return": total_ret,
                    "sharpe": sharpe,
                    "sortino": sortino,
                    "calmar": calmar,
                    "total_return_pct": total_ret * 100,
                    "max_drawdown_pct": max_dd * 100,
                    "win_rate": win_rate,
                    "tail_ratio": tail_ratio,
                    "n_bars": len(rets),
                }

        result = {
            "metrics": overall_metrics,
            "phase_metrics": phase_metrics,
            "score_history": score_history,
            "allocation_history": allocation_history[-20:],
            "config_history": config_history,
            "reflections": len(reflections),
            "total_episodes": episode_count,
            "total_days": len(trading_days),
            "portfolio_final": {
                "total_value": self.portfolio.total_value,
                "max_drawdown": overall_metrics.get("max_drawdown", 0),
                "sharpe_ratio": overall_metrics.get("sharpe_ratio", 0),
            },
        }

        # Verify AEL loop was actually active
        self._verify_loop_activity()

        # Save
        self._save_results(result)
        return result

    def _verify_loop_activity(self) -> None:
        """Post-run verification that the AEL loop was actually active."""
        checks = []

        # 1. Planner bandit was used
        try:
            pb = self.loop.meta.planner_bandit
            arms = pb.arms if hasattr(pb, 'arms') else {}
            total_pulls = 0
            for name, arm in arms.items():
                if isinstance(arm, dict):
                    n = arm.get('pull_count', 0)  # LinUCB arms are dicts
                else:
                    n = getattr(arm, 'pull_count', getattr(arm, 'n', 0))
                total_pulls += n
            checks.append(f"OK: planner bandit total pulls = {total_pulls}" if total_pulls > 0
                           else "FAIL: planner bandit pull_count = 0")
        except Exception as e:
            checks.append(f"WARN: could not inspect planner bandit: {e}")

        # 2. Meta-controller episodes
        eps = getattr(self.loop.meta, '_episodes_elapsed', 0)
        checks.append(f"OK: meta episodes_elapsed = {eps}" if eps > 0
                       else "FAIL: meta episodes = 0")

        # 3. Episodic memory was written
        try:
            ms = self.loop.memory
            entries = getattr(ms, '_entries', getattr(ms, 'entries', []))
            if isinstance(entries, dict):
                entries = list(entries.values())
            tiers = {}
            for e in entries:
                t = getattr(e, 'tier', 'unknown')
                tiers[t] = tiers.get(t, 0) + 1
            checks.append(f"OK: memory tiers: {tiers}")
            if tiers.get('episodic', 0) == 0:
                checks.append("WARN: no episodic memory entries")
        except Exception as e:
            checks.append(f"WARN: could not inspect memory: {e}")

        # 4. Reflections were generated
        n_ref = len(self.loop._reflections) if hasattr(self.loop, '_reflections') else 0
        checks.append(f"OK: reflections = {n_ref}" if n_ref > 0
                       else "WARN: no reflections (expected if < 1 full day)")

        # 5. Day index advanced
        day_idx = getattr(self.loop, '_day_index', 0)
        checks.append(f"OK: day_index = {day_idx}" if day_idx > 0
                       else "WARN: day_index = 0")

        # Log all checks
        logger.info("=" * 60)
        logger.info("AEL LOOP VERIFICATION")
        for c in checks:
            logger.info(f"  {c}")
        logger.info("=" * 60)

    def _build_portfolio_features(self, day: str, tool_outputs: dict) -> TaskFeatures:
        """Aggregate per-ticker signals into portfolio-level TaskFeatures for LinUCB."""
        features = TaskFeatures()

        # Average volatility across tickers
        vols = []
        for ticker, tools in tool_outputs.items():
            risk = tools.get("compute_quant_risk", {})
            vol_data = risk.get("volatility", {})
            vol = vol_data.get("annual", 0) if isinstance(vol_data, dict) else 0
            if vol > 0:
                vols.append(vol)
        features.volatility = float(np.mean(vols)) if vols else 0.3

        # Data richness: fraction of tickers with decent tool coverage
        features.data_richness = sum(
            1 for t in tool_outputs if len(tool_outputs[t]) >= 8
        ) / max(len(self.tickers), 1)

        # Average momentum
        moms = []
        for ticker, tools in tool_outputs.items():
            mom = tools.get("compute_momentum", {})
            ret = mom.get("return_20bar", 0)
            if ret:
                moms.append(ret)
        features.momentum_30d = float(np.mean(moms)) if moms else 0

        return features

    def _save_results(self, results: dict) -> None:
        log_dir = Path(self.config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        path = log_dir / f"backtest_{self.config.name}.json"
        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Portfolio results saved to {path}")

        self.loop.save_state()
