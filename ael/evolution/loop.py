"""Main evolution loop: execute episodes, assign credit, update modules.

EAEL v2: Day-level human-like loop with cold-start, morning planning,
daily reflection, and nightly planner evolution.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ael.config import ExperimentConfig
from ael.evolution.credit import compute_fcc, uniform_credit, structural_credit, portfolio_credit, llm_uniform_credit
from ael.evolution.meta_controller import MetaController, TaskFeatures
from ael.evolution.reflection import (
    apply_cold_start_priors,
    architecture_reflection,
    cold_start,
    compute_daily_side_info,
    daily_reflection,
    evolve_planner,
    evolve_memory_policy,
    morning_planning,
)
from ael.memory.store import MemoryStore
from ael.memory.xmem_client import XMemClient
from ael.planner.pool import PlannerPool
from ael.skills.store import SkillStore
from ael.skills.learner import SkillLearner
from ael.tools.registry import ToolRegistry
from ael.types import (
    CrossModuleInsight,
    MemoryHit,
    PlanStep,
    TrajectoryRecord,
)

logger = logging.getLogger(__name__)


class EvolutionLoop:
    """Orchestrates the AEL evolution cycle.

    For each episode:
    1. Receive task (ticker + date)
    2. Meta-controller selects (planner, tools, memory_policy)
    3. Execute plan with selected configuration
    4. Log trajectory
    5. When outcome is available: compute FCC credit
    6. Update module profiles
    7. Extract cross-module insights
    """

    def __init__(
        self,
        config: ExperimentConfig,
        tool_registry: ToolRegistry,
        planner_pool: PlannerPool,
        memory_store: MemoryStore | XMemClient,
        meta_controller: MetaController,
        predict_fn: Callable | None = None,
    ):
        self.config = config
        self.tools = tool_registry
        self.planners = planner_pool
        self.memory = memory_store
        self.meta = meta_controller
        self.predict_fn = predict_fn  # (task, tool_outputs, memory_hits) → PredictionResult

        self.trajectories: list[TrajectoryRecord] = []
        self.insights: list[CrossModuleInsight] = []
        self._log_dir = Path(config.log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)

        # Cache for baseline computations (FCC)
        self._baseline_cache: dict[str, float] = {}

        # EAEL v2: day-level state
        self._day_index: int = 0
        self._morning_strategy: dict[str, Any] = {}
        self._reflections: list[dict] = []
        self._planner_failure_streak: dict[str, int] = {}

        # Skills learning system
        self.skill_store = SkillStore(
            max_per_type=config.skills.max_skills_per_type
        )
        self.skill_learner = SkillLearner(config)

        # Session buffer: accumulate raw episode results for batch causal summary
        self._session_buffer: list[dict] = []

        # Architecture reflection state
        self._daily_scores: list[float] = []  # rolling window of daily avg scores
        self._last_regime: str = "unknown"
        self._architecture_patches: list[dict] = []  # applied code patches
        self._prompt_patches: list = []  # stored prompt modifications
        self._metadata_patches: list = []  # stored metadata modifications
        self._last_retrieved_memories: list[dict] = []  # for diagnostics

        # Freeze state: when True, all learning is disabled (test phase)
        self._frozen: bool = False

    # ------------------------------------------------------------------
    # Freeze / unfreeze for test-phase evaluation
    # ------------------------------------------------------------------

    def freeze(self) -> None:
        """Freeze the agent: disable all learning for test-phase evaluation.

        When frozen:
        - assign_credit() becomes a no-op (returns zeros)
        - do_morning_planning() / do_daily_reflection() / do_nightly_evolution() are skipped
        - Memory writes are disabled
        Episodes still execute (tools run, predictions made) using the learned config.
        """
        self._frozen = True
        # Disable memory writes
        if hasattr(self.memory, 'set_read_only'):
            self.memory.set_read_only(True)
        # Skills: no new skills in test phase (extraction skipped via _frozen check)
        logger.info("Agent FROZEN — learning disabled for test phase")

    def unfreeze(self) -> None:
        """Unfreeze the agent: re-enable learning."""
        self._frozen = False
        if hasattr(self.memory, 'set_read_only'):
            self.memory.set_read_only(False)
        logger.info("Agent UNFROZEN — learning re-enabled")

    @property
    def frozen(self) -> bool:
        return self._frozen

    # ------------------------------------------------------------------
    # EAEL v2: Day-level methods
    # ------------------------------------------------------------------

    def do_cold_start(
        self,
        tickers: list[str],
        tickers_metadata: dict[str, dict],
    ) -> dict[str, Any]:
        """Day 0: LLM reads components, generates warm-start priors."""
        if not self.config.cold_start.enabled:
            logger.info("Cold-start disabled, using uniform priors")
            return {}

        priors = cold_start(
            tool_registry=self.tools,
            planner_pool=self.planners,
            tickers_metadata=tickers_metadata,
            config=self.config,
        )

        if priors:
            apply_cold_start_priors(priors, self.meta, self.tools)
            logger.info("Cold-start priors applied")

        return priors

    def do_morning_planning(
        self,
        yesterday_reflection: dict[str, Any],
        side_info: dict[str, Any],
    ) -> dict[str, Any]:
        """Morning: review yesterday, form today's strategy."""
        if self._frozen or not self.config.reflection.enabled:
            return {}

        strategy = morning_planning(
            yesterday_reflection=yesterday_reflection,
            side_info=side_info,
            memory_store=self.memory,
            config=self.config,
        )
        self._morning_strategy = strategy
        # Read-only: store for logging but do NOT override bandits
        # self.meta.set_morning_overrides(strategy)  # removed: bandits keep full control
        return strategy

    def do_daily_reflection(
        self,
        day_results: list[dict],
        side_info: dict[str, Any],
    ) -> dict[str, Any]:
        """Evening: reflect on today's results."""
        if self._frozen or not self.config.reflection.enabled:
            return {}

        reflection = daily_reflection(
            day_results=day_results,
            side_info=side_info,
            memory_store=self.memory,
            config=self.config,
            recent_reflections=self._reflections[-3:],
            evolve_planners=self.config.reflection.evolve_planners,
        )
        self._reflections.append(reflection)

        # Track planner failure streaks
        critique = reflection.get("planner_critique", {})
        failing = critique.get("failing_planner")
        if failing:
            self._planner_failure_streak[failing] = (
                self._planner_failure_streak.get(failing, 0) + 1
            )
        # Reset streak for successful planner
        successful = critique.get("successful_planner")
        if successful and successful in self._planner_failure_streak:
            self._planner_failure_streak[successful] = 0

        return reflection

    def do_nightly_evolution(
        self,
        reflection: dict[str, Any],
        day_results: list[dict] | None = None,
    ) -> None:
        """Night: memory consolidation, planner evolution, skill extraction."""
        if self._frozen:
            return
        self._day_index += 1

        # Memory consolidation (every night for hourly data, every 2 nights for daily)
        consolidate_interval = 1 if self.config.benchmark.bars_per_day >= 4 else 2
        if self._day_index % consolidate_interval == 0:
            self.memory.auto_distill(min_episodes=8)
            self.memory.promote_to_procedural(min_samples=8)

        # Update planner prompts from procedural memory
        for planner_name in self.planners.all_names:
            self.planners.get(planner_name).update_prompt_from_memory(self.memory)

        # Clear morning overrides for tomorrow
        self.meta.clear_morning_overrides()

        # Skill extraction
        if (
            self.config.skills.enabled
            and self._day_index % self.config.skills.extraction_interval_days == 0
        ):
            try:
                new_skills = self.skill_learner.extract_all(
                    reflection=reflection,
                    memory_store=self.memory,
                    tool_registry=self.tools,
                    planner_pool=self.planners,
                    config=self.config,
                    day_results=day_results,
                    day_index=self._day_index,
                )
                for skill in new_skills:
                    added = self.skill_store.add_skill(skill)
                    if added and skill.skill_type == "tool_code":
                        tool_instance = self.skill_learner.instantiate_tool(skill)
                        if tool_instance:
                            self.tools.register_dynamic(tool_instance)
                            if hasattr(self.meta, 'tool_bandit'):
                                self.meta.tool_bandit.add_arm(tool_instance.name)
                            logger.info(f"Registered dynamic tool: {tool_instance.name}")
            except Exception as e:
                logger.warning(f"Skill extraction failed: {e}")

            # Prune underperforming skills periodically
            self.skill_store.prune(
                min_episodes=self.config.skills.skill_probation_episodes,
                min_success_rate=self.config.skills.min_procedural_confidence,
            )

        # Memory policy evolution: when bandit shows low reward across all policies
        if self.config.memory.evolve and self._day_index >= 10 and self._day_index % 5 == 0:
            bandit_stats = self.meta.memory_bandit.state_dict()
            avg_reward = sum(
                s.get("alpha", 1) / (s.get("alpha", 1) + s.get("beta", 1))
                for s in bandit_stats.values()
            ) / max(len(bandit_stats), 1)
            if avg_reward < 0.4:
                new_policy = evolve_memory_policy(
                    reflection=reflection,
                    current_policies=self.meta.memory_registry.policies,
                    bandit_stats=bandit_stats,
                    recent_trajectories=self.trajectories[-50:],
                    config=self.config,
                )
                if new_policy:
                    self.meta.memory_registry.add(new_policy["name"], new_policy)
                    self.meta.memory_bandit.add_arm(new_policy["name"])
                    logger.info(f"Added evolved memory policy: {new_policy['name']}")

        # Architecture reflection: diagnose and fix system design flaws
        if self._should_reflect_architecture(day_results or []):
            side_info = {"regime": reflection.get("market_regime", "unknown")}
            self.do_architecture_reflection(day_results or [], side_info)

        # Planner evolution: only if structural problem detected
        structural = reflection.get("structural_problem", {})
        if not structural.get("detected", False):
            return

        # Check if the failure streak exceeds threshold
        failing = reflection.get("planner_critique", {}).get("failing_planner")
        streak = self._planner_failure_streak.get(failing, 0)
        threshold = self.config.reflection.structural_problem_threshold

        if streak >= threshold:
            logger.info(
                f"Planner '{failing}' failed {streak} consecutive days — "
                f"triggering evolution"
            )
            new_planner = evolve_planner(
                reflection=reflection,
                planner_pool=self.planners,
                recent_trajectories=self.trajectories[-50:],
                config=self.config,
            )
            if new_planner:
                # Add to meta-controller's planner bandit
                self.meta.planner_bandit.add_arm(new_planner.name)
                # Reset failure streak
                self._planner_failure_streak[failing] = 0

    # ------------------------------------------------------------------
    # Single episode execution
    # ------------------------------------------------------------------

    def run_episode(
        self,
        task: dict,
        task_features: TaskFeatures,
    ) -> TrajectoryRecord:
        """Execute one episode: select config → plan → execute → log.

        Args:
            task: Task specification (ticker, date, etc.).
            task_features: Extracted features for bandit context.

        Returns:
            TrajectoryRecord with execution trace (outcome filled later).
        """
        start_time = time.time()

        # 1. Meta-controller selects configuration
        ticker = task.get("ticker")
        selection = self.meta.select(task_features, ticker=ticker)
        planner_name = selection["planner"]
        tool_config_name = selection["tool_config"]
        memory_policy_name = selection["memory_policy"]

        # Resolve to actual objects
        planner = self.planners.get(planner_name)

        # EAEL v2: per-tool selection or legacy presets
        if tool_config_name == "__per_tool__":
            available_tools = self.meta.select_tools(tool_registry=self.tools)
        else:
            available_tools = self.meta.get_tool_list(tool_config_name)
            if available_tools is None:
                available_tools = self.tools.all_names

        memory_policy = self.meta.get_memory_policy(memory_policy_name)

        # 1b. Retrieve applicable skills (with quality gate)
        applicable_skills = []
        if self.config.skills.enabled:
            side_info = task.get("side_info", {})
            all_applicable = self.skill_store.get_applicable(task, side_info)
            # Quality gate: only inject new (untested) or proven skills
            applicable_skills = [
                s for s in all_applicable
                if s.episode_count < 10 or s.success_rate > 0.45
            ]

            # Inject prompt templates into planner context
            prompt_templates = [
                s for s in applicable_skills if s.skill_type == "prompt_template"
            ]
            if prompt_templates:
                task = dict(task)  # shallow copy to avoid mutating original
                template_text = "\n".join(
                    f"- {s.content}" for s in prompt_templates
                )
                task["skill_templates"] = template_text

            # Use strategy playbooks to bias tool/planner selection
            strategy_skills = [
                s for s in applicable_skills if s.skill_type == "strategy"
            ]
            if strategy_skills and not self._frozen:
                import json as _json
                best_strategy = strategy_skills[0]
                try:
                    playbook = _json.loads(best_strategy.content)
                    strategy_tools = playbook.get("tool_sequence", [])
                    if strategy_tools:
                        # Bias available_tools toward strategy recommendation
                        available_tools = list(
                            dict.fromkeys(strategy_tools + list(available_tools))
                        )
                except (ValueError, KeyError):
                    pass

        # 2. Retrieve relevant memories (skip during warm-up)
        memory_hits: list[MemoryHit] = []
        in_warmup = task.get("_warmup", False)
        if not in_warmup:
            mem_query = {
                "task_type": task.get("task_type", "price_prediction"),
                "ticker": task.get("ticker", ""),
                "sector": task.get("sector", ""),
            }
            if memory_policy.get("format") and hasattr(self.memory, "retrieve_with_strategy"):
                memory_hits = self.memory.retrieve_with_strategy(mem_query, memory_policy)
            elif memory_policy.get("retrieval_top_k", 0) > 0:
                memory_hits = self.memory.retrieve(
                    query=mem_query,
                    tiers=memory_policy.get("enabled_tiers", []),
                    top_k=memory_policy["retrieval_top_k"],
                )

        memory_context = [
            {"tier": h.tier, "content": h.content, "score": h.relevance_score}
            for h in memory_hits
        ]

        # 3. Generate plan
        plan_steps = planner.plan(task, available_tools, memory_context)

        # 4. Execute plan steps
        tool_calls = []
        tool_outputs = {}
        total_cost = 0.0

        for step in plan_steps:
            if step.action.startswith("call:") or step.action.startswith("call_if_ambiguous:"):
                tool_name = step.action.split(":", 1)[1]
                if tool_name not in set(available_tools):
                    step.success = False
                    step.outcome = f"Tool {tool_name} not available"
                    continue

                # Prepare tool kwargs from task
                tool_kwargs = self._prepare_tool_kwargs(
                    tool_name, task, tool_outputs
                )
                result = self.tools.call(
                    tool_name,
                    task_type=task.get("task_type", ""),
                    **tool_kwargs,
                )
                tool_calls.append(result)
                total_cost += result.cost

                if result.success:
                    tool_outputs[tool_name] = result.output
                    step.success = True
                    step.outcome = f"Success (latency={result.latency:.2f}s)"
                else:
                    step.success = False
                    step.outcome = f"Failed: {result.error}"

            elif step.action.startswith("sub_synthesize:"):
                # Sub-task synthesis — aggregate outputs for this sub-task
                subtask = step.action.split(":", 1)[1]
                step.success = True
                step.outcome = f"sub_result:{subtask}"

            elif step.action in ("synthesize", "quick_synthesize"):
                # LLM synthesis step — generate prediction
                if self.predict_fn and step.action == "synthesize":
                    try:
                        prediction = self.predict_fn(task, tool_outputs, memory_hits)
                        step.success = True
                        step.outcome = json.dumps(prediction.predictions, default=str)
                    except Exception as e:
                        step.success = True
                        step.outcome = f"synthesis_error:{e}"
                else:
                    # Quick synthesis or no LLM: use heuristic
                    step.success = True
                    step.outcome = self._quick_synthesis(tool_outputs)

        wall_time = time.time() - start_time

        # 5. Build trajectory record
        trajectory = TrajectoryRecord(
            task_type=task.get("task_type", "price_prediction"),
            task_features=task,
            planner_id=planner_name,
            tool_config=available_tools,
            memory_policy=memory_policy,
            plan_steps=plan_steps,
            tools_invoked=tool_calls,
            memory_retrievals=memory_hits,
            cost=total_cost,
            wall_time=wall_time,
        )

        # Store selection info for later credit update
        trajectory._selection = selection
        trajectory._task_features = task_features
        trajectory._tool_outputs = tool_outputs
        trajectory._used_skills = applicable_skills

        self.trajectories.append(trajectory)
        return trajectory

    def _quick_synthesis(self, tool_outputs: dict) -> str:
        """Quick heuristic synthesis for the adaptive planner's first pass."""
        tech = tool_outputs.get("compute_technicals", {})
        mom = tool_outputs.get("compute_momentum", {})

        score = 0
        if tech:
            score += (tech.get("technical_score", 2.5) - 2.5) * 20
        if mom:
            ret = mom.get("return_20d", 0) or 0
            score += ret * 200

        return str(score)

    def _prepare_tool_kwargs(
        self, tool_name: str, task: dict, prior_outputs: dict
    ) -> dict:
        """Prepare keyword arguments for a tool call based on task and prior outputs."""
        ticker = task.get("ticker", "")
        as_of = task.get("as_of_datetime")  # hourly: "2025-03-10 11:30:00-04:00"

        # Dynamic tools (re-run each bar) — pass as_of_datetime for temporal filtering
        dynamic_tools = {
            "get_price_history", "compute_technicals", "compute_quant_risk",
            "compute_momentum", "compute_correlations",
        }
        if tool_name in dynamic_tools:
            kwargs = {"ticker": ticker}
            if as_of:
                kwargs["as_of_datetime"] = as_of
            return kwargs

        # Static tools (daily data, no temporal filtering needed)
        static_tools = {
            "get_fundamentals", "get_analyst_data", "get_options_data",
            "get_earnings_data", "run_dcf_model", "score_risk",
        }
        if tool_name in static_tools:
            return {"ticker": ticker}

        # Composite signal scorer needs prior tool outputs
        if tool_name == "score_composite_signal":
            return {"ticker": ticker, "tool_outputs": prior_outputs}

        return {"ticker": ticker}

    # ------------------------------------------------------------------
    # Credit assignment and module update
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # V2: Tool-level ground truth evaluation
    # ------------------------------------------------------------------

    @staticmethod
    def extract_tool_direction(tool_name: str, output: Any) -> str | None:
        """Extract the implicit directional signal from a tool's output."""
        if not isinstance(output, dict):
            return None
        if tool_name == "compute_technicals":
            score = output.get("technical_score", 2.5)
            return "up" if score > 3 else ("down" if score < 2 else None)
        if tool_name == "compute_momentum":
            ret = output.get("return_20d")
            if ret is not None:
                return "up" if ret > 0.005 else ("down" if ret < -0.005 else None)
        if tool_name == "run_dcf_model":
            upside = output.get("implied_upside")
            if upside is not None:
                return "up" if upside > 0.05 else ("down" if upside < -0.05 else None)
        if tool_name == "get_analyst_data":
            rec = output.get("recommendation_mean")
            if rec is not None:
                return "up" if rec < 2.5 else ("down" if rec > 3.5 else None)
        if tool_name == "score_composite_signal":
            cs = output.get("composite_score", 0)
            return "up" if cs > 10 else ("down" if cs < -10 else None)
        if tool_name == "compute_quant_risk":
            rr = output.get("risk_return", {})
            sharpe = rr.get("sharpe_ratio")
            if sharpe is not None:
                return "up" if sharpe > 0.5 else ("down" if sharpe < -0.5 else None)
        if tool_name == "score_risk":
            overall = output.get("overall", 5)
            return "down" if overall > 7 else ("up" if overall < 3 else None)
        return None

    def evaluate_tools_against_ground_truth(
        self,
        trajectory: TrajectoryRecord,
        actual_direction: str,
    ) -> dict[str, dict]:
        """Evaluate each tool's signal against actual direction.

        Returns: tool_name → {"output_summary", "signal", "correct"}
        """
        tool_outputs = getattr(trajectory, "_tool_outputs", {})
        ticker = trajectory.task_features.get("ticker", "")
        evaluations = {}

        for tc in trajectory.tools_invoked:
            if not tc.success:
                continue
            output = tool_outputs.get(tc.tool_name, tc.output)
            signal = self.extract_tool_direction(tc.tool_name, output)

            # Build summary
            summary = self._summarize_tool_output(tc.tool_name, output)

            correct = None
            if signal and actual_direction in ("up", "down"):
                correct = (signal == actual_direction)
                # Update tool profile
                profile = self.tools._profiles.get(tc.tool_name)
                if profile:
                    profile.update_directional(correct, ticker)

            evaluations[tc.tool_name] = {
                "output_summary": summary,
                "signal": signal or "neutral",
                "correct": correct,
            }

        return evaluations

    @staticmethod
    def _summarize_tool_output(tool_name: str, output: Any) -> str:
        """Create a brief NL summary of a tool's output."""
        if not isinstance(output, dict):
            return str(output)[:100]
        if tool_name == "compute_technicals":
            rsi = output.get("rsi_14", {}).get("value", "?")
            macd = output.get("macd", {}).get("trend", "?")
            score = output.get("technical_score", "?")
            return f"RSI={rsi:.0f}, MACD={macd}, score={score}/5" if isinstance(rsi, (int, float)) else f"score={score}"
        if tool_name == "compute_momentum":
            r20 = output.get("return_20d")
            trend = output.get("trend", {}).get("direction", "?")
            return f"20d_return={r20:+.1%}, trend={trend}" if isinstance(r20, (int, float)) else f"trend={trend}"
        if tool_name == "run_dcf_model":
            sig = output.get("signal", "?")
            upside = output.get("implied_upside")
            return f"DCF={sig}, upside={upside:+.1%}" if isinstance(upside, (int, float)) else f"DCF={sig}"
        if tool_name == "get_analyst_data":
            rec = output.get("recommendation_key", "?")
            target = output.get("target_mean_price", "?")
            return f"rec={rec}, target=${target}"
        if tool_name == "score_composite_signal":
            cs = output.get("composite_score", 0)
            sig = output.get("signal", "?")
            return f"composite={cs:+.1f}, signal={sig}"
        if tool_name == "compute_quant_risk":
            sharpe = output.get("risk_return", {}).get("sharpe_ratio", "?")
            vol = output.get("volatility", {}).get("realized_30d_annualized", "?")
            return f"Sharpe={sharpe:.2f}, vol={vol:.1%}" if isinstance(sharpe, (int, float)) and isinstance(vol, (int, float)) else "quant_risk"
        if tool_name == "score_risk":
            overall = output.get("overall", "?")
            rating = output.get("rating", "?")
            return f"risk={overall}/10 ({rating})"
        if tool_name == "get_fundamentals":
            info = output.get("info", {})
            pe = info.get("trailingPE", "?")
            mc = info.get("marketCap")
            mc_str = f"${mc/1e9:.0f}B" if isinstance(mc, (int, float)) else "?"
            return f"PE={pe}, mktcap={mc_str}"
        return f"{tool_name}: {len(output)} fields"

    # ------------------------------------------------------------------
    # Credit assignment and module update (V2: hybrid FCC)
    # ------------------------------------------------------------------

    def assign_credit(
        self,
        trajectory: TrajectoryRecord,
        outcome_score: float,
        external_signal: dict | None = None,
    ) -> dict[str, float]:
        """V2: Assign credit with tool ground truth evaluation + hybrid FCC."""
        trajectory.outcome_score = outcome_score
        trajectory.external_signal = external_signal or {}

        # Frozen mode: record score but skip all learning updates
        if self._frozen:
            self.trajectories.append(trajectory) if trajectory not in self.trajectories else None
            return {"planner": 0.0, "tools": 0.0, "memory": 0.0}

        # V2: Evaluate each tool against ground truth
        actual_direction = "flat"
        if external_signal:
            # Use 7-day return as primary ground truth
            for horizon in [7, 14, 1]:
                if horizon in external_signal:
                    actual_direction = external_signal[horizon].get("direction", "flat")
                    break

        tool_evals = self.evaluate_tools_against_ground_truth(trajectory, actual_direction)

        # Accumulate to session buffer for batch causal summary (skip during warm-up)
        in_warmup = trajectory.task_features.get("_warmup", False)
        if not in_warmup:
            self._session_buffer.append({
                "ticker": trajectory.task_features.get("ticker", "?"),
                "datetime": trajectory.task_features.get("date", "?"),
                "tools_used": [tc.tool_name for tc in trajectory.tools_invoked if tc.success],
                "tool_correct": {name: ev.get("correct") for name, ev in tool_evals.items()},
                "predicted_dir": trajectory.prediction.get("direction", "?") if trajectory.prediction else "?",
                "actual_dir": actual_direction,
                "outcome_score": outcome_score,
                "key_signals": self._extract_key_signals(trajectory),
            })

        # Credit assignment
        method = self.config.credit.method
        episode_num = len(self.trajectories)
        fcc_weights = (
            self.config.credit.fcc_weight_structural,
            self.config.credit.fcc_weight_counterfactual,
            self.config.credit.fcc_weight_shapley,
        )

        # Curriculum: start uniform, switch to target method after enough episodes
        if method == "curriculum":
            if episode_num < self.config.credit.curriculum_switch_episode:
                method = "uniform"
            else:
                method = "fcc"
        elif method == "curriculum_llm":
            if episode_num < self.config.credit.curriculum_switch_episode:
                method = "uniform"
            else:
                method = "llm_fcc"

        # Ground truth for LLM credit (from trajectory's external_signal)
        ground_truth = trajectory.external_signal if trajectory.external_signal else None

        if method == "uniform":
            credits = uniform_credit()
        elif method == "structural":
            credits = structural_credit(trajectory)
        elif method == "llm_uniform":
            credits = llm_uniform_credit(
                trajectory=trajectory,
                outcome_score=outcome_score,
                ground_truth=ground_truth,
                model=self.config.llm_model,
                alpha=self.config.credit.llm_blend_alpha,
            )
        elif method in ("llm", "llm_fcc"):
            # LLM-driven credit: the LLM reasons about module contributions
            credits = compute_fcc(
                trajectory=trajectory,
                outcome_score=outcome_score,
                method=method,
                weights=fcc_weights,
                ground_truth=ground_truth,
                llm_model=self.config.llm_model,
            )
        elif method == "fcc":
            # Portfolio mode: use portfolio-specific credit with per-ticker data
            if trajectory.task_type == "portfolio_allocation" and trajectory.prediction:
                alpha = getattr(trajectory, "_planner_alpha", None)
                tool_outputs = getattr(trajectory, "_tool_outputs", {})
                per_ticker = (external_signal or {}).get("per_ticker", {})
                ticker_rets = {t: v.get("return", 0) for t, v in per_ticker.items()}
                n_tickers = len(self.config.benchmark.tickers) if hasattr(self.config, 'benchmark') else 10
                # No-memory baseline from recent episodes without memory
                no_mem_scores = [t.outcome_score for t in self.trajectories[-50:]
                                 if not t.memory_retrievals
                                 and hasattr(t, 'outcome_score') and t.outcome_score is not None]
                no_mem_base = float(np.mean(no_mem_scores)) if len(no_mem_scores) >= 5 else None
                credits = portfolio_credit(
                    weights=trajectory.prediction.get("weights", {}),
                    ticker_returns=ticker_rets,
                    tool_outputs=tool_outputs,
                    memory_hits=trajectory.memory_retrievals,
                    equal_weight=1.0 / n_tickers,
                    planner_alpha=alpha,
                    no_memory_baseline=no_mem_base,
                )
            else:
                # Non-portfolio: use original heuristic FCC
                shapley_interval = self.config.credit.shapley_interval
                if episode_num > 0 and episode_num % shapley_interval == 0:
                    baseline_outcomes = self._true_counterfactual(
                        trajectory.task_features,
                        getattr(trajectory, "_task_features", None),
                        getattr(trajectory, "_selection", {}),
                        outcome_score,
                    )
                else:
                    baseline_outcomes = self._compute_baselines_v2(trajectory, outcome_score)
                credits = compute_fcc(
                    trajectory=trajectory,
                    outcome_score=outcome_score,
                    baseline_outcomes=baseline_outcomes,
                    method="fcc",
                    weights=fcc_weights,
                )
        else:
            credits = structural_credit(trajectory)

        trajectory.module_credits = credits

        # EAEL v2: record planner score for pruning decisions
        self.planners.record_score(trajectory.planner_id, outcome_score)

        # Update per-tool Thompson priors (EAEL v2)
        # Use directional correctness (graded) instead of binary outcome sign
        if self.meta.per_tool_selection:
            for tc in trajectory.tools_invoked:
                if tc.success:
                    eval_info = tool_evals.get(tc.tool_name, {})
                    correct = eval_info.get("correct")
                    if correct is True:
                        tool_reward = 0.8  # directionally correct
                    elif correct is False:
                        tool_reward = 0.2  # directionally wrong
                    else:
                        # No directional signal: use continuous outcome
                        tool_reward = float(np.clip((outcome_score + 1) / 2, 0.3, 0.7))
                    self.meta.update_per_tool(tc.tool_name, tool_reward)

        # Update meta-controller
        selection = getattr(trajectory, "_selection", None)
        features = getattr(trajectory, "_task_features", None)
        if selection and features:
            reward = np.clip((outcome_score + 1) / 2, 0.05, 0.95)
            alpha = getattr(trajectory, "_planner_alpha", None)
            alpha_z = getattr(trajectory, "_alpha_z", None)
            self.meta.update(selection, features, reward, credits,
                             planner_alpha=alpha, alpha_z=alpha_z)

        # Update skill performance (differential: vs rolling baseline)
        used_skills = getattr(trajectory, "_used_skills", [])
        if used_skills:
            recent = [t.outcome_score for t in self.trajectories[-20:]
                      if hasattr(t, 'outcome_score') and t.outcome_score is not None]
            skill_baseline = float(np.mean(recent)) if len(recent) >= 5 else 0.0
            for skill in used_skills:
                self.skill_store.update_performance(skill.skill_id, outcome_score, skill_baseline)

        # Mark memory retrievals as useful/not
        if trajectory.memory_retrievals:
            if self.config.memory.usefulness_mode == "differential":
                recent_scores = [t.outcome_score for t in self.trajectories[-20:]
                                 if hasattr(t, 'outcome_score') and t.outcome_score is not None]
                baseline = float(np.mean(recent_scores)) if len(recent_scores) >= 5 else 0.0
                is_useful = outcome_score > baseline
            else:
                is_useful = outcome_score > 0
            for hit in trajectory.memory_retrievals:
                self.memory.mark_useful(hit.memory_id, is_useful)

        # V2: Auto-distillation + procedural promotion periodically
        self.memory.increment_episode()
        if episode_num > 0 and episode_num % 10 == 0:
            self.memory.auto_distill(min_episodes=8)
            self.memory.promote_to_procedural(min_samples=8)

        # V2: Update planner prompts from procedural memory
        if episode_num > 0 and episode_num % 10 == 0:
            for planner_name in self.planners.all_names:
                self.planners.get(planner_name).update_prompt_from_memory(self.memory)

        return credits

    def _compute_baselines_v2(
        self,
        trajectory: TrajectoryRecord,
        actual_score: float,
    ) -> dict[str, float]:
        """V2: Improved heuristic baselines using historical tool accuracy."""
        baselines = {}

        # Planner baseline: use historical planner scores
        planner_scores = self._get_planner_historical_scores()
        current_planner_avg = planner_scores.get(trajectory.planner_id, 0)
        baseline_planner_avg = planner_scores.get("sequential", 0)
        # Credit = how much better/worse this planner is vs baseline
        planner_diff = current_planner_avg - baseline_planner_avg
        baselines["planner"] = actual_score - planner_diff

        # Tools baseline: use per-tool directional accuracy
        tools_used = [t for t in trajectory.tools_invoked if t.success]
        if tools_used:
            tool_accs = []
            for tc in tools_used:
                profile = self.tools._profiles.get(tc.tool_name)
                if profile:
                    tool_accs.append(profile.directional_accuracy)
            if tool_accs:
                avg_tool_acc = np.mean(tool_accs)
                # Baseline = what score would be with random (0.5 accuracy) tools
                baselines["tools"] = actual_score * (0.5 / max(avg_tool_acc, 0.01))
            else:
                baselines["tools"] = actual_score * 0.8
        else:
            baselines["tools"] = actual_score * 0.8

        # Memory baseline: score from episodes without memory
        no_mem_scores = [
            t.outcome_score for t in self.trajectories[-50:]
            if not t.memory_retrievals and t.outcome_score != 0
        ]
        if no_mem_scores:
            baselines["memory"] = np.mean(no_mem_scores)
        else:
            baselines["memory"] = actual_score

        return baselines

    def _true_counterfactual(
        self,
        task: dict,
        task_features,
        selection: dict,
        actual_score: float,
    ) -> dict[str, float]:
        """V2: True counterfactual — re-run with each module at baseline.

        For now, uses historical averages from trajectories with baseline
        configurations (cheaper than full LLM re-runs, but data-driven).
        """
        baselines = {}

        # Planner counterfactual: average score when sequential was used
        seq_scores = [t.outcome_score for t in self.trajectories[-100:]
                      if t.planner_id == "sequential" and t.outcome_score != 0]
        baselines["planner"] = np.mean(seq_scores) if seq_scores else actual_score

        # Tools counterfactual: average score when "core" config was used
        core_scores = [t.outcome_score for t in self.trajectories[-100:]
                       if len(t.tool_config) <= 4 and t.outcome_score != 0]
        baselines["tools"] = np.mean(core_scores) if core_scores else actual_score * 0.9

        # Memory counterfactual: average score with no memory
        no_mem_scores = [t.outcome_score for t in self.trajectories[-100:]
                         if not t.memory_retrievals and t.outcome_score != 0]
        baselines["memory"] = np.mean(no_mem_scores) if no_mem_scores else actual_score

        return baselines

    def _get_planner_historical_scores(self) -> dict[str, float]:
        """Get average outcome score per planner from recent trajectories."""
        from collections import defaultdict
        scores = defaultdict(list)
        for t in self.trajectories[-100:]:
            if t.outcome_score != 0:
                scores[t.planner_id].append(t.outcome_score)
        return {k: float(np.mean(v)) for k, v in scores.items()}

    def _store_episode_memory_v2(
        self,
        trajectory: TrajectoryRecord,
        tool_evals: dict[str, dict],
        actual_direction: str,
    ) -> None:
        """V2: Store rich episodic memory with per-tool ground-truth evaluation."""
        ticker = trajectory.task_features.get("ticker", "unknown")
        date = trajectory.task_features.get("date", "?")
        sector = trajectory.task_features.get("sector", "")
        score = trajectory.outcome_score

        # Build prediction info from trajectory
        prediction = trajectory.prediction or {}
        actual = {"direction": actual_direction}
        if trajectory.external_signal:
            for h in [7, 14, 1]:
                if h in trajectory.external_signal:
                    actual["return_pct"] = trajectory.external_signal[h].get("return_pct", 0)
                    break

        self.memory.add_episode_experience(
            ticker=ticker,
            date=date,
            planner_id=trajectory.planner_id,
            tool_evaluations=tool_evals,
            prediction=prediction,
            actual=actual,
            outcome_score=score,
            sector=sector,
        )

    # ------------------------------------------------------------------
    # Session-level causal memory
    # ------------------------------------------------------------------

    def _extract_key_signals(self, trajectory: TrajectoryRecord) -> dict:
        """Extract the most important signal values from tool outputs."""
        outputs = getattr(trajectory, "_tool_outputs", {})
        signals = {}
        tech = outputs.get("compute_technicals", {})
        if tech:
            rsi = tech.get("rsi_14", {})
            signals["rsi"] = rsi.get("value") if isinstance(rsi, dict) else rsi
            macd = tech.get("macd", {})
            signals["macd_trend"] = macd.get("trend") if isinstance(macd, dict) else None
            signals["technical_score"] = tech.get("technical_score")
        mom = outputs.get("compute_momentum", {})
        if mom:
            signals["momentum_4bar"] = mom.get("return_4bar")
            signals["trend_dir"] = mom.get("trend", {}).get("direction") if isinstance(mom.get("trend"), dict) else None
        comp = outputs.get("score_composite_signal", {})
        if comp:
            signals["composite_score"] = comp.get("composite_score")
        return signals

    def generate_session_summary(self, session_label: str = "") -> str | None:
        """Generate a causal summary of the session from the accumulated buffer.

        Called at session boundaries. Uses the meta LLM to analyze all 5 data
        sources and produce a narrative explaining WHY prices moved.

        Returns the summary text, or None if buffer is empty.
        """
        if not self._session_buffer:
            return None

        from ael.llm_utils import llm_call

        buffer = self._session_buffer
        meta_model = self.config.reflection.meta_llm_model or self.config.llm_model

        # 1. Price action per ticker
        from collections import defaultdict
        ticker_prices = defaultdict(list)
        for ep in buffer:
            ticker_prices[ep["ticker"]].append(ep)

        price_lines = []
        for ticker, eps in sorted(ticker_prices.items()):
            outcomes = [f"{ep['actual_dir']}" for ep in eps]
            correct = sum(1 for ep in eps if ep["predicted_dir"] == ep["actual_dir"])
            price_lines.append(f"  {ticker}: {' → '.join(outcomes)} (predicted {correct}/{len(eps)} correct)")

        # 2. Signal values (aggregate key signals)
        signal_lines = []
        for ep in buffer[:6]:  # limit to first 6 for prompt brevity
            sigs = ep.get("key_signals", {})
            sig_str = ", ".join(f"{k}={v}" for k, v in sigs.items() if v is not None)
            signal_lines.append(f"  {ep['ticker']}@{ep['datetime']}: {sig_str}")

        # 3. Predictions vs actual
        pred_lines = []
        for ep in buffer:
            mark = "✓" if ep["predicted_dir"] == ep["actual_dir"] else "✗"
            pred_lines.append(f"  {ep['ticker']}: predicted {ep['predicted_dir']} → actual {ep['actual_dir']} {mark}")

        # 4. Cross-ticker patterns
        directions = defaultdict(list)
        for ep in buffer:
            directions[ep["actual_dir"]].append(ep["ticker"])
        cross_lines = [f"  {d}: {', '.join(set(t))}" for d, t in directions.items()]

        # 5. Aggregate stats
        total = len(buffer)
        correct_total = sum(1 for ep in buffer if ep["predicted_dir"] == ep["actual_dir"])
        avg_score = sum(ep["outcome_score"] for ep in buffer) / max(total, 1)

        prompt = f"""You are a quantitative analyst reviewing a trading session.

## Session: {session_label}
Episodes: {total}, Correct: {correct_total}/{total} ({correct_total/max(total,1):.0%}), Avg score: {avg_score:.3f}

## Price Action Per Ticker
{chr(10).join(price_lines)}

## Key Signal Values
{chr(10).join(signal_lines[:8])}

## Predictions vs Actual
{chr(10).join(pred_lines[:10])}

## Cross-Ticker Patterns
{chr(10).join(cross_lines)}

## Task
Write a 2-3 sentence CAUSAL summary explaining:
1. What market conditions drove the outcomes this session
2. Which signals were reliable vs misleading and WHY
3. A specific lesson for similar future situations

Be concise and specific. Focus on causal reasoning, not just accuracy stats."""

        try:
            response = llm_call(
                model=meta_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200,
            )
            summary = response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"Session summary LLM call failed: {e}")
            summary = f"Session {session_label}: {correct_total}/{total} correct, avg score {avg_score:.3f}"

        # Store as semantic memory
        # Compute regime from session outcomes
        session_returns = [ep.get("outcome_score", 0) for ep in buffer]
        avg_session_score = np.mean(session_returns) if session_returns else 0
        session_regime = "bullish" if avg_session_score > 0.01 else ("bearish" if avg_session_score < -0.01 else "mixed")

        self.memory.add_semantic(
            content=summary,
            metadata={
                "task_type": "session_summary",
                "session": session_label,
                "market_regime": session_regime,
                "n_episodes": total,
                "accuracy": correct_total / max(total, 1),
            },
            quality=max(0.5, min(0.9, 0.5 + correct_total / max(total, 1) * 0.4)),
        )

        # Clear buffer for next session
        self._session_buffer = []

        logger.info(f"Session summary: {summary[:100]}...")
        return summary

    # ------------------------------------------------------------------
    # Architecture Reflection (Level 3): Self-modifying agent design
    # ------------------------------------------------------------------

    def _should_reflect_architecture(self, day_results: list[dict]) -> bool:
        """Check if architecture reflection should trigger.

        Triggers when: performance drops significantly after a regime change.
        """
        if not day_results:
            return False

        # Compute today's avg score
        today_scores = [r.get("score", 0) for r in day_results if "score" in r]
        if not today_scores:
            return False
        today_avg = sum(today_scores) / len(today_scores)
        self._daily_scores.append(today_avg)

        # Need at least 5 days of history
        if len(self._daily_scores) < 5:
            return False

        # Check if recent performance dropped significantly
        recent_avg = sum(self._daily_scores[-2:]) / 2
        historical_avg = sum(self._daily_scores[-7:-2]) / min(5, len(self._daily_scores) - 2)

        # Trigger if recent is significantly worse AND we have memory entries
        drop = historical_avg - recent_avg
        has_memories = len([e for e in self.memory._entries.values()]) > 0 if hasattr(self.memory, '_entries') else False

        if drop > 0.15 and has_memories:
            logger.info(f"Architecture reflection triggered: score drop {drop:.3f} (historical={historical_avg:.3f}, recent={recent_avg:.3f})")
            return True

        return False

    def do_architecture_reflection(self, day_results: list[dict], side_info: dict) -> dict | None:
        """Run architecture-level reflection to diagnose and fix system design flaws.

        Called during nightly evolution when performance degrades after regime change.
        """
        import inspect

        # Gather diagnostics
        current_regime = side_info.get("regime", "unknown")
        if not current_regime or current_regime == "unknown":
            # Infer from recent scores
            recent = self._daily_scores[-3:] if self._daily_scores else [0]
            avg = sum(recent) / len(recent)
            current_regime = "bullish" if avg > 0.05 else ("bearish" if avg < -0.05 else "mixed")

        # Analyze retrieved memories' regime distribution
        mem_regimes = {"bullish": 0, "bearish": 0, "mixed": 0, "unknown": 0}
        for mem in self._last_retrieved_memories:
            regime = mem.get("regime", "unknown")
            mem_regimes[regime] = mem_regimes.get(regime, 0) + 1

        memory_diagnostics = {
            "retrieved_memories": self._last_retrieved_memories[-5:],
            "current_regime": current_regime,
            "memory_regimes": mem_regimes,
            "performance_before": sum(self._daily_scores[-7:-2]) / max(len(self._daily_scores[-7:-2]), 1),
            "performance_after": sum(self._daily_scores[-2:]) / max(len(self._daily_scores[-2:]), 1),
        }

        # Get current code for the LLM to see
        current_code = {}
        try:
            current_code["relevance_score"] = inspect.getsource(self.memory._relevance_score)
        except Exception:
            current_code["relevance_score"] = "# could not extract source"

        current_code["portfolio_prompt_snippet"] = (
            "# Portfolio prompt includes: signals, portfolio state, memory context\n"
            "# Memory context is injected as 'Learned Rules' (procedural) + 'Similar Past Situations' (semantic)\n"
            "# NO regime context is currently included in the prompt"
        )

        # Call the architecture reflection LLM
        fix = architecture_reflection(
            recent_results=day_results,
            memory_diagnostics=memory_diagnostics,
            current_code=current_code,
            config=self.config,
        )

        if fix and fix.get("code_patch"):
            success = self._apply_architecture_fix(fix)
            if success:
                self._architecture_patches.append(fix)
                logger.info(f"Architecture fix applied: {fix.get('target')}")
            return fix

        return None

    def _apply_architecture_fix(self, fix: dict) -> bool:
        """Validate and apply a code-level architecture fix."""
        import ast

        target = fix.get("target", "")
        code = fix.get("code_patch", "")

        if not code or target == "no_fix":
            return False

        # 1. Syntax check
        try:
            ast.parse(code)
        except SyntaxError as e:
            logger.warning(f"Architecture fix has syntax error: {e}")
            return False

        # 2. Sandbox execution
        import math
        namespace = {"math": math, "np": np}
        try:
            exec(code, namespace)
        except Exception as e:
            logger.warning(f"Architecture fix execution failed: {e}")
            return False

        # 3. Apply based on target
        if target == "relevance_score_patch" and "relevance_score_patch" in namespace:
            # Store the patch function — memory store will use it
            patch_fn = namespace["relevance_score_patch"]
            original_fn = self.memory._relevance_score

            def patched_relevance(query, entry):
                base_score = original_fn(query, entry)
                try:
                    return patch_fn(query, entry, base_score)
                except Exception:
                    return base_score

            self.memory._relevance_score = patched_relevance
            logger.info("Applied relevance_score_patch to memory store")
            return True

        elif target == "prompt_patch" and "prompt_patch" in namespace:
            self._prompt_patches.append(namespace["prompt_patch"])
            logger.info("Stored prompt_patch for portfolio allocation")
            return True

        elif target == "metadata_patch" and "metadata_patch" in namespace:
            self._metadata_patches.append(namespace["metadata_patch"])
            logger.info("Stored metadata_patch for memory writes")
            return True

        logger.warning(f"Architecture fix target '{target}' not found in generated code")
        return False

    # ------------------------------------------------------------------
    # Cross-module insight extraction
    # ------------------------------------------------------------------

    def extract_insights(self, min_samples: int = 5) -> list[CrossModuleInsight]:
        """Discover cross-module configuration patterns from trajectory history."""
        from collections import defaultdict

        config_groups = defaultdict(list)
        for traj in self.trajectories:
            if traj.outcome_score == 0.0 and not traj.external_signal:
                continue
            key = (
                traj.planner_id,
                tuple(sorted(traj.tool_config)),
                json.dumps(traj.memory_policy, sort_keys=True),
            )
            config_groups[key].append(traj)

        new_insights = []
        for (planner, tools, mem_json), trajs in config_groups.items():
            if len(trajs) < min_samples:
                continue
            scores = [t.outcome_score for t in trajs]
            avg_score = np.mean(scores)
            if avg_score > 0.3:
                insight = CrossModuleInsight(
                    planner_id=planner,
                    tool_set=list(tools),
                    memory_policy=json.loads(mem_json),
                    task_type_scope=trajs[0].task_type,
                    success_rate=avg_score,
                    sample_count=len(trajs),
                    natural_language=(
                        f"Config (planner={planner}, tools={list(tools)[:3]}..., "
                        f"memory={json.loads(mem_json).get('enabled_tiers', [])}) "
                        f"achieved {avg_score:.2f} avg score over {len(trajs)} episodes."
                    ),
                )
                new_insights.append(insight)

                self.memory.add_semantic(
                    content=insight.natural_language,
                    metadata={
                        "task_type": insight.task_type_scope,
                        "planner": planner,
                        "tools": list(tools)[:5],
                    },
                    quality=min(1.0, avg_score),
                )

        self.insights.extend(new_insights)
        return new_insights

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self, path: str | Path | None = None) -> None:
        """Save full evolution state for resumption."""
        if path is None:
            path = self._log_dir / "evolution_state.json"
        state = {
            "meta_controller": self.meta.state_dict(),
            "tool_registry": self.tools.state_dict(),
            "planner_pool": self.planners.state_dict(),
            "memory_store": self.memory.state_dict(),
            "skill_store": self.skill_store.state_dict(),
            "trajectories": [asdict(t) for t in self.trajectories[-100:]],  # keep last 100
            "insights": [asdict(i) for i in self.insights],
            # Loop-level state for checkpoint resume
            "_day_index": self._day_index,
            "_frozen": self._frozen,
            "_planner_failure_streak": self._planner_failure_streak,
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2, default=str)
        logger.info(f"Evolution state saved to {path}")

    def load_state(self, path: str | Path) -> None:
        """Restore evolution state."""
        with open(path) as f:
            state = json.load(f)
        self.meta.load_state_dict(state["meta_controller"])
        self.tools.load_state_dict(state["tool_registry"])
        self.planners.load_state_dict(state["planner_pool"])
        self.memory.load_state_dict(state["memory_store"])
        if "skill_store" in state:
            self.skill_store.load_state_dict(state["skill_store"])
        # Restore loop-level state
        if "_day_index" in state:
            self._day_index = state["_day_index"]
        if "_frozen" in state:
            self._frozen = state["_frozen"]
        if "_planner_failure_streak" in state:
            self._planner_failure_streak = state["_planner_failure_streak"]
        logger.info(f"Evolution state loaded from {path}")

    def save_trajectories(self, path: str | Path | None = None) -> None:
        """Save trajectory log as JSONL."""
        if path is None:
            path = self._log_dir / "trajectories.jsonl"
        with open(path, "a") as f:
            for traj in self.trajectories:
                f.write(json.dumps(asdict(traj), default=str) + "\n")

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Return high-level evolution statistics."""
        if not self.trajectories:
            return {"episodes": 0}

        scores = [t.outcome_score for t in self.trajectories if t.outcome_score != 0]
        return {
            "episodes": len(self.trajectories),
            "avg_score": np.mean(scores) if scores else 0.0,
            "total_cost": sum(t.cost for t in self.trajectories),
            "insights_discovered": len(self.insights),
            "memory_size": self.memory.size,
            "tool_reliability": self.tools.get_reliability_summary(),
        }
