"""Historical backtest runner for AEL/EAEL evolution experiments.

EAEL v2: Day-level human-like loop — cold start → for each day:
  morning planning → execute all tickers → evening reflection → nightly evolution.

Supports train/val/test temporal split: agent learns during train+val,
is frozen (no learning) during test. Metrics are reported per-phase.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ael.benchmarks.finance import (
    PredictionTask,
    PredictionResult,
    compute_metrics,
    compute_outcome_score,
    evaluate_prediction,
    get_realized_returns,
)
from ael.config import ExperimentConfig
from ael.evolution.loop import EvolutionLoop
from ael.evolution.meta_controller import TaskFeatures
from ael.evolution.reflection import compute_daily_side_info

logger = logging.getLogger(__name__)


def get_trading_days(start: str, end: str) -> list[str]:
    """Get list of NYSE trading days between start and end dates."""
    try:
        import exchange_calendars as xcals
        nyse = xcals.get_calendar("XNYS")
        sessions = nyse.sessions_in_range(start, end)
        return [s.strftime("%Y-%m-%d") for s in sessions]
    except ImportError:
        # Fallback: use pandas business days
        days = pd.bdate_range(start, end)
        return [d.strftime("%Y-%m-%d") for d in days]


def get_trading_bars(start: str, end: str, tickers: list[str]) -> list[str]:
    """Get list of hourly bar timestamps from pre-cached data.

    Reads the first ticker's CSV to discover available timestamps,
    then filters to the [start, end] date range.
    """
    from pathlib import Path
    prices_path = Path(__file__).parent.parent.parent / "data" / "prices_1h" / f"{tickers[0]}.csv"
    if not prices_path.exists():
        raise FileNotFoundError(f"No hourly data at {prices_path}")

    import csv
    with open(prices_path) as f:
        reader = csv.DictReader(f)
        all_timestamps = []
        for row in reader:
            dt = list(row.values())[0]  # first column is the datetime index
            date_part = dt.split(" ")[0]
            if start <= date_part <= end:
                all_timestamps.append(dt)

    return all_timestamps


def extract_task_features(
    ticker: str,
    date: str,
    sector_map: dict[str, int] | None = None,
    data_dir: Path | None = None,
) -> TaskFeatures:
    """Extract TaskFeatures for a (ticker, date) pair.

    Reads from cached data first, falls back to yfinance.
    """
    import os

    if data_dir is None:
        data_dir = Path(os.environ.get(
            "AEL_DATA_DIR",
            Path(__file__).parent.parent.parent / "data",
        ))

    features = TaskFeatures()

    try:
        # Sector/metadata from cache
        meta_path = data_dir / "metadata" / "tickers.json"
        if meta_path.exists():
            import json
            with open(meta_path) as f:
                meta = json.load(f)
            info = meta.get(ticker, {})
            sector = info.get("sector", "Unknown")
            if sector_map:
                features.sector_id = sector_map.get(sector, 0)
            mc = info.get("market_cap", 0) or 0
            features.market_cap_log = np.log(mc + 1) if mc else 0.0
            features.has_options = True
            features.has_analyst = True

        # Volatility and momentum from cached prices
        prices_path = data_dir / "prices" / f"{ticker}.csv"
        if prices_path.exists():
            df = pd.read_csv(prices_path, index_col=0, parse_dates=True)
            # Filter to dates up to the as-of date
            end_dt = pd.Timestamp(date)
            df = df[df.index <= end_dt]
            if len(df) > 5:
                close = df["Close"]
                returns = close.pct_change().dropna()
                features.volatility = float(returns.tail(30).std() * np.sqrt(252))
                if len(close) >= 30:
                    features.momentum_30d = float(
                        close.iloc[-1] / close.iloc[-min(30, len(close))] - 1
                    )
                features.data_richness = 1.0
        else:
            features.data_richness = 0.5

    except Exception as e:
        logger.warning(f"Failed to extract features for {ticker}: {e}")

    return features


def _build_tickers_metadata(
    tickers: list[str],
    data_dir: Path | None = None,
) -> dict[str, dict]:
    """Build ticker metadata dict for cold-start agent."""
    import os
    if data_dir is None:
        data_dir = Path(os.environ.get(
            "AEL_DATA_DIR",
            Path(__file__).parent.parent.parent / "data",
        ))

    metadata = {}
    meta_path = data_dir / "metadata" / "tickers.json"
    if meta_path.exists():
        with open(meta_path) as f:
            all_meta = json.load(f)
        for ticker in tickers:
            info = all_meta.get(ticker, {})
            metadata[ticker] = {
                "sector": info.get("sector", "Unknown"),
                "market_cap": info.get("market_cap"),
                "beta": info.get("beta"),
            }
    else:
        for ticker in tickers:
            metadata[ticker] = {"sector": "Unknown"}
    return metadata


def _determine_phase(day: str, train_end: str, val_end: str) -> str:
    """Determine which phase a trading day belongs to.

    Returns 'train', 'val', or 'test'.
    """
    if not train_end or not val_end:
        return "train"  # no split configured — everything is train
    if day <= train_end:
        return "train"
    elif day <= val_end:
        return "val"
    else:
        return "test"


class BacktestRunner:
    """Runs a historical backtest with AEL/EAEL evolution.

    EAEL v2: Day-level loop with morning/evening/night phases.
    Supports train/val/test temporal split with frozen test evaluation.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        evolution_loop: EvolutionLoop,
        tickers: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ):
        self.config = config
        self.loop = evolution_loop
        self.tickers = tickers or config.benchmark.tickers
        self.start_date = start_date or config.benchmark.start_date
        self.end_date = end_date or config.benchmark.end_date

        self.evaluations = []
        self.episode_scores = []
        self._sector_map = self._build_sector_map()

        # Per-phase tracking
        self._phase_evaluations: dict[str, list] = {"train": [], "val": [], "test": []}
        self._phase_scores: dict[str, list] = {"train": [], "val": [], "test": []}

    # ------------------------------------------------------------------
    # Checkpointing: save/load at day boundaries for resume
    # ------------------------------------------------------------------

    def _save_checkpoint(
        self, day_idx: int, episode_count: int, current_phase: str,
        config_history: list, score_history: list, reflections: list,
        yesterday_reflection: dict,
    ) -> None:
        """Save checkpoint after nightly evolution for resume capability."""
        checkpoint_dir = Path(self.config.log_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "day_idx": day_idx,
            "episode_count": episode_count,
            "current_phase": current_phase,
            "config_history": config_history,
            "score_history": score_history,
            "reflections": reflections,
            "yesterday_reflection": yesterday_reflection,
            "episode_scores": self.episode_scores,
            "phase_scores": {p: list(s) for p, s in self._phase_scores.items()},
        }

        # Save checkpoint metadata
        path = checkpoint_dir / f"day_{day_idx:04d}.json"
        with open(path, "w") as f:
            json.dump(checkpoint, f, default=str)

        # Save evolution loop state
        self.loop.save_state(checkpoint_dir / f"loop_day_{day_idx:04d}.json")

        # Rotate: keep last 3
        all_ckpts = sorted(checkpoint_dir.glob("day_*.json"))
        for old in all_ckpts[:-3]:
            old.unlink(missing_ok=True)
            loop_old = checkpoint_dir / f"loop_{old.stem}.json"
            loop_old.unlink(missing_ok=True)

        logger.info(f"Checkpoint saved: day {day_idx}, episode {episode_count}")

    def _load_checkpoint(self) -> dict | None:
        """Load most recent checkpoint for resume. Returns None if no checkpoint."""
        checkpoint_dir = Path(self.config.log_dir) / "checkpoints"
        if not checkpoint_dir.exists():
            return None

        checkpoints = sorted(checkpoint_dir.glob("day_*.json"))
        if not checkpoints:
            return None

        latest = checkpoints[-1]
        try:
            checkpoint = json.loads(latest.read_text())

            # Restore loop state
            day_idx = checkpoint["day_idx"]
            loop_path = checkpoint_dir / f"loop_day_{day_idx:04d}.json"
            if loop_path.exists():
                self.loop.load_state(loop_path)

            # Restore instance-level accumulators
            self.episode_scores = checkpoint.get("episode_scores", [])
            self._phase_scores = {
                p: checkpoint.get("phase_scores", {}).get(p, [])
                for p in ("train", "val", "test")
            }

            logger.info(
                f"Resumed from checkpoint: day {day_idx}, "
                f"episode {checkpoint['episode_count']}"
            )
            return checkpoint
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return None

    def _build_sector_map(self) -> dict[str, int]:
        """Map GICS sectors to integer IDs."""
        sectors = [
            "Technology", "Healthcare", "Financials", "Consumer Cyclical",
            "Communication Services", "Industrials", "Consumer Defensive",
            "Energy", "Utilities", "Real Estate", "Basic Materials",
        ]
        return {s: i for i, s in enumerate(sectors)}

    def run(
        self,
        predict_fn=None,
        max_episodes: int | None = None,
    ) -> dict:
        """Run full backtest with EAEL day-level loop.

        Structure:
          Cold Start → for each day:
            Morning Planning → Execute all tickers → Evening Reflection → Night Evolution

        When train_end_date and val_end_date are configured, the agent is frozen
        at the start of the test phase. Metrics are reported per-phase.

        Args:
            predict_fn: Callable(task_dict, tool_outputs, memory_context) -> PredictionResult.
            max_episodes: Cap on total episodes (for debugging).

        Returns:
            Dict with metrics, learning curves, configuration history, and reflections.
        """
        frequency = self.config.benchmark.frequency
        bars_per_day = self.config.benchmark.bars_per_day
        flat_threshold = self.config.benchmark.flat_threshold

        # Set global frequency for tools
        from ael.tools.finance_tools import set_frequency
        set_frequency(frequency)

        # Build bar schedule
        if frequency == "1h":
            all_bars = get_trading_bars(self.start_date, self.end_date, self.tickers)
            # Group bars by day
            bars_by_day: dict[str, list[str]] = {}
            for bar in all_bars:
                day = bar.split(" ")[0]
                bars_by_day.setdefault(day, []).append(bar)
            trading_days = sorted(bars_by_day.keys())
        else:
            trading_days = get_trading_days(self.start_date, self.end_date)
            bars_by_day = {d: [d] for d in trading_days}  # daily: 1 "bar" per day

        train_end = self.config.benchmark.train_end_date
        val_end = self.config.benchmark.val_end_date
        has_split = bool(train_end and val_end)

        total_bars = sum(len(v) for v in bars_by_day.values())
        logger.info(
            f"Backtest: {len(trading_days)} days × {bars_per_day} bars × "
            f"{len(self.tickers)} tickers = {total_bars * len(self.tickers)} episodes "
            f"(frequency={frequency})"
        )
        if has_split:
            logger.info(f"Temporal split: train≤{train_end} | val≤{val_end} | test>{val_end}")

        episode_count = 0
        config_history = []
        score_history = []
        reflections = []
        current_phase = "train"
        sessions_per_day = self.config.reflection.evolution_sessions_per_day

        # ═══════════════════════════════════════════════════
        # RESUME FROM CHECKPOINT (if available)
        # ═══════════════════════════════════════════════════
        resume_day_idx = 0
        yesterday_reflection: dict[str, Any] = {}

        checkpoint = self._load_checkpoint()
        if checkpoint:
            resume_day_idx = checkpoint["day_idx"] + 1
            episode_count = checkpoint["episode_count"]
            config_history = checkpoint.get("config_history", [])
            score_history = checkpoint.get("score_history", [])
            reflections = checkpoint.get("reflections", [])
            yesterday_reflection = checkpoint.get("yesterday_reflection", {})
            current_phase = checkpoint.get("current_phase", "train")
            logger.info(f"Resuming from day {resume_day_idx}, episode {episode_count}")
        else:
            # ═══════════════════════════════════════════════════
            # COLD START (Day 0) — only if not resuming
            # ═══════════════════════════════════════════════════
            tickers_metadata = _build_tickers_metadata(self.tickers)
            cold_start_result = self.loop.do_cold_start(self.tickers, tickers_metadata)
            if cold_start_result:
                logger.info("Cold-start complete: LLM-generated priors applied")

        for day_idx, day in enumerate(trading_days):
            # Skip days already processed (resume)
            if day_idx < resume_day_idx:
                continue
            if max_episodes and episode_count >= max_episodes:
                break

            # Determine phase and handle transitions
            if has_split:
                phase = _determine_phase(day, train_end, val_end)
                if phase != current_phase:
                    logger.info(f"=== Phase transition: {current_phase} → {phase} (day={day}) ===")
                    if phase == "test" and not self.loop.frozen:
                        self.loop.freeze()
                    current_phase = phase

            # ═══════════════════════════════════════════════════
            # MORNING: Plan the Day (skip during warm-up)
            # ═══════════════════════════════════════════════════
            warm_up = self.config.warm_up_episodes
            in_warmup = warm_up > 0 and episode_count < warm_up
            side_info = compute_daily_side_info(day, self.tickers)

            if day_idx > 0 and yesterday_reflection and not in_warmup:
                morning_strategy = self.loop.do_morning_planning(
                    yesterday_reflection, side_info
                )
            else:
                morning_strategy = {}

            # ═══════════════════════════════════════════════════
            # EXECUTION: Run All Bars × Tickers
            # ═══════════════════════════════════════════════════
            day_results = []
            day_bars = bars_by_day.get(day, [day])
            session_results = []

            for bar_idx, bar_time in enumerate(day_bars):
                for ticker in self.tickers:
                    if max_episodes and episode_count >= max_episodes:
                        break

                    in_warmup = warm_up > 0 and episode_count < warm_up
                    if in_warmup and episode_count % 100 == 0:
                        logger.info(f"Warm-up: {episode_count}/{warm_up}")
                    logger.info(f"Episode {episode_count}: {ticker} @ {bar_time} [{current_phase}]{'[warmup]' if in_warmup else ''}")

                    # Extract features
                    features = extract_task_features(ticker, day, self._sector_map)

                    # Build task with as_of_datetime for hourly filtering
                    task = PredictionTask(
                        ticker=ticker,
                        date=bar_time,  # hourly: full datetime string
                        horizons=self.config.benchmark.prediction_horizons,
                    ).to_dict()
                    if frequency == "1h":
                        task["as_of_datetime"] = bar_time
                    if in_warmup:
                        task["_warmup"] = True  # signal to run_episode to skip memory

                    # Run episode (select config → execute tools)
                    trajectory = self.loop.run_episode(task, features)

                    # Record configuration selection
                    selection = getattr(trajectory, "_selection", {})
                    config_history.append({
                        "episode": episode_count,
                        "date": bar_time,
                        "ticker": ticker,
                        "phase": current_phase,
                        **selection,
                    })

                    # Get realized returns (from cache for hourly, yfinance for daily)
                    realized = get_realized_returns(
                        ticker, bar_time,
                        self.config.benchmark.prediction_horizons,
                        frequency=frequency,
                        flat_threshold=flat_threshold,
                    )

                episode_summary = {
                    "ticker": ticker,
                    "date": day,
                    "phase": current_phase,
                    "planner": trajectory.planner_id,
                    "tools_used": [tc.tool_name for tc in trajectory.tools_invoked if tc.success],
                    "score": 0.0,
                    "dir_acc": 0.0,
                    "tool_evals": {},
                }

                if realized:
                    # Generate prediction
                    prediction = self._make_prediction(
                        trajectory, task, predict_fn
                    )

                    # Evaluate
                    evals = evaluate_prediction(prediction, realized)
                    self.evaluations.extend(evals)
                    self._phase_evaluations[current_phase].extend(evals)

                    # Compute outcome score
                    score = compute_outcome_score(evals)
                    self.episode_scores.append(score)
                    self._phase_scores[current_phase].append(score)

                    # Credit assignment + module update (no-op when frozen)
                    credits = self.loop.assign_credit(
                        trajectory, score,
                        external_signal=realized,
                    )

                    dir_acc = float(np.mean([e.direction_correct for e in evals]))

                    episode_summary["score"] = score
                    episode_summary["dir_acc"] = dir_acc

                    # Collect per-tool evals for reflection
                    tool_outputs = getattr(trajectory, "_tool_outputs", {})
                    actual_dir = "flat"
                    for h in [7, 14, 1]:
                        if h in realized:
                            actual_dir = realized[h].get("direction", "flat")
                            break
                    episode_summary["tool_evals"] = (
                        self.loop.evaluate_tools_against_ground_truth(
                            trajectory, actual_dir
                        )
                    )

                    score_history.append({
                        "episode": episode_count,
                        "ticker": ticker,
                        "date": day,
                        "phase": current_phase,
                        "score": score,
                        "dir_acc": dir_acc,
                    })

                    day_results.append(episode_summary)
                    session_results.append(episode_summary)
                    episode_count += 1

                # Session boundary: generate causal summary + optional reflection
                if sessions_per_day > 1 and len(day_bars) > 1:
                    bars_per_session = max(1, len(day_bars) // sessions_per_day)
                    if (bar_idx + 1) % bars_per_session == 0 and bar_idx < len(day_bars) - 1:
                        if not self.loop.frozen and not in_warmup:
                            session_label = f"{day} bar{bar_idx+1}"
                            self.loop.generate_session_summary(session_label)
                        session_results = []

            if max_episodes and episode_count >= max_episodes:
                break

            # ═══════════════════════════════════════════════════
            # EVENING: Reflect on the Day (skipped during warm-up and when frozen)
            # ═══════════════════════════════════════════════════
            in_warmup = warm_up > 0 and episode_count < warm_up
            if day_results and not in_warmup:
                yesterday_reflection = self.loop.do_daily_reflection(
                    day_results, side_info
                )
                reflections.append(yesterday_reflection)

                # ═══════════════════════════════════════════════════
                # NIGHT: Evolve Components (skipped during warm-up and when frozen)
                # ═══════════════════════════════════════════════════
                self.loop.do_nightly_evolution(yesterday_reflection, day_results=day_results)

                # Save checkpoint for resume capability
                self._save_checkpoint(
                    day_idx, episode_count, current_phase,
                    config_history, score_history, reflections,
                    yesterday_reflection,
                )

        # Unfreeze if we froze
        if self.loop.frozen:
            self.loop.unfreeze()

        # Overall metrics
        metrics = compute_metrics(self.evaluations)

        # Per-phase metrics
        phase_metrics = {}
        for phase in ("train", "val", "test"):
            if self._phase_evaluations[phase]:
                phase_metrics[phase] = compute_metrics(self._phase_evaluations[phase])

        # Learning curve
        learning_curve = self._compute_learning_curve(score_history)

        # Health report
        from ael.memory import get_health_issues
        health = {
            "memory_backend": type(self.loop.memory).__name__,
            "health_issues": get_health_issues(),
        }

        result = {
            "metrics": metrics,
            "phase_metrics": phase_metrics,
            "learning_curve": learning_curve,
            "config_history": config_history,
            "score_history": score_history,
            "reflections": reflections,
            "total_episodes": episode_count,
            "total_days": len(trading_days),
            "planner_pool": self.loop.planners.get_planner_stats(),
            "evolution_summary": self.loop.summary(),
            "temporal_split": {
                "train_end_date": train_end,
                "val_end_date": val_end,
                "has_split": has_split,
            },
            "health": health,
        }

        # Save results
        self._save_results(result)
        return result

    def _make_prediction(
        self,
        trajectory,
        task: dict,
        predict_fn=None,
    ) -> PredictionResult:
        """Generate a PredictionResult from trajectory outputs."""
        tool_outputs = {
            tc.tool_name: tc.output
            for tc in trajectory.tools_invoked
            if tc.success and tc.output is not None
        }

        if predict_fn is not None:
            return predict_fn(task, tool_outputs, trajectory.memory_retrievals)

        return self._baseline_predict(task, tool_outputs)

    def _baseline_predict(
        self, task: dict, tool_outputs: dict
    ) -> PredictionResult:
        """Simple baseline predictor from tool outputs."""
        horizons = task.get("horizons", [1, 7, 14])
        predictions = {}

        # Try to use signal_score output
        signal = tool_outputs.get("signal_score", {})
        composite = signal.get("composite_score", 0)

        # Try technicals
        technicals = tool_outputs.get("compute_technicals", {})
        tech_score = technicals.get("technical_score", 2.5)

        # Combined heuristic
        if composite != 0:
            base_magnitude = abs(composite) / 100 * 0.1
            direction = "up" if composite > 10 else ("down" if composite < -10 else "flat")
            confidence = min(0.9, abs(composite) / 100)
        else:
            base_magnitude = abs(tech_score - 2.5) / 5 * 0.05
            direction = "up" if tech_score > 3 else ("down" if tech_score < 2 else "flat")
            confidence = 0.3 + abs(tech_score - 2.5) / 5 * 0.3

        for h in horizons:
            scale = np.sqrt(h)
            predictions[h] = {
                "direction": direction,
                "magnitude": float(base_magnitude * scale),
                "confidence": float(confidence * (1 - 0.1 * np.log(h))),
            }

        return PredictionResult(
            ticker=task["ticker"],
            date=task["date"],
            predictions=predictions,
        )

    def _compute_learning_curve(
        self, score_history: list[dict], window: int = 20
    ) -> list[dict]:
        """Compute rolling accuracy over episodes."""
        if len(score_history) < window:
            return score_history

        curve = []
        scores = [s["score"] for s in score_history]
        for i in range(window, len(scores) + 1):
            window_scores = scores[i - window:i]
            curve.append({
                "episode_end": i,
                "mean_score": float(np.mean(window_scores)),
                "std_score": float(np.std(window_scores)),
            })
        return curve

    def _save_results(self, results: dict) -> None:
        """Save backtest results to log directory."""
        log_dir = Path(self.config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        path = log_dir / f"backtest_{self.config.name}.json"
        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Backtest results saved to {path}")

        # Also save trajectories
        self.loop.save_trajectories()
        self.loop.save_state()
