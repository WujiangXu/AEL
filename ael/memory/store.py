"""V2 Three-tier memory system — experience learner for all modules.

Episodic: per-episode NL records (tool results + ground truth comparison)
Semantic: auto-distilled cross-episode patterns (tool accuracy, planner effectiveness)
Procedural: learned strategies promoted from consistent semantic patterns
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from ael.types import MemoryEntry, MemoryHit


class MemoryStore:

    def __init__(
        self,
        retrieval_top_k: int = 5,
        write_quality_threshold: float = 0.3,
        eviction_min_score: float = 0.2,
        max_entries_per_tier: int = 500,
    ):
        self._entries: dict[str, MemoryEntry] = {}
        self._top_k = retrieval_top_k
        self._write_threshold = write_quality_threshold
        self._eviction_min = eviction_min_score
        self._max_per_tier = max_entries_per_tier
        self._episode_counter = 0  # for recency decay
        self._read_only = False  # when True, all writes are silently skipped

    # ------------------------------------------------------------------
    # Read-only mode (for frozen test-phase evaluation)
    # ------------------------------------------------------------------

    def set_read_only(self, read_only: bool) -> None:
        """Enable/disable read-only mode. When read-only, all writes are skipped."""
        self._read_only = read_only

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add(self, entry: MemoryEntry) -> bool:
        if self._read_only:
            return False
        if entry.quality_score < self._write_threshold:
            return False
        self._entries[entry.entry_id] = entry
        tier_entries = [e for e in self._entries.values() if e.tier == entry.tier]
        if len(tier_entries) > self._max_per_tier:
            self._evict(entry.tier)
        return True

    def add_episodic(self, content: str, metadata: dict | None = None,
                     quality: float = 0.5) -> bool:
        entry = MemoryEntry(tier="episodic", content=content,
                            metadata=metadata or {}, quality_score=quality)
        entry.metadata["episode_num"] = self._episode_counter
        return self.add(entry)

    def add_semantic(self, content: str, metadata: dict | None = None,
                     quality: float = 0.6) -> bool:
        entry = MemoryEntry(tier="semantic", content=content,
                            metadata=metadata or {}, quality_score=quality)
        return self.add(entry)

    def add_procedural(self, content: str, metadata: dict | None = None,
                       quality: float = 0.7) -> bool:
        entry = MemoryEntry(tier="procedural", content=content,
                            metadata=metadata or {}, quality_score=quality)
        return self.add(entry)

    def increment_episode(self):
        """Call after each episode to advance recency counter."""
        self._episode_counter += 1

    # ------------------------------------------------------------------
    # V2: Rich Episodic Memory from Tool Evaluations
    # ------------------------------------------------------------------

    def add_episode_experience(
        self,
        ticker: str,
        date: str,
        planner_id: str,
        tool_evaluations: dict[str, dict],
        prediction: dict,
        actual: dict,
        outcome_score: float,
        sector: str = "",
    ) -> bool:
        """Write a rich episodic memory with per-tool ground-truth evaluations.

        Args:
            tool_evaluations: tool_name → {"output_summary": str, "signal": "up"/"down",
                                           "correct": bool}
            prediction: {"direction": str, "magnitude": float}
            actual: {"direction": str, "return_pct": float}
        """
        lines = [f"{ticker}@{date} (planner={planner_id}):"]

        helped = []
        hurt = []
        for tool_name, ev in tool_evaluations.items():
            summary = ev.get("output_summary", "")
            signal = ev.get("signal", "?")
            correct = ev.get("correct")
            status = "correct" if correct else ("wrong" if correct is False else "N/A")
            lines.append(f"  {tool_name}: {summary} → signal={signal} [{status}]")
            if correct is True:
                helped.append(tool_name)
            elif correct is False:
                hurt.append(tool_name)

        pred_dir = prediction.get("direction", "?")
        pred_mag = prediction.get("magnitude", 0)
        act_dir = actual.get("direction", "?")
        act_ret = actual.get("return_pct", 0)

        lines.append(f"  Predicted: {pred_dir} {pred_mag:+.1%}. Actual: {act_dir} {act_ret:+.1%}.")
        if helped:
            lines.append(f"  Tools that helped: {', '.join(helped)}")
        if hurt:
            lines.append(f"  Tools that hurt: {', '.join(hurt)}")

        content = "\n".join(lines)
        quality = max(0.3, min(1.0, (outcome_score + 1) / 2))

        return self.add_episodic(
            content=content,
            metadata={
                "ticker": ticker,
                "date": date,
                "sector": sector,
                "planner": planner_id,
                "outcome_score": outcome_score,
                "tools_helped": helped,
                "tools_hurt": hurt,
                "prediction_direction": pred_dir,
                "actual_direction": act_dir,
                "tool_names": list(tool_evaluations.keys()),
            },
            quality=quality,
        )

    # ------------------------------------------------------------------
    # Retrieve (V2: recency decay + cross-ticker + tool-specific)
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: dict,
        tiers: list[str] | None = None,
        top_k: int | None = None,
    ) -> list[MemoryHit]:
        if top_k is None:
            top_k = self._top_k
        if tiers is None:
            tiers = ["episodic", "semantic", "procedural"]

        candidates = [e for e in self._entries.values() if e.tier in tiers]
        if not candidates:
            return []

        # Quality filtering: skip low-quality episodic entries when memory is sparse
        total_episodic = sum(1 for e in self._entries.values() if e.tier == "episodic")
        min_quality = 0.4 if total_episodic < 100 else 0.0

        scored = []
        for entry in candidates:
            # Filter low-quality episodic entries (semantic/procedural always pass)
            if entry.tier == "episodic" and entry.quality_score < min_quality:
                continue
            score = self._relevance_score(query, entry)
            if score > 0:
                scored.append((entry, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        hits = []
        for entry, score in scored[:top_k]:
            entry.retrieval_count += 1
            hits.append(MemoryHit(
                memory_id=entry.entry_id,
                tier=entry.tier,
                content=entry.content,
                relevance_score=score,
            ))
        return hits

    def _relevance_score(self, query: dict, entry: MemoryEntry) -> float:
        score = 0.0

        # Ticker match (strong signal)
        if query.get("ticker") and entry.metadata.get("ticker"):
            if query["ticker"] == entry.metadata["ticker"]:
                score += 2.0

        # Sector match (cross-ticker learning)
        if query.get("sector") and entry.metadata.get("sector"):
            if query["sector"] == entry.metadata["sector"]:
                score += 0.8  # higher than before to enable cross-ticker

        # Task type match
        if query.get("task_type") and entry.metadata.get("task_type"):
            if query["task_type"] == entry.metadata["task_type"]:
                score += 0.5

        # Planner match
        if query.get("planner") and entry.metadata.get("planner"):
            if query["planner"] == entry.metadata["planner"]:
                score += 0.5

        # Tool-specific query
        if query.get("tool_name") and entry.metadata.get("tool_names"):
            if query["tool_name"] in entry.metadata["tool_names"]:
                score += 1.0

        # Quality + usefulness boost
        score *= (0.5 + 0.5 * entry.quality_score)
        if entry.retrieval_count > 0:
            score *= (0.5 + 0.5 * entry.avg_usefulness)

        # Recency decay: exponential decay based on episodes ago
        ep_num = entry.metadata.get("episode_num", 0)
        episodes_ago = max(0, self._episode_counter - ep_num)
        recency = math.exp(-0.01 * episodes_ago)
        score *= (0.3 + 0.7 * recency)  # floor at 30% relevance for old entries

        # Tier priority: procedural > semantic > episodic
        tier_boost = {"procedural": 1.5, "semantic": 1.2, "episodic": 1.0}
        score *= tier_boost.get(entry.tier, 1.0)

        # Regime match: gently boost same-regime, mildly penalize cross-regime
        if query.get("market_regime") and entry.metadata.get("market_regime"):
            if query["market_regime"] == entry.metadata["market_regime"]:
                score *= 1.3  # same regime = somewhat more relevant
            else:
                score *= 0.7  # different regime = somewhat less relevant (not harsh)

        return score

    # ------------------------------------------------------------------
    # EAEL v2: Strategy-based retrieval
    # ------------------------------------------------------------------

    def retrieve_with_strategy(
        self,
        query: dict,
        memory_policy: dict,
    ) -> list[MemoryHit]:
        """Retrieve memories and apply formatting strategy.

        Mirrors XMemClient.retrieve_with_strategy for local backend.
        """
        fmt = memory_policy.get("format", "full")
        top_k = memory_policy.get("retrieval_top_k", 0)
        tiers = memory_policy.get("enabled_tiers", [])

        if fmt == "none" or top_k == 0:
            return []

        hits = self.retrieve(query, tiers=tiers, top_k=top_k)
        if not hits:
            return []

        if fmt == "full":
            return hits

        if fmt == "sliding_window":
            keep_first = memory_policy.get("keep_first", 1)
            keep_last = memory_policy.get("keep_last", 3)
            if len(hits) <= keep_first + keep_last:
                return hits
            return hits[:keep_first] + hits[-(keep_last):]

        if fmt == "ranked_truncate":
            token_budget = memory_policy.get("token_budget", 500)
            hits.sort(key=lambda h: h.relevance_score, reverse=True)
            selected = []
            tokens_used = 0
            for h in hits:
                est_tokens = len(h.content) // 4
                if tokens_used + est_tokens > token_budget and selected:
                    break
                selected.append(h)
                tokens_used += est_tokens
            return selected

        if fmt == "llm_summary":
            summary_budget = memory_policy.get("summary_budget", 200)
            combined = "\n".join(h.content for h in hits)
            try:
                from ael.llm_utils import llm_call
                response = llm_call(
                    model=memory_policy.get("model", "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"),
                    messages=[{
                        "role": "user",
                        "content": (
                            f"Summarize these trading analysis memories in under "
                            f"{summary_budget} tokens. Focus on actionable patterns.\n\n"
                            f"{combined}"
                        ),
                    }],
                    temperature=0.2,
                    max_tokens=summary_budget,
                )
                summary = response.choices[0].message.content
                return [MemoryHit(
                    memory_id="__summary__",
                    tier="semantic",
                    content=summary,
                    relevance_score=1.0,
                )]
            except Exception:
                truncated = combined[:summary_budget * 4]
                return [MemoryHit(
                    memory_id="__summary__",
                    tier="semantic",
                    content=truncated,
                    relevance_score=0.8,
                )]

        return hits

    # ------------------------------------------------------------------
    # Feedback
    # ------------------------------------------------------------------

    def mark_useful(self, memory_id: str, useful: bool) -> None:
        if memory_id in self._entries:
            entry = self._entries[memory_id]
            entry.usefulness_sum += 1.0 if useful else 0.0

    # ------------------------------------------------------------------
    # V2: Auto-Distillation (episodic → semantic)
    # ------------------------------------------------------------------

    def auto_distill(self, min_episodes: int = 8) -> list[str]:
        """Scan episodic memories for patterns, create semantic entries.

        Groups by (tool_name, ticker) and (planner, sector) to find
        consistent tool accuracy and planner effectiveness patterns.

        Returns list of created semantic entry IDs.
        """
        created = []
        episodic = [e for e in self._entries.values() if e.tier == "episodic"]

        # Adaptive threshold: lower min_episodes for small datasets
        if len(episodic) < 100:
            min_episodes = min(min_episodes, max(3, len(episodic) // 20))

        # --- Pattern 1: Per-tool directional accuracy ---
        tool_results = defaultdict(lambda: {"correct": 0, "wrong": 0, "tickers": set()})
        for ep in episodic:
            tools_helped = ep.metadata.get("tools_helped", [])
            tools_hurt = ep.metadata.get("tools_hurt", [])
            ticker = ep.metadata.get("ticker", "")
            for t in tools_helped:
                tool_results[t]["correct"] += 1
                tool_results[t]["tickers"].add(ticker)
            for t in tools_hurt:
                tool_results[t]["wrong"] += 1
                tool_results[t]["tickers"].add(ticker)

        for tool_name, stats in tool_results.items():
            total = stats["correct"] + stats["wrong"]
            if total < min_episodes:
                continue
            acc = stats["correct"] / total
            tickers = list(stats["tickers"])

            if acc > 0.6:
                content = (f"{tool_name} is reliable: directional accuracy "
                           f"{acc:.0%} over {total} episodes "
                           f"(tickers: {', '.join(tickers[:5])}). Keep using it.")
                eid = self._add_semantic_if_new(content, {
                    "tool_name": tool_name, "accuracy": acc, "sample_count": total,
                    "pattern_type": "tool_accuracy",
                })
                if eid:
                    created.append(eid)
            elif acc < 0.4:
                content = (f"{tool_name} is unreliable: directional accuracy "
                           f"{acc:.0%} over {total} episodes "
                           f"(tickers: {', '.join(tickers[:5])}). Consider downweighting.")
                eid = self._add_semantic_if_new(content, {
                    "tool_name": tool_name, "accuracy": acc, "sample_count": total,
                    "pattern_type": "tool_accuracy",
                })
                if eid:
                    created.append(eid)

        # --- Pattern 2: Per-planner effectiveness ---
        planner_scores = defaultdict(list)
        for ep in episodic:
            planner = ep.metadata.get("planner", "")
            score = ep.metadata.get("outcome_score", None)
            if planner and score is not None:
                planner_scores[planner].append(score)

        for planner, scores in planner_scores.items():
            if len(scores) < min_episodes:
                continue
            import numpy as np
            avg = float(np.mean(scores))
            content = (f"Planner '{planner}' avg score: {avg:.3f} over "
                       f"{len(scores)} episodes.")
            eid = self._add_semantic_if_new(content, {
                "planner": planner, "avg_score": avg, "sample_count": len(scores),
                "pattern_type": "planner_effectiveness",
            })
            if eid:
                created.append(eid)

        # --- Pattern 3: Per-ticker tool comparison ---
        ticker_tool = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "wrong": 0}))
        for ep in episodic:
            ticker = ep.metadata.get("ticker", "")
            if not ticker:
                continue
            for t in ep.metadata.get("tools_helped", []):
                ticker_tool[ticker][t]["correct"] += 1
            for t in ep.metadata.get("tools_hurt", []):
                ticker_tool[ticker][t]["wrong"] += 1

        for ticker, tools in ticker_tool.items():
            best_tool, best_acc = None, 0
            worst_tool, worst_acc = None, 1
            for tool_name, stats in tools.items():
                total = stats["correct"] + stats["wrong"]
                if total < 5:
                    continue
                acc = stats["correct"] / total
                if acc > best_acc:
                    best_acc, best_tool = acc, tool_name
                if acc < worst_acc:
                    worst_acc, worst_tool = acc, tool_name

            if best_tool and worst_tool and best_tool != worst_tool:
                content = (f"For {ticker}: {best_tool} is most predictive "
                           f"({best_acc:.0%}), {worst_tool} is least predictive "
                           f"({worst_acc:.0%}).")
                eid = self._add_semantic_if_new(content, {
                    "ticker": ticker, "best_tool": best_tool, "worst_tool": worst_tool,
                    "pattern_type": "ticker_tool_comparison",
                })
                if eid:
                    created.append(eid)

        return created

    def _add_semantic_if_new(self, content: str, metadata: dict) -> str | None:
        """Add semantic entry only if no similar entry exists."""
        pattern_type = metadata.get("pattern_type", "")
        key_field = metadata.get("tool_name") or metadata.get("planner") or metadata.get("ticker", "")

        for e in self._entries.values():
            if (e.tier == "semantic"
                and e.metadata.get("pattern_type") == pattern_type
                and (e.metadata.get("tool_name") == key_field
                     or e.metadata.get("planner") == key_field
                     or e.metadata.get("ticker") == key_field)):
                # Update existing instead of creating duplicate
                e.content = content
                e.metadata.update(metadata)
                return None

        entry = MemoryEntry(tier="semantic", content=content,
                            metadata=metadata, quality_score=0.7)
        if self.add(entry):
            return entry.entry_id
        return None

    # ------------------------------------------------------------------
    # V2: Procedural Promotion (semantic → procedural)
    # ------------------------------------------------------------------

    def promote_to_procedural(self, min_samples: int = 10,
                              min_confidence: float = 0.6) -> list[str]:
        """Promote consistent semantic patterns to procedural strategies."""
        created = []
        semantic = [e for e in self._entries.values() if e.tier == "semantic"]

        # Adaptive threshold for small datasets
        if len(semantic) < 20:
            min_samples = min(min_samples, max(3, len(semantic) // 3))

        for entry in semantic:
            sample_count = entry.metadata.get("sample_count", 0)
            if sample_count < min_samples:
                continue

            pattern_type = entry.metadata.get("pattern_type", "")

            if pattern_type == "tool_accuracy":
                acc = entry.metadata.get("accuracy", 0.5)
                tool = entry.metadata.get("tool_name", "")
                if acc >= min_confidence:
                    content = f"STRATEGY: Prioritize {tool} (accuracy={acc:.0%}, N={sample_count})."
                elif acc <= (1 - min_confidence):
                    content = f"STRATEGY: Deprioritize {tool} (accuracy={acc:.0%}, N={sample_count})."
                else:
                    continue
                eid = self._add_procedural_if_new(content, {
                    "tool_name": tool, "strategy_type": "tool_priority",
                })
                if eid:
                    created.append(eid)

            elif pattern_type == "planner_effectiveness":
                avg_score = entry.metadata.get("avg_score", 0)
                planner = entry.metadata.get("planner", "")
                if avg_score > 0:
                    content = (f"STRATEGY: Planner '{planner}' is effective "
                               f"(avg_score={avg_score:.3f}, N={sample_count}).")
                    eid = self._add_procedural_if_new(content, {
                        "planner": planner, "strategy_type": "planner_preference",
                    })
                    if eid:
                        created.append(eid)

        return created

    def _add_procedural_if_new(self, content: str, metadata: dict) -> str | None:
        strategy_type = metadata.get("strategy_type", "")
        key = metadata.get("tool_name") or metadata.get("planner", "")
        for e in self._entries.values():
            if (e.tier == "procedural"
                and e.metadata.get("strategy_type") == strategy_type
                and (e.metadata.get("tool_name") == key
                     or e.metadata.get("planner") == key)):
                e.content = content
                e.metadata.update(metadata)
                return None

        entry = MemoryEntry(tier="procedural", content=content,
                            metadata=metadata, quality_score=0.8)
        if self.add(entry):
            return entry.entry_id
        return None

    # ------------------------------------------------------------------
    # Eviction
    # ------------------------------------------------------------------

    def _evict(self, tier: str) -> None:
        tier_entries = [
            (eid, e) for eid, e in self._entries.items() if e.tier == tier
        ]
        tier_entries.sort(
            key=lambda x: x[1].quality_score * (0.5 + 0.5 * x[1].avg_usefulness)
        )
        excess = len(tier_entries) - self._max_per_tier
        removed = 0
        for eid, entry in tier_entries:
            if removed >= excess:
                break
            if entry.quality_score < self._eviction_min or removed < excess:
                del self._entries[eid]
                removed += 1

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        from dataclasses import asdict
        return {
            "episode_counter": self._episode_counter,
            "entries": {eid: asdict(entry) for eid, entry in self._entries.items()},
        }

    def load_state_dict(self, state: dict) -> None:
        # V2 format: {"episode_counter": int, "entries": {...}}
        # V1 format: flat dict of entries (no "episode_counter" key)
        if "episode_counter" in state and "entries" in state:
            # V2 format
            self._episode_counter = state["episode_counter"]
            entries = state["entries"]
        elif "entries" in state:
            # Partial V2 (entries key but no counter)
            self._episode_counter = state.get("episode_counter", 0)
            entries = state["entries"]
        else:
            # V1 format: entire dict is entries
            self._episode_counter = 0
            entries = state

        for eid, data in entries.items():
            if isinstance(data, dict) and "tier" in data:
                # Filter to known MemoryEntry fields to handle forward-compat
                known_fields = {
                    "entry_id", "tier", "content", "metadata", "created_at",
                    "quality_score", "retrieval_count", "usefulness_sum",
                }
                filtered = {k: v for k, v in data.items() if k in known_fields}
                self._entries[eid] = MemoryEntry(**filtered)

    def save(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(self.state_dict(), f, indent=2)

    def load(self, path: str | Path) -> None:
        with open(path) as f:
            self.load_state_dict(json.load(f))

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        return len(self._entries)

    def tier_counts(self) -> dict[str, int]:
        counts = {"episodic": 0, "semantic": 0, "procedural": 0}
        for entry in self._entries.values():
            counts[entry.tier] = counts.get(entry.tier, 0) + 1
        return counts

    def summary(self) -> dict:
        counts = self.tier_counts()
        return {
            "total": self.size,
            **counts,
            "episode_counter": self._episode_counter,
            "avg_quality": (
                sum(e.quality_score for e in self._entries.values()) / self.size
                if self.size > 0 else 0.0
            ),
        }
