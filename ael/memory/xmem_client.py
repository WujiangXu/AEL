"""XMemClient — drop-in replacement for MemoryStore backed by XMem HTTP API.

XMem provides hybrid vector+BM25 search, temporal decay, Zettelkasten
consolidation, and always-loaded procedural rules. This client wraps the
XMem server's REST API to provide the same public interface as MemoryStore.

Server must be started separately:
    python xmem_server.py --port 8200 --preset gpt4o-mini --db-path /tmp/ael_experiment.db
"""

from __future__ import annotations

import json
import logging
from typing import Any

import requests

from ael.types import MemoryHit

logger = logging.getLogger(__name__)


class XMemClient:
    """Memory backend that delegates to an XMem server over HTTP.

    Drop-in replacement for MemoryStore. When enabled=False, all writes
    are no-ops and retrieve returns [] (stateless ablation mode).
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8200",
        top_k: int = 5,
        enabled: bool = True,
        timeout: float = 10.0,
    ):
        self.base_url = base_url.rstrip("/")
        self._top_k = top_k
        self._enabled = enabled
        self._timeout = timeout
        self._episode_counter = 0
        self._read_only = False

    def set_read_only(self, read_only: bool) -> None:
        """Enable/disable read-only mode. When read-only, all writes are skipped."""
        self._read_only = read_only

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def is_healthy(self) -> bool:
        """Check if XMem server is reachable."""
        try:
            r = requests.get(f"{self.base_url}/health", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Write: episodic (raw) memories
    # ------------------------------------------------------------------

    def add_episodic(
        self,
        content: str,
        metadata: dict | None = None,
        quality: float = 0.5,
    ) -> bool:
        """Add an episodic memory as an XMem raw entry (zero LLM cost)."""
        if not self._enabled or self._read_only:
            return False

        metadata = metadata or {}
        tags = []
        if "ticker" in metadata:
            tags.append(metadata["ticker"])
        if "planner" in metadata:
            tags.append(metadata["planner"])
        if "sector" in metadata:
            tags.append(metadata["sector"])

        payload = {
            "content": content,
            "category": "episode",
            "tags": tags,
        }
        return self._post("/memory/add-raw", payload) is not None

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
        """Write a rich episodic memory with per-tool ground-truth evaluations."""
        if not self._enabled or self._read_only:
            return False

        # Build NL content (same format as MemoryStore)
        lines = [f"{ticker}@{date} (planner={planner_id}):"]
        helped, hurt = [], []
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

        tags = [ticker, planner_id]
        if sector:
            tags.append(sector)

        payload = {
            "content": content,
            "category": "episode",
            "tags": tags,
        }
        return self._post("/memory/add-raw", payload) is not None

    # ------------------------------------------------------------------
    # Write: semantic (handled by consolidation, but allow explicit add)
    # ------------------------------------------------------------------

    def add_semantic(
        self,
        content: str,
        metadata: dict | None = None,
        quality: float = 0.6,
    ) -> bool:
        """Add a semantic note. XMem normally creates these via consolidation,
        but this allows explicit insertion for cross-module insights."""
        if not self._enabled or self._read_only:
            return False

        metadata = metadata or {}
        tags = []
        if "planner" in metadata:
            tags.append(metadata["planner"])
        if "ticker" in metadata:
            tags.append(metadata["ticker"])

        payload = {
            "content": content,
            "category": "insight",
            "tags": tags,
        }
        return self._post("/memory/add-raw", payload) is not None

    # ------------------------------------------------------------------
    # Write: procedural rules
    # ------------------------------------------------------------------

    def add_procedural(
        self,
        content: str,
        metadata: dict | None = None,
        quality: float = 0.7,
    ) -> bool:
        """Add a procedural rule (always-loaded behavioral strategy)."""
        if not self._enabled or self._read_only:
            return False

        payload = {"content": content, "priority": 0}
        return self._post("/procedural", payload) is not None

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: dict,
        tiers: list[str] | None = None,
        top_k: int | None = None,
    ) -> list[MemoryHit]:
        """Hybrid vector+BM25 search across XMem memory tiers."""
        if not self._enabled:
            return []

        if top_k is None:
            top_k = self._top_k

        # Build query string from dict
        parts = []
        if query.get("ticker"):
            parts.append(query["ticker"])
        if query.get("sector"):
            parts.append(query["sector"])
        if query.get("task_type"):
            parts.append(query["task_type"])
        if query.get("planner"):
            parts.append(query["planner"])
        if query.get("tool_name"):
            parts.append(query["tool_name"])
        query_str = " ".join(parts) if parts else "stock analysis prediction"

        # Map AEL tiers to XMem tiers
        tier_filter = None
        if tiers:
            if tiers == ["episodic"]:
                tier_filter = "raw"
            elif tiers == ["semantic", "procedural"] or tiers == ["semantic"]:
                tier_filter = "note"
            # else: no filter (search all)

        hits = []

        # Search main memories
        payload: dict[str, Any] = {"query": query_str, "k": top_k}
        if tier_filter:
            payload["tier"] = tier_filter

        data = self._post("/memory/search", payload)
        if data and "results" in data:
            for r in data["results"]:
                hits.append(MemoryHit(
                    memory_id=r.get("id", ""),
                    tier=self._xmem_type_to_ael_tier(r.get("type", "raw")),
                    content=r.get("content", ""),
                    relevance_score=r.get("score", 0.0),
                ))

        # Also fetch procedural rules if requested
        if tiers is None or "procedural" in tiers:
            rules = self._get("/procedural")
            if rules and isinstance(rules, list):
                for rule in rules:
                    hits.append(MemoryHit(
                        memory_id=rule.get("id", ""),
                        tier="procedural",
                        content=rule.get("content", ""),
                        relevance_score=1.0,  # always-loaded = max relevance
                    ))

        return hits[:top_k]

    def retrieve_with_strategy(
        self,
        query: dict,
        memory_policy: dict,
    ) -> list[MemoryHit]:
        """EAEL v2: Retrieve memories and apply formatting strategy.

        Strategies:
        - 'none': return empty
        - 'full': return all retrieved memories verbatim
        - 'sliding_window': keep first N + last M memories
        - 'llm_summary': compress memories into a summary
        - 'ranked_truncate': sort by relevance, cut at token budget

        Args:
            query: Retrieval query dict.
            memory_policy: Full memory policy dict with format, budgets, etc.

        Returns:
            Formatted list of MemoryHit.
        """
        fmt = memory_policy.get("format", "full")
        top_k = memory_policy.get("retrieval_top_k", 0)
        tiers = memory_policy.get("enabled_tiers", [])

        if fmt == "none" or top_k == 0:
            return []

        # Base retrieval
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
            # Sort by relevance (highest first)
            hits.sort(key=lambda h: h.relevance_score, reverse=True)
            selected = []
            tokens_used = 0
            for h in hits:
                # Rough token estimate: ~4 chars per token
                est_tokens = len(h.content) // 4
                if tokens_used + est_tokens > token_budget and selected:
                    break
                selected.append(h)
                tokens_used += est_tokens
            return selected

        if fmt == "llm_summary":
            summary_budget = memory_policy.get("summary_budget", 200)
            # Compress all memories into a single summary hit
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
                # Fallback: truncate manually
                truncated = combined[:summary_budget * 4]
                return [MemoryHit(
                    memory_id="__summary__",
                    tier="semantic",
                    content=truncated,
                    relevance_score=0.8,
                )]

        # Unknown format → return as-is
        return hits

    @staticmethod
    def _xmem_type_to_ael_tier(xmem_type: str) -> str:
        """Map XMem memory type to AEL tier name."""
        return {"raw": "episodic", "note": "semantic", "session": "semantic"}.get(
            xmem_type, "episodic"
        )

    # ------------------------------------------------------------------
    # Auto-distillation (consolidation)
    # ------------------------------------------------------------------

    def auto_distill(self, min_episodes: int = 10) -> list[str]:
        """Trigger XMem consolidation (raw episodes → semantic notes)."""
        if not self._enabled:
            return []

        data = self._post("/memory/consolidate?force=false", {})
        if data:
            n_created = data.get("notes_created", 0)
            if n_created > 0:
                logger.info(f"XMem consolidation: {n_created} notes created")
            return [f"note_{i}" for i in range(n_created)]
        return []

    # ------------------------------------------------------------------
    # Procedural promotion
    # ------------------------------------------------------------------

    def promote_to_procedural(
        self,
        min_samples: int = 10,
        min_confidence: float = 0.6,
    ) -> list[str]:
        """Search for high-confidence semantic notes and promote to procedural rules.

        This replicates the MemoryStore pattern: find consistent notes,
        create procedural rules from them.
        """
        if not self._enabled:
            return []

        created = []

        # Search for tool accuracy patterns
        data = self._post("/memory/search", {
            "query": "tool reliable accuracy directional",
            "k": 10,
            "tier": "note",
        })
        if data and "results" in data:
            for r in data["results"]:
                content = r.get("content", "")
                # Only promote high-confidence findings
                if ("reliable" in content.lower() or "unreliable" in content.lower()) \
                        and r.get("score", 0) > 0.5:
                    rule_content = f"STRATEGY: {content}"
                    resp = self._post("/procedural", {
                        "content": rule_content,
                        "priority": 0,
                    })
                    if resp:
                        created.append(resp.get("id", ""))

        return created

    # ------------------------------------------------------------------
    # Feedback
    # ------------------------------------------------------------------

    def mark_useful(self, memory_id: str, useful: bool) -> None:
        """Adjust memory importance based on usefulness feedback."""
        if not self._enabled or not memory_id:
            return
        weight = 0.8 if useful else 0.2
        self._post(f"/memory/{memory_id}/important", {"weight": weight})

    # ------------------------------------------------------------------
    # Episode counter
    # ------------------------------------------------------------------

    def increment_episode(self) -> None:
        self._episode_counter += 1

    # ------------------------------------------------------------------
    # Persistence (export/import)
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Export full XMem state for persistence."""
        data = self._get("/memory/export")
        if data:
            data["episode_counter"] = self._episode_counter
            return data
        return {"episode_counter": self._episode_counter}

    def load_state_dict(self, state: dict) -> None:
        """Import XMem state from a previous export."""
        self._episode_counter = state.get("episode_counter", 0)
        # Only import if there's actual memory data
        if "memories" in state:
            self._post("/memory/import", state)

    def save(self, path) -> None:
        """Save XMem state to file."""
        import json
        from pathlib import Path
        state = self.state_dict()
        with open(Path(path), "w") as f:
            json.dump(state, f, indent=2)

    def load(self, path) -> None:
        """Load XMem state from file."""
        import json
        from pathlib import Path
        with open(Path(path)) as f:
            self.load_state_dict(json.load(f))

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        stats = self._get("/memory/stats")
        return stats.get("total", 0) if stats else 0

    def tier_counts(self) -> dict[str, int]:
        stats = self._get("/memory/stats")
        if stats and "by_type" in stats:
            by_type = stats["by_type"]
            return {
                "episodic": by_type.get("raw", 0),
                "semantic": by_type.get("note", 0),
                "procedural": stats.get("procedural_rules", 0),
            }
        return {"episodic": 0, "semantic": 0, "procedural": 0}

    def summary(self) -> dict:
        counts = self.tier_counts()
        return {
            "total": self.size,
            **counts,
            "episode_counter": self._episode_counter,
            "backend": "xmem",
        }

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _post(self, endpoint: str, payload: dict) -> dict | None:
        """POST to XMem server, return JSON response or None on failure."""
        try:
            r = requests.post(
                f"{self.base_url}{endpoint}",
                json=payload,
                timeout=self._timeout,
            )
            if r.status_code in (200, 201):
                return r.json()
            logger.warning(f"XMem POST {endpoint} returned {r.status_code}: {r.text[:200]}")
        except requests.ConnectionError:
            logger.warning(f"XMem server unreachable at {self.base_url}")
        except Exception as e:
            logger.warning(f"XMem POST {endpoint} failed: {e}")
        return None

    def _get(self, endpoint: str, params: dict | None = None) -> dict | list | None:
        """GET from XMem server, return JSON response or None on failure."""
        try:
            r = requests.get(
                f"{self.base_url}{endpoint}",
                params=params,
                timeout=self._timeout,
            )
            if r.status_code == 200:
                return r.json()
            logger.warning(f"XMem GET {endpoint} returned {r.status_code}: {r.text[:200]}")
        except requests.ConnectionError:
            logger.warning(f"XMem server unreachable at {self.base_url}")
        except Exception as e:
            logger.warning(f"XMem GET {endpoint} failed: {e}")
        return None
