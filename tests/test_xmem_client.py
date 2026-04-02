"""Tests for XMemClient — AEL's XMem server wrapper.

These tests use unittest.mock to avoid requiring a running XMem server.
For integration tests with a live server, set XMEM_LIVE_URL env var.
"""

from __future__ import annotations

import json
import os
import unittest
from unittest.mock import MagicMock, patch

from ael.memory.xmem_client import XMemClient
from ael.types import MemoryHit


class TestXMemClientDisabled(unittest.TestCase):
    """When enabled=False, all ops are no-ops."""

    def setUp(self):
        self.client = XMemClient(enabled=False)

    def test_add_episodic_noop(self):
        self.assertFalse(self.client.add_episodic("test content"))

    def test_add_episode_experience_noop(self):
        self.assertFalse(self.client.add_episode_experience(
            ticker="AAPL", date="2025-01-02", planner_id="sequential",
            tool_evaluations={}, prediction={}, actual={}, outcome_score=0.5,
        ))

    def test_retrieve_empty(self):
        hits = self.client.retrieve({"ticker": "AAPL"})
        self.assertEqual(hits, [])

    def test_auto_distill_noop(self):
        self.assertEqual(self.client.auto_distill(), [])

    def test_promote_to_procedural_noop(self):
        self.assertEqual(self.client.promote_to_procedural(), [])

    def test_state_dict_minimal(self):
        state = self.client.state_dict()
        self.assertIn("episode_counter", state)


class TestXMemClientMocked(unittest.TestCase):
    """Test XMemClient with mocked HTTP calls."""

    def setUp(self):
        self.client = XMemClient(base_url="http://localhost:8200", enabled=True)

    @patch("ael.memory.xmem_client.requests.get")
    def test_is_healthy(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200)
        self.assertTrue(self.client.is_healthy())

    @patch("ael.memory.xmem_client.requests.get")
    def test_is_healthy_unreachable(self, mock_get):
        from requests import ConnectionError
        mock_get.side_effect = ConnectionError("refused")
        self.assertFalse(self.client.is_healthy())

    @patch("ael.memory.xmem_client.requests.post")
    def test_add_episodic(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200, json=lambda: {"id": "abc123"}
        )
        result = self.client.add_episodic(
            "AAPL price went up",
            metadata={"ticker": "AAPL", "sector": "Technology"},
        )
        self.assertTrue(result)
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        self.assertEqual(payload["category"], "episode")
        self.assertIn("AAPL", payload["tags"])

    @patch("ael.memory.xmem_client.requests.post")
    def test_add_episode_experience(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200, json=lambda: {"id": "exp123"}
        )
        result = self.client.add_episode_experience(
            ticker="MSFT", date="2025-01-15", planner_id="decompose",
            tool_evaluations={
                "compute_technicals": {
                    "output_summary": "RSI=65",
                    "signal": "up",
                    "correct": True,
                },
            },
            prediction={"direction": "up", "magnitude": 0.02},
            actual={"direction": "up", "return_pct": 0.015},
            outcome_score=0.7,
            sector="Technology",
        )
        self.assertTrue(result)

    @patch("ael.memory.xmem_client.requests.get")
    @patch("ael.memory.xmem_client.requests.post")
    def test_retrieve(self, mock_post, mock_get):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "results": [
                    {"id": "m1", "content": "AAPL trend up", "type": "raw",
                     "score": 0.85},
                    {"id": "m2", "content": "Tech stocks bullish", "type": "note",
                     "score": 0.72},
                ]
            },
        )
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: [
                {"id": "r1", "content": "STRATEGY: Prioritize compute_momentum",
                 "priority": 0},
            ],
        )

        hits = self.client.retrieve(
            {"ticker": "AAPL", "sector": "Technology"},
            tiers=["episodic", "semantic", "procedural"],
            top_k=5,
        )
        self.assertGreater(len(hits), 0)
        self.assertIsInstance(hits[0], MemoryHit)
        # Should have both search results and procedural rules
        tiers = {h.tier for h in hits}
        self.assertIn("episodic", tiers)  # from raw result

    @patch("ael.memory.xmem_client.requests.post")
    def test_add_procedural(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200, json=lambda: {"ok": True, "id": "proc1"}
        )
        result = self.client.add_procedural("STRATEGY: Prioritize compute_momentum")
        self.assertTrue(result)
        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        self.assertIn("content", payload)

    @patch("ael.memory.xmem_client.requests.post")
    def test_auto_distill(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"notes_created": 3, "raws_consolidated": 15},
        )
        notes = self.client.auto_distill()
        self.assertEqual(len(notes), 3)

    @patch("ael.memory.xmem_client.requests.post")
    def test_mark_useful(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"ok": True, "id": "m1", "importance": 0.8},
        )
        self.client.mark_useful("m1", useful=True)
        mock_post.assert_called_once()

    @patch("ael.memory.xmem_client.requests.get")
    def test_state_dict(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"memories": [], "profile": {}, "procedural": []},
        )
        state = self.client.state_dict()
        self.assertIn("episode_counter", state)
        self.assertIn("memories", state)

    @patch("ael.memory.xmem_client.requests.get")
    def test_tier_counts(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "total": 50,
                "by_type": {"raw": 30, "note": 15, "session": 0},
                "procedural_rules": 5,
            },
        )
        counts = self.client.tier_counts()
        self.assertEqual(counts["episodic"], 30)
        self.assertEqual(counts["semantic"], 15)
        self.assertEqual(counts["procedural"], 5)

    def test_increment_episode(self):
        self.assertEqual(self.client._episode_counter, 0)
        self.client.increment_episode()
        self.assertEqual(self.client._episode_counter, 1)


class TestXMemClientIntegration(unittest.TestCase):
    """Integration tests against a live XMem server.

    Skipped unless XMEM_LIVE_URL is set (e.g., http://127.0.0.1:8200).
    """

    @classmethod
    def setUpClass(cls):
        url = os.environ.get("XMEM_LIVE_URL")
        if not url:
            raise unittest.SkipTest("XMEM_LIVE_URL not set; skipping integration tests")
        cls.client = XMemClient(base_url=url, enabled=True)
        if not cls.client.is_healthy():
            raise unittest.SkipTest(f"XMem server not reachable at {url}")

    def test_roundtrip_episodic(self):
        """Add episodic memory and retrieve it."""
        self.client.add_episodic(
            "Integration test: AAPL went up 2% on strong earnings",
            metadata={"ticker": "AAPL", "sector": "Technology"},
        )
        hits = self.client.retrieve(
            {"ticker": "AAPL"},
            tiers=["episodic"],
            top_k=3,
        )
        self.assertGreater(len(hits), 0)
        # At least one hit should mention AAPL
        contents = " ".join(h.content for h in hits)
        self.assertIn("AAPL", contents)

    def test_procedural_roundtrip(self):
        """Add and retrieve a procedural rule."""
        self.client.add_procedural(
            "TEST STRATEGY: Always check RSI before buying"
        )
        hits = self.client.retrieve(
            {"task_type": "price_prediction"},
            tiers=["procedural"],
            top_k=10,
        )
        contents = " ".join(h.content for h in hits)
        self.assertIn("RSI", contents)

    def test_stats(self):
        """Verify stats endpoint returns valid data."""
        counts = self.client.tier_counts()
        self.assertIn("episodic", counts)
        self.assertIn("semantic", counts)
        self.assertIn("procedural", counts)


if __name__ == "__main__":
    unittest.main()
