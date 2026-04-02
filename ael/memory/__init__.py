"""AEL memory subsystem — local store and XMem-backed client."""

import logging

from ael.memory.store import MemoryStore
from ael.memory.xmem_client import XMemClient

logger = logging.getLogger(__name__)

# Health issues tracked across the experiment for post-hoc reporting
_health_issues: list[dict] = []


def get_health_issues() -> list[dict]:
    """Return accumulated health issues (XMem fallbacks, etc.)."""
    return list(_health_issues)


def create_memory_backend(config) -> MemoryStore | XMemClient:
    """Factory: create the appropriate memory backend from config.

    Args:
        config: MemoryConfig dataclass (has backend, xmem_url, etc.).

    Returns:
        MemoryStore (local) or XMemClient (XMem server).
    """
    enabled = bool(config.enabled_tiers)

    if config.backend == "xmem":
        client = XMemClient(
            base_url=config.xmem_url,
            top_k=config.retrieval_top_k,
            enabled=enabled,
        )
        if enabled and not client.is_healthy():
            logger.warning(
                f"XMem server NOT reachable at {config.xmem_url}. "
                f"Falling back to local MemoryStore. "
                f"Start XMem server or switch to backend: local in config."
            )
            _health_issues.append({
                "type": "xmem_fallback",
                "url": config.xmem_url,
                "reason": "server unreachable",
            })
            return MemoryStore(
                retrieval_top_k=config.retrieval_top_k,
                write_quality_threshold=config.write_quality_threshold,
                eviction_min_score=config.eviction_min_score,
                max_entries_per_tier=config.max_entries_per_tier,
            )
        return client

    return MemoryStore(
        retrieval_top_k=config.retrieval_top_k,
        write_quality_threshold=config.write_quality_threshold,
        eviction_min_score=config.eviction_min_score,
        max_entries_per_tier=config.max_entries_per_tier,
    )

__all__ = ["MemoryStore", "XMemClient", "create_memory_backend", "get_health_issues"]
