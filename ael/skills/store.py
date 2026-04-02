"""Skill store: manages learned tools, prompt templates, and strategy playbooks."""

from __future__ import annotations

import json
import logging
from typing import Any

from ael.types import Skill

logger = logging.getLogger(__name__)


class SkillStore:
    """Registry and lifecycle manager for learned skills.

    Three skill types:
    - tool_code: LLM-written Python tool (BaseTool subclass)
    - prompt_template: Reusable prompt snippet for specific scenarios
    - strategy: Named strategy playbook (tool+planner combo + conditions)
    """

    def __init__(self, max_per_type: int = 15):
        self.skills: dict[str, Skill] = {}
        self.max_per_type = max_per_type

    def add_skill(self, skill: Skill) -> bool:
        """Validate and register a new skill. Returns True if added."""
        if not skill.name or not skill.skill_type:
            logger.warning("Skill missing name or type, skipping")
            return False

        # Check capacity per type
        type_count = sum(
            1 for s in self.skills.values()
            if s.skill_type == skill.skill_type and s.active
        )
        if type_count >= self.max_per_type:
            # Prune worst performer of this type to make room
            pruned = self._prune_worst(skill.skill_type)
            if not pruned:
                logger.info(f"Skill store full for type '{skill.skill_type}', skipping")
                return False

        # Check for duplicate names
        for existing in self.skills.values():
            if existing.name == skill.name and existing.active:
                logger.info(f"Skill '{skill.name}' already exists, updating")
                existing.content = skill.content
                existing.description = skill.description
                existing.preconditions = skill.preconditions
                return True

        self.skills[skill.skill_id] = skill
        logger.info(f"Added skill: {skill.name} (type={skill.skill_type})")
        return True

    def get_applicable(
        self,
        task: dict,
        scenario: dict | None = None,
    ) -> list[Skill]:
        """Return active skills whose preconditions match the task/scenario."""
        applicable = []
        for skill in self.skills.values():
            if not skill.active:
                continue
            if self._matches_preconditions(skill, task, scenario):
                applicable.append(skill)
        return applicable

    def get_by_type(self, skill_type: str) -> list[Skill]:
        """Return all active skills of a given type."""
        return [
            s for s in self.skills.values()
            if s.skill_type == skill_type and s.active
        ]

    def update_performance(self, skill_id: str, outcome_score: float, baseline: float = 0.0) -> None:
        """Track success/failure for a skill based on episode outcome.

        Uses differential scoring (outcome > baseline) instead of raw sign
        to avoid marking all skills as failures in declining markets.
        """
        skill = self.skills.get(skill_id)
        if skill is None:
            return
        skill.episode_count += 1
        if outcome_score > baseline:
            skill.success_count += 1
        else:
            skill.failure_count += 1

    def prune(
        self,
        min_episodes: int = 20,
        min_success_rate: float = 0.4,
    ) -> list[str]:
        """Deactivate underperforming skills that have had enough trials."""
        pruned = []
        for skill in self.skills.values():
            if not skill.active:
                continue
            if skill.episode_count >= min_episodes and skill.success_rate < min_success_rate:
                skill.active = False
                pruned.append(skill.name)
                logger.info(
                    f"Pruned skill '{skill.name}': "
                    f"{skill.success_rate:.0%} over {skill.episode_count} episodes"
                )
        return pruned

    def _prune_worst(self, skill_type: str) -> bool:
        """Remove the worst-performing active skill of this type."""
        candidates = [
            s for s in self.skills.values()
            if s.skill_type == skill_type and s.active and s.episode_count > 0
        ]
        if not candidates:
            return False
        worst = min(candidates, key=lambda s: s.success_rate)
        worst.active = False
        logger.info(f"Auto-pruned skill '{worst.name}' (success_rate={worst.success_rate:.0%})")
        return True

    @staticmethod
    def _matches_preconditions(
        skill: Skill,
        task: dict,
        scenario: dict | None,
    ) -> bool:
        """Check if a skill's preconditions match the current task/scenario."""
        preconds = skill.preconditions
        if not preconds:
            return True  # no preconditions = always applicable

        # Check task_type match
        if "task_type" in preconds:
            if task.get("task_type") != preconds["task_type"]:
                return False

        # Check sector match
        if "sector" in preconds:
            if task.get("sector") != preconds["sector"]:
                return False

        # Check ticker match
        if "tickers" in preconds:
            if task.get("ticker") not in preconds["tickers"]:
                return False

        # Check scenario conditions (e.g., volatility_regime, near_earnings)
        if scenario and "conditions" in preconds:
            for key, val in preconds["conditions"].items():
                if scenario.get(key) != val:
                    return False

        return True

    def state_dict(self) -> dict:
        """Serialize for persistence."""
        from dataclasses import asdict
        return {
            sid: asdict(skill)
            for sid, skill in self.skills.items()
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore from serialized state."""
        self.skills.clear()
        for sid, data in state.items():
            self.skills[sid] = Skill(**data)

    @property
    def size(self) -> int:
        return sum(1 for s in self.skills.values() if s.active)

    def summary(self) -> dict[str, Any]:
        """Return summary stats."""
        by_type: dict[str, int] = {}
        for s in self.skills.values():
            if s.active:
                by_type[s.skill_type] = by_type.get(s.skill_type, 0) + 1
        return {
            "total_active": self.size,
            "by_type": by_type,
            "total_ever": len(self.skills),
        }
