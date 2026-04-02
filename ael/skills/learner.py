"""Skill learner: LLM-based skill extraction from experience.

Extracts three types of skills:
- tool_code: New analysis tools (BaseTool subclasses)
- prompt_template: Reusable prompt snippets from reflections
- strategy: Named strategy playbooks consolidating patterns
"""

from __future__ import annotations

import ast
import json
import logging
from typing import Any

from ael.config import ExperimentConfig
from ael.llm_utils import llm_call
from ael.types import Skill

logger = logging.getLogger(__name__)


class SkillLearner:
    """Extracts skills from agent experience using LLM calls."""

    def __init__(self, config: ExperimentConfig):
        self.config = config

    def extract_all(
        self,
        reflection: dict[str, Any],
        memory_store: Any,
        tool_registry: Any,
        planner_pool: Any,
        config: ExperimentConfig,
        day_results: list[dict] | None = None,
        day_index: int = 0,
    ) -> list[Skill]:
        """Run all enabled extraction methods and return new skills."""
        skills: list[Skill] = []
        sc = config.skills

        if sc.extract_templates:
            template_skills = self.extract_prompt_skills(
                reflection, day_results or [], config, day_index
            )
            skills.extend(template_skills)

        if sc.extract_tools and day_results:
            tool_skills = self.extract_tool_skills(
                day_results, memory_store, config, day_index
            )
            skills.extend(tool_skills)

        if sc.extract_strategies:
            strategy_skills = self.extract_strategy_skills(
                reflection, memory_store, config, day_index
            )
            skills.extend(strategy_skills)

        return skills

    # ------------------------------------------------------------------
    # Tool skill extraction
    # ------------------------------------------------------------------

    def extract_tool_skills(
        self,
        day_results: list[dict],
        memory_store: Any,
        config: ExperimentConfig,
        day_index: int = 0,
    ) -> list[Skill]:
        """Extract new tool code from repeated analysis gaps.

        Triggered when reflection identifies patterns of tool failures or
        missing analysis capabilities.
        """
        pass  # llm_call imported at module level

        # Identify analysis gaps: tools that failed or missing data patterns
        gap_summary = self._identify_analysis_gaps(day_results)
        if not gap_summary:
            return []

        llm_model = config.llm_model

        prompt = f"""You are an expert Python developer building analysis tools for a stock prediction system.

## Analysis Gaps Identified
{gap_summary}

## Base Class (your tool MUST inherit from this):
```python
class BaseTool(ABC):
    name: str = ""
    description: str = ""
    cost_per_call: float = 0.0

    def execute(self, **kwargs) -> Any:
        # Return a dict with analysis results
        ...
```

## Available imports in execute():
- pandas as pd, numpy as np
- Standard library modules (json, math, statistics, collections, etc.)
- kwargs always includes 'ticker' (str)

## Requirements
1. Must inherit BaseTool and implement execute()
2. Must have a unique, descriptive name (snake_case)
3. Must return a dict with analysis results
4. Keep it under 50 lines
5. Do NOT import BaseTool — it's already available
6. Handle missing data gracefully (return empty dict)

Generate ONE new tool that addresses the most impactful gap.
Return ONLY the Python class code, no explanation or markdown fences."""

        try:
            response = llm_call(
                model=llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=1024,
            )
            code_str = _extract_python_code(response.choices[0].message.content)

            tool_instance = self._validate_tool_code(code_str)
            if tool_instance is not None:
                skill = Skill(
                    skill_type="tool_code",
                    name=tool_instance.name,
                    description=tool_instance.description or f"LLM-generated tool: {tool_instance.name}",
                    content=code_str,
                    source_code=code_str,
                    created_day=day_index,
                    derived_from=f"analysis_gaps_day_{day_index}",
                    preconditions={"task_type": "price_prediction"},
                )
                logger.info(f"Extracted tool skill: {tool_instance.name}")
                return [skill]
        except Exception as e:
            logger.warning(f"Tool skill extraction failed: {e}")

        return []

    def _validate_tool_code(self, code_str: str) -> Any:
        """Validate and instantiate LLM-generated tool code."""
        from ael.tools.base import BaseTool

        # Syntax check
        try:
            ast.parse(code_str)
        except SyntaxError as e:
            logger.warning(f"Generated tool has syntax error: {e}")
            return None

        # Execute in isolated namespace
        import numpy as np
        import pandas as pd
        namespace = {
            "BaseTool": BaseTool,
            "ABC": __import__("abc").ABC,
            "abstractmethod": __import__("abc").abstractmethod,
            "np": np,
            "pd": pd,
            "Any": Any,
        }
        try:
            exec(code_str, namespace)
        except Exception as e:
            logger.warning(f"Generated tool execution error: {e}")
            return None

        # Find the new tool class
        for obj in namespace.values():
            if (
                isinstance(obj, type)
                and issubclass(obj, BaseTool)
                and obj is not BaseTool
                and hasattr(obj, "name")
                and obj.name
            ):
                try:
                    instance = obj()
                    # Smoke test with mock ticker
                    result = instance.execute(ticker="AAPL")
                    if isinstance(result, dict):
                        return instance
                except Exception as e:
                    logger.warning(f"Tool smoke test failed: {e}")
                    continue

        return None

    def instantiate_tool(self, skill: Skill) -> Any:
        """Instantiate a BaseTool from a tool_code skill."""
        if skill.skill_type != "tool_code":
            return None
        return self._validate_tool_code(skill.source_code or skill.content)

    @staticmethod
    def _identify_analysis_gaps(day_results: list[dict]) -> str:
        """Identify repeated tool failures or missing analysis types."""
        from collections import Counter

        failed_tools: Counter = Counter()
        low_accuracy_tools: Counter = Counter()

        for r in day_results:
            for tool_name, eval_info in r.get("tool_evals", {}).items():
                correct = eval_info.get("correct")
                if correct is False:
                    low_accuracy_tools[tool_name] += 1
            for tool_name in r.get("failed_tools", []):
                failed_tools[tool_name] += 1

        lines = []
        if failed_tools:
            lines.append("Frequently failing tools:")
            for tool, count in failed_tools.most_common(3):
                lines.append(f"  - {tool}: failed {count} times")
        if low_accuracy_tools:
            lines.append("Low accuracy tools (wrong directional signal):")
            for tool, count in low_accuracy_tools.most_common(3):
                lines.append(f"  - {tool}: incorrect {count} times")

        if not lines:
            return ""
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Prompt template extraction
    # ------------------------------------------------------------------

    def extract_prompt_skills(
        self,
        reflection: dict[str, Any],
        day_results: list[dict],
        config: ExperimentConfig,
        day_index: int = 0,
    ) -> list[Skill]:
        """Extract reusable prompt templates from reflection insights.

        Converts actionable insights into "When {condition}, always {action}" rules.
        """
        pass  # llm_call imported at module level

        key_insight = reflection.get("key_insight", "")
        if not key_insight or key_insight == "LLM reflection unavailable":
            return []

        tool_updates = reflection.get("tool_updates", {})
        tomorrow_strategy = reflection.get("tomorrow_strategy", "")

        llm_model = config.llm_model

        prompt = f"""You are distilling trading insights into reusable prompt templates.

## Today's Key Insight
{key_insight}

## Tool Adjustments
{json.dumps(tool_updates, default=str)}

## Tomorrow's Strategy
{tomorrow_strategy}

## Instructions
If the insight above contains an actionable, generalizable pattern, convert it
into a reusable prompt template. The template should be a concise instruction
that can be injected into a planner's context when similar conditions arise.

Format: "When [specific condition], [specific action/consideration]."

Return JSON:
{{
  "has_template": true/false,
  "template_name": "<short_snake_case_name>",
  "template_text": "When [condition], [action].",
  "conditions": {{
    "sector": "<sector or null>",
    "volatility_regime": "<high/normal or null>",
    "task_type": "price_prediction"
  }}
}}

If the insight is too vague or situation-specific, set has_template to false."""

        try:
            response = llm_call(
                model=llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=512,
                response_format={"type": "json_object"},
            )
            result = json.loads(response.choices[0].message.content)

            if result.get("has_template"):
                conditions = result.get("conditions", {})
                # Remove null values
                conditions = {k: v for k, v in conditions.items() if v is not None}

                skill = Skill(
                    skill_type="prompt_template",
                    name=result["template_name"],
                    description=result.get("template_text", "")[:100],
                    content=result["template_text"],
                    preconditions=conditions,
                    created_day=day_index,
                    derived_from=f"reflection_day_{day_index}",
                )
                logger.info(f"Extracted prompt skill: {skill.name}")
                return [skill]
        except Exception as e:
            logger.warning(f"Prompt skill extraction failed: {e}")

        return []

    # ------------------------------------------------------------------
    # Strategy extraction
    # ------------------------------------------------------------------

    def extract_strategy_skills(
        self,
        reflection: dict[str, Any],
        memory_store: Any,
        config: ExperimentConfig,
        day_index: int = 0,
    ) -> list[Skill]:
        """Extract named strategy playbooks from accumulated reflections.

        Consolidates recurring patterns into structured playbooks with
        conditions, tool sequences, and planner choices.
        """
        pass  # llm_call imported at module level

        # Retrieve recent procedural memories
        procedural_rules = ""
        try:
            hits = memory_store.retrieve(
                query={"task_type": "procedural_strategy"},
                tiers=["procedural"],
                top_k=8,
            )
            if hits:
                procedural_rules = "\n".join(f"- {h.content}" for h in hits)
        except Exception:
            pass

        if not procedural_rules:
            return []

        llm_model = config.llm_model

        prompt = f"""You are a quantitative strategist consolidating trading patterns into named playbooks.

## Accumulated Strategy Rules
{procedural_rules}

## Recent Reflection
Key insight: {reflection.get('key_insight', 'N/A')}
Tool priorities: {json.dumps(reflection.get('tool_updates', {}), default=str)}

## Instructions
Identify ONE recurring strategy pattern from the rules above and formalize it
as a named playbook. A playbook defines:
- Name (e.g., "earnings_momentum", "high_vol_defensive")
- When to apply (conditions)
- Which tools to use and in what order
- Which planner strategy fits best

Return JSON:
{{
  "has_strategy": true/false,
  "strategy": {{
    "name": "<snake_case_name>",
    "description": "<1-2 sentence description>",
    "conditions": {{
      "sector": "<sector or null>",
      "volatility_regime": "<high/normal or null>",
      "near_earnings": <true/false/null>
    }},
    "tool_sequence": ["<tool_1>", "<tool_2>", ...],
    "planner": "<recommended_planner>",
    "rationale": "<why this combo works>"
  }}
}}

If no clear recurring pattern exists, set has_strategy to false."""

        try:
            response = llm_call(
                model=llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=512,
                response_format={"type": "json_object"},
            )
            result = json.loads(response.choices[0].message.content)

            if result.get("has_strategy"):
                strategy = result["strategy"]
                conditions = strategy.get("conditions", {})
                conditions = {k: v for k, v in conditions.items() if v is not None}

                skill = Skill(
                    skill_type="strategy",
                    name=strategy["name"],
                    description=strategy.get("description", ""),
                    content=json.dumps(strategy, default=str),
                    preconditions={"conditions": conditions} if conditions else {},
                    created_day=day_index,
                    derived_from=f"strategy_consolidation_day_{day_index}",
                )
                logger.info(f"Extracted strategy skill: {skill.name}")
                return [skill]
        except Exception as e:
            logger.warning(f"Strategy skill extraction failed: {e}")

        return []


def _extract_python_code(text: str) -> str:
    """Extract Python code from LLM response, stripping markdown fences."""
    if "```python" in text:
        parts = text.split("```python")
        if len(parts) > 1:
            return parts[1].split("```")[0].strip()
    if "```" in text:
        parts = text.split("```")
        if len(parts) > 1:
            code = parts[1]
            if code.startswith("python"):
                code = code[6:]
            return code.strip()
    return text.strip()
