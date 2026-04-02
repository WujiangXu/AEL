"""HyperAgent-style recursive self-modification baseline for portfolio allocation.

Implements Meta AI's Hyperagents (arxiv 2603.19461) core mechanism:
the agent writes its own allocation strategy as Python code, evaluates it,
and a meta-agent proposes improvements — including modifying the improvement
procedure itself (recursive metacognition).

Key difference from AEL:
- HyperAgent: rewrites the ENTIRE allocation function each generation (batch evolution)
- AEL: evolves modular components (planners, tools, memory) online with credit assignment

Fair comparison: same tools, same data, same LLM, same test-phase freeze.

Usage:
    python -m baselines.run_hyperagent --dataset d_full --seed 42 --generations 10
"""
from __future__ import annotations

import ast
import json
import logging
import re
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ── Initial task agent: simple equal-weight allocation ──────────────────

INITIAL_TASK_AGENT = '''
def allocate(bar_time, tickers, tool_outputs, portfolio_state):
    """Allocate portfolio weights across tickers.

    Args:
        bar_time: Current timestamp string
        tickers: List of ticker symbols
        tool_outputs: Dict {ticker: {tool_name: output_dict}} with signals from 12 finance tools
        portfolio_state: Dict with current portfolio info (weights, total_value, history)

    Returns:
        Dict {ticker: weight} where weights sum to 1.0
    """
    n = len(tickers)
    return {t: 1.0 / n for t in tickers}
'''

# ── Initial meta-agent improvement prompt template ─────────────────────

INITIAL_META_PROMPT = '''You are a meta-agent improving a portfolio allocation strategy.

## Current Strategy Code
```python
{task_agent_code}
```

## Performance History
{generation_history}

## Available Tool Outputs — EXACT KEY STRUCTURE
`tool_outputs` is a dict: `tool_outputs[ticker][tool_name]` → dict of signals.
Here are the EXACT keys (use these, not guessed names):

```
tool_outputs["AAPL"]["compute_technicals"] = {{
    "rsi_14": {{"value": 60.0, "signal": "NEUTRAL"}},
    "macd": {{"macd": -0.92, "signal": -1.73, "histogram": 0.81}},
    "bollinger": {{"upper": 217.8, "middle": 214.4, "lower": 211.0}},
    "technical_score": 0.55,
    "technical_outlook": "NEUTRAL",
    "moving_averages": {{"sma_50": 220.2, "sma_200": 231.9, "above_50": False}},
}}
tool_outputs["AAPL"]["compute_momentum"] = {{
    "return_4bar": 0.012, "return_8bar": -0.005, "return_20bar": 0.03,
    "return_60bar": -0.08, "trend": "bullish", "volume_trend": "rising",
}}
tool_outputs["AAPL"]["compute_quant_risk"] = {{
    "volatility": {{"annual": 0.35, "daily": 0.022}},
    "var": {{"95": -0.034}},
    "risk_return": {{"sharpe": 0.5, "sortino": 0.8}},
}}
tool_outputs["AAPL"]["score_composite_signal"] = {{
    "signal": "BUY", "composite_score": 0.62, "confidence": "medium",
}}
tool_outputs["AAPL"]["score_risk"] = {{
    "overall": 5.2, "rating": "MODERATE",
    "scores": {{"valuation": 6, "financial": 4, "growth": 5}},
}}
```

IMPORTANT: Always wrap tool_outputs access in try/except — some tickers may be missing keys.
You may use `np` (numpy) in the function body.

## Instructions
Write an IMPROVED version of the `allocate()` function.
The function signature must remain: `allocate(bar_time, tickers, tool_outputs, portfolio_state)`

Key improvement areas:
- Use tool outputs to make informed allocations (not just equal weight)
- Consider momentum signals for trend-following
- Consider risk scores for defensive positioning
- Consider correlations for diversification
- ALWAYS include a try/except fallback to equal weight

Return ONLY the complete Python function (no explanation, no markdown).
Start with `def allocate(`:'''


# ── Meta-modification prompt (recursive self-improvement) ──────────────

META_MODIFY_PROMPT = '''You are improving your own improvement procedure.

## Current Meta-Agent Prompt Template
{meta_prompt}

## Generation Performance Trend
{performance_trend}

## What to Improve
The meta-prompt above is used to guide improvements to a portfolio allocation strategy.
Based on the performance trend, modify the meta-prompt to give BETTER improvement guidance.

For example:
- If performance plateaued → add more specific technical analysis instructions
- If drawdowns are high → emphasize risk management in the prompt
- If returns are inconsistent → add regime-awareness instructions

Return the COMPLETE improved meta-prompt template (must contain {{task_agent_code}} and {{generation_history}} placeholders).
Start with: You are a meta-agent'''


@dataclass
class GenerationResult:
    """Record of one generation's evaluation."""
    gen_idx: int
    train_sharpe: float
    train_return: float
    code_hash: str
    accepted: bool
    meta_modified: bool = False


class HyperAgentPortfolio:
    """HyperAgent-style recursive self-modification for portfolio allocation.

    Core loop (per generation):
    1. Meta-agent reads current task_agent code + performance history
    2. Meta-agent proposes code modifications (new allocation function)
    3. Evaluate modified agent on training data (all bars)
    4. If improved → accept; else → revert
    5. After generation 3, meta-agent can also modify its own prompt
    """

    def __init__(
        self,
        model: str,
        tool_registry: Any,
        n_generations: int = 20,
        meta_modify_interval: int = 3,
        accept_epsilon: float = 0.05,
        temperature: float = 0.7,
    ):
        self.model = model
        self.tool_registry = tool_registry
        self.n_generations = n_generations
        self.meta_modify_interval = meta_modify_interval
        self.accept_epsilon = accept_epsilon
        self.temperature = temperature

        # Evolvable state
        self.task_agent_code = INITIAL_TASK_AGENT
        self.meta_prompt = INITIAL_META_PROMPT
        self.best_sharpe = -float("inf")
        self.best_code = INITIAL_TASK_AGENT

        # History
        self.generation_history: list[GenerationResult] = []
        self.evolution_log: list[dict] = []

    def run_evolution(
        self,
        train_bars: list[dict],
        tickers: list[str],
        tool_outputs_by_bar: dict[str, dict],
        portfolio_histories: list[dict] | None = None,
    ) -> None:
        """Run the full evolutionary loop over N generations."""
        from ael.llm_utils import llm_call

        for gen_idx in range(self.n_generations):
            logger.info(f"[HyperAgent] Generation {gen_idx}/{self.n_generations}")

            # 1. Meta-agent proposes improved code
            history_str = self._format_history()
            # Use replace instead of format() to avoid breaking on { } in code
            prompt = self.meta_prompt.replace(
                "{task_agent_code}", self.task_agent_code
            ).replace(
                "{generation_history}", history_str
            )

            try:
                response = llm_call(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=2000,
                )
                new_code = response.choices[0].message.content.strip()
                new_code = self._clean_code(new_code)
            except Exception as e:
                logger.warning(f"[HyperAgent] Gen {gen_idx} LLM failed: {e}")
                self.generation_history.append(GenerationResult(
                    gen_idx=gen_idx, train_sharpe=self.best_sharpe,
                    train_return=0, code_hash="failed", accepted=False,
                ))
                continue

            # 2. Validate the code (syntax + safety)
            if not self._validate_code(new_code):
                logger.warning(f"[HyperAgent] Gen {gen_idx} code validation failed")
                self.generation_history.append(GenerationResult(
                    gen_idx=gen_idx, train_sharpe=self.best_sharpe,
                    train_return=0, code_hash="invalid", accepted=False,
                ))
                continue

            # 3. Evaluate on training data
            train_sharpe, train_return = self._evaluate(
                new_code, train_bars, tickers, tool_outputs_by_bar,
            )

            # 4. Accept if improved (or within epsilon of best)
            accepted = train_sharpe >= (self.best_sharpe - self.accept_epsilon)
            if accepted:
                self.task_agent_code = new_code
                if train_sharpe > self.best_sharpe:
                    self.best_sharpe = train_sharpe
                    self.best_code = new_code
                logger.info(
                    f"[HyperAgent] Gen {gen_idx} ACCEPTED: "
                    f"sharpe={train_sharpe:.3f} return={train_return:.3%}"
                )
            else:
                logger.info(
                    f"[HyperAgent] Gen {gen_idx} rejected: "
                    f"sharpe={train_sharpe:.3f} < best={self.best_sharpe:.3f}"
                )

            code_hash = str(hash(new_code))[:8]
            result = GenerationResult(
                gen_idx=gen_idx,
                train_sharpe=train_sharpe,
                train_return=train_return,
                code_hash=code_hash,
                accepted=accepted,
            )

            # 5. Recursive meta-modification (after initial generations)
            if (gen_idx > 0
                    and gen_idx % self.meta_modify_interval == 0
                    and len(self.generation_history) >= 2):
                self._meta_modify()
                result.meta_modified = True

            self.generation_history.append(result)
            self.evolution_log.append({
                "gen": gen_idx,
                "sharpe": train_sharpe,
                "return": train_return,
                "accepted": accepted,
                "code_length": len(new_code),
                "meta_modified": result.meta_modified,
            })

        # Ensure we use the best code found
        self.task_agent_code = self.best_code

    def allocate(
        self,
        bar_time: str,
        tickers: list[str],
        tool_outputs: dict[str, dict],
        portfolio_state: dict,
    ) -> dict[str, float]:
        """Run the evolved allocation function (frozen, for test phase)."""
        try:
            ns = {}
            exec(self.task_agent_code, {"np": np, "json": json}, ns)
            allocate_fn = ns["allocate"]
            weights = allocate_fn(bar_time, tickers, tool_outputs, portfolio_state)

            # Validate weights
            if not isinstance(weights, dict):
                return self._equal_weight(tickers)
            total = sum(abs(v) for v in weights.values())
            if total == 0:
                return self._equal_weight(tickers)
            # Normalize
            return {t: max(0, weights.get(t, 0)) / total for t in tickers}

        except Exception as e:
            logger.warning(f"[HyperAgent] Allocation failed: {e}")
            return self._equal_weight(tickers)

    def _evaluate(
        self,
        code: str,
        train_bars: list[dict],
        tickers: list[str],
        tool_outputs_by_bar: dict[str, dict],
    ) -> tuple[float, float]:
        """Evaluate an allocation function on training data. Returns (sharpe, total_return)."""
        try:
            ns = {}
            exec(code, {"np": np, "json": json, "defaultdict": defaultdict}, ns)
            allocate_fn = ns["allocate"]
        except Exception:
            return -10.0, 0.0

        returns = []
        portfolio_state = {"weights": {}, "total_value": 1.0, "history": []}

        for bar in train_bars:
            bar_time = bar["bar_time"]
            ticker_returns = bar.get("ticker_returns", {})
            tool_out = tool_outputs_by_bar.get(bar_time, {})

            try:
                weights = allocate_fn(bar_time, tickers, tool_out, portfolio_state)
                if not isinstance(weights, dict):
                    weights = self._equal_weight(tickers)
                # Normalize
                total = sum(abs(v) for v in weights.values())
                if total > 0:
                    weights = {t: max(0, weights.get(t, 0)) / total for t in tickers}
                else:
                    weights = self._equal_weight(tickers)
            except Exception:
                weights = self._equal_weight(tickers)

            # Compute portfolio return
            port_ret = sum(weights.get(t, 0) * ticker_returns.get(t, 0) for t in tickers)
            returns.append(port_ret)

            # Update portfolio state for next bar
            portfolio_state["weights"] = weights
            portfolio_state["total_value"] *= (1 + port_ret)
            portfolio_state["history"].append({
                "bar_time": bar_time, "return": port_ret,
                "value": portfolio_state["total_value"],
            })

        if not returns:
            return -10.0, 0.0

        arr = np.array(returns)
        mean_ret = float(np.mean(arr))
        vol = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.01
        ann_factor = 252 * 4
        sharpe = (mean_ret * ann_factor) / (vol * np.sqrt(ann_factor)) if vol > 0 else 0
        total_ret = float(np.prod(1 + arr) - 1)

        return sharpe, total_ret

    def _meta_modify(self) -> None:
        """Recursive self-modification: improve the meta-agent's own prompt."""
        from ael.llm_utils import llm_call

        trend = "\n".join(
            f"Gen {r.gen_idx}: sharpe={r.train_sharpe:.3f}, accepted={r.accepted}"
            for r in self.generation_history[-5:]
        )

        try:
            modify_prompt = META_MODIFY_PROMPT.replace(
                "{meta_prompt}", self.meta_prompt[:1000]
            ).replace(
                "{performance_trend}", trend
            )
            response = llm_call(
                model=self.model,
                messages=[{"role": "user", "content": modify_prompt}],
                temperature=0.3,
                max_tokens=1000,
            )
            new_prompt = response.choices[0].message.content.strip()

            # Validate the new prompt has required placeholders
            if "{task_agent_code}" in new_prompt and "{generation_history}" in new_prompt:
                self.meta_prompt = new_prompt
                logger.info("[HyperAgent] Meta-prompt modified (recursive self-improvement)")
            else:
                logger.warning("[HyperAgent] Meta-modification rejected: missing placeholders")
        except Exception as e:
            logger.warning(f"[HyperAgent] Meta-modification failed: {e}")

    def _validate_code(self, code: str) -> bool:
        """Validate that code is syntactically correct and safe."""
        try:
            # Strip import lines for numpy/json (they're pre-injected via exec namespace)
            clean_lines = []
            for line in code.split("\n"):
                stripped = line.strip()
                if stripped in ("import numpy as np", "import numpy", "import json",
                                "from collections import defaultdict"):
                    continue  # skip — these are already available in exec namespace
                clean_lines.append(line)
            code = "\n".join(clean_lines)

            tree = ast.parse(code)
            # Must contain a function named 'allocate'
            func_names = [
                node.name for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef)
            ]
            if "allocate" not in func_names:
                return False

            # Block dangerous imports/operations
            code_lower = code.lower()
            blocked = ["import os", "import sys", "subprocess", "eval(", "exec(",
                       "__import__", "open(", "file("]
            if any(b in code_lower for b in blocked):
                return False

            return True
        except SyntaxError:
            return False

    def _clean_code(self, code: str) -> str:
        """Clean LLM output to extract just the Python function."""
        # Remove markdown code blocks
        code = re.sub(r"```python\n?", "", code)
        code = re.sub(r"```\n?", "", code)
        code = code.strip()

        # Strip common import lines (numpy/json are pre-injected)
        clean_lines = []
        for line in code.split("\n"):
            stripped = line.strip()
            if stripped in ("import numpy as np", "import numpy", "import json",
                            "from collections import defaultdict"):
                continue
            clean_lines.append(line)
        code = "\n".join(clean_lines).strip()

        # Ensure it starts with 'def allocate'
        if "def allocate" in code:
            idx = code.index("def allocate")
            code = code[idx:]

        return code

    def _format_history(self) -> str:
        """Format generation history for the meta-agent."""
        if not self.generation_history:
            return "No previous generations."
        lines = []
        for r in self.generation_history[-5:]:
            status = "ACCEPTED" if r.accepted else "rejected"
            meta = " (meta-modified)" if r.meta_modified else ""
            lines.append(
                f"Gen {r.gen_idx}: sharpe={r.train_sharpe:.3f}, "
                f"return={r.train_return:.3%}, {status}{meta}"
            )
        return "\n".join(lines)

    @staticmethod
    def _equal_weight(tickers: list[str]) -> dict[str, float]:
        n = len(tickers)
        return {t: 1.0 / n for t in tickers}

    def get_final_code(self) -> str:
        """Return the evolved allocation function."""
        return self.best_code

    def get_evolution_log(self) -> list[dict]:
        """Return the evolution history."""
        return self.evolution_log
