# Baselines

Baseline implementations for comparison with the AEL/EAEL framework.

## Overview

All baselines use the **same tool pipeline** (12 finance tools) and the **same LLM** (Bedrock Haiku) as AEL methods, ensuring fair comparison. They differ only in their learning/memory mechanisms.

## Simple Baselines (No LLM, No Learning)

| Baseline | File | Mechanism | Expected DA |
|----------|------|-----------|-------------|
| Random | `predictors.py` | Uniform random direction | ~33% |
| Buy-and-Hold | `predictors.py` | Always predict "up" | ~47% (market bias) |
| Momentum | `predictors.py` | Follow last-bar direction | ~33% |
| MA Crossover | `predictors.py` | SMA(4) vs SMA(12) signal | ~40% |

**Runner:** `python -m baselines.evaluate --dataset d_small`

## Paper Baselines (LLM-based, Learning)

| Baseline | File | Paper | Year | Key Mechanism |
|----------|------|-------|------|---------------|
| Reflexion | `reflexion.py` | [Shinn et al.](https://arxiv.org/abs/2303.11366) | 2023 | LLM-generated verbal self-critique, accumulated as growing context |
| ExpeL | `expel.py` | [Zhao et al.](https://arxiv.org/abs/2308.10144) | 2024 | LLM-extracted lessons from trajectories, keyword retrieval |
| FactorMiner | `factorminer.py` | [arxiv 2602.14670](https://arxiv.org/abs/2602.14670) | 2026 | Skills (tool combos) + experience memory for financial alpha |
| Meta-Policy Reflexion | `meta_reflexion.py` | [arxiv 2509.03990](https://arxiv.org/abs/2509.03990) | 2025 | Reflections distilled into reusable rules with admissibility check |
| EvoTool | `evotool.py` | [arxiv 2603.04900](https://arxiv.org/abs/2603.04900) | 2026 | Evolutionary tool-use policy optimization with blame attribution |

**Runner:** `python -m baselines.run_baselines --dataset d_small --parallel 3`

## Comparison with AEL

| Feature | Reflexion | ExpeL | FactorMiner | Meta-Reflexion | EvoTool | **AEL** | **EAEL** |
|---------|-----------|-------|-------------|----------------|---------|---------|----------|
| Tool selection | Fixed | Fixed | Skill-guided | Fixed | Evolutionary | Thompson | Per-tool Thompson |
| Planner selection | Fixed | Fixed | Fixed | Fixed | Fixed | LinUCB | LinUCB + evolve |
| Memory | Growing text | Flat lessons | Skills + exp | Distilled rules | None | 3-tier | 3-tier + evolve |
| Credit assignment | None | None | None | None | Blame-aware | FCC + Shapley | FCC + Shapley |
| Reflection | Verbal | Lesson extract | Experience update | Rule distillation | Mutation | Daily reflection | Session reflection |
| Code generation | No | No | No | No | No | No | Planner + memory policy |

## Implementation Fidelity Notes

### Reflexion
- **Faithful:** LLM-generated self-critique (not template), accumulated context window with cap.
- **Domain adaptation:** No multi-trial retries (financial time series can't be re-done).

### ExpeL
- **Faithful:** LLM-based lesson extraction that generates reusable rules (not templates).
- **Simplified:** Keyword retrieval instead of semantic (documented trade-off). No cross-episode contrastive analysis.

### FactorMiner
- **Structural match:** Skills (tool combos) + experience memory, both with success tracking.
- **Simplified:** No factor hypothesis-test loop (replaced with tool combination tracking). Flat skills (no hierarchical decomposition).

### Meta-Policy Reflexion
- **Faithful:** Reflections distilled into meta-policy rules. Rules tracked for success rate. Low-performing rules pruned (admissibility).
- **Extension:** Periodic distillation via LLM (every N episodes) rather than per-episode.

### EvoTool
- **Faithful:** Population of tool policies. Fitness-proportional selection. Blame-aware mutation removes tools that gave wrong signals.
- **Simplified:** Blame attribution is heuristic (signal contradiction check) rather than causal analysis.

## File Structure

```
baselines/
├── __init__.py              # Package docstring
├── README.md                # This file
├── predictors.py            # Simple baselines (random, buy-hold, momentum, MA)
├── evaluate.py              # Runner for simple baselines (no LLM)
├── reflexion.py             # Reflexion (2023)
├── expel.py                 # ExpeL (2024)
├── factorminer.py           # FactorMiner-style (2026)
├── meta_reflexion.py        # Meta-Policy Reflexion (2025)
├── evotool.py               # EvoTool (2026)
└── run_baselines.py         # Runner for paper baselines (uses LLM)
```

## Related Work Summary

The baselines span the 2023-2026 evolution of self-improving LLM agents:

1. **Reflexion (2023)** -- Foundational verbal RL. Simple but effective.
2. **ExpeL (2024)** -- Structured experience extraction. Adds lesson generalization.
3. **Meta-Policy Reflexion (2025)** -- Bridges Reflexion and structured memory. Adds rule distillation + admissibility.
4. **FactorMiner (2026)** -- Domain-specific (finance). Skills + experience memory. Closest to EAEL.
5. **EvoTool (2026)** -- Evolutionary tool policy. Isolates tool evolution contribution.

**EAEL's unique contribution:** Joint evolution of ALL components (planner + tools + memory policy) with factored credit assignment. No prior work evolves all three simultaneously with principled credit attribution.
