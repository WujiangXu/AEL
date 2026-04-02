# EAEL Architecture & Experiment Design

## 1. Overall Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    EAEL: Evolving Autonomous Evolving Learner                          │
│                                                                         │
│  ┌─────────────────────── Meta-Controller ────────────────────────┐     │
│  │                                                                │     │
│  │  ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐   │     │
│  │  │ Planner      │   │ Tool         │   │ Memory Policy    │   │     │
│  │  │ Bandit       │   │ Bandit       │   │ Bandit           │   │     │
│  │  │ (LinUCB)     │   │ (Thompson)   │   │ (Thompson)       │   │     │
│  │  │              │   │              │   │                  │   │     │
│  │  │ contextual   │   │ per-tool     │   │ 5 default +      │   │     │
│  │  │ 6+ planners  │   │ 12 tools     │   │ LLM-evolved      │   │     │
│  │  └──────┬───────┘   └──────┬───────┘   └──────┬───────────┘   │     │
│  │         │                  │                   │               │     │
│  │         └──────────┬───────┘───────────────────┘               │     │
│  │                    │ select(task_features)                     │     │
│  └────────────────────┼──────────────────────────────────────────┘     │
│                       ▼                                                 │
│  ┌──────────── Episode Execution ────────────────────────────────┐     │
│  │                                                                │     │
│  │  Task ──→ Planner.plan() ──→ Tool Execution ──→ LLM Synthesis │     │
│  │  (ticker    │                  │                   │           │     │
│  │   + date)   ▼                  ▼                   ▼           │     │
│  │         [PlanSteps]      {tool_outputs}      PredictionResult  │     │
│  │                                                   │           │     │
│  │                                              ┌────▼────┐      │     │
│  │                                              │Evaluate │      │     │
│  │                                              │vs Actual│      │     │
│  │                                              └────┬────┘      │     │
│  │                                                   │           │     │
│  │                                              outcome_score    │     │
│  └───────────────────────────────────────────────────┼───────────┘     │
│                                                      │                  │
│  ┌──────────── Credit Assignment (FCC) ──────────────▼──────────┐     │
│  │                                                               │     │
│  │  Planner credit ─→ LinUCB update                             │     │
│  │  Tool credits   ─→ Thompson update (per-tool directional acc)│     │
│  │  Memory credit  ─→ Thompson update (policy effectiveness)    │     │
│  │                                                               │     │
│  │  Every 20 episodes: Shapley value computation                │     │
│  └───────────────────────────────────────────────────────────────┘     │
│                                                                         │
│  ┌──────────── Memory System ────────────────────────────────────┐     │
│  │                                                                │     │
│  │  Episodic ──auto_distill()──→ Semantic ──promote()──→ Procedural│    │
│  │  (raw exp)    every 10 ep    (patterns)   conf>0.6   (strategies)│   │
│  │                                                          │      │    │
│  │                                               ┌──────────▼──┐   │    │
│  │                                               │Inject into  │   │    │
│  │                                               │Planner      │   │    │
│  │                                               │Prompts      │   │    │
│  │                                               └─────────────┘   │    │
│  └────────────────────────────────────────────────────────────────┘     │
│                                                                         │
│  ┌──────────── Nightly Evolution ────────────────────────────────┐     │
│  │                                                                │     │
│  │  Reflection                                                    │     │
│  │  ├── Daily reflection ──→ insights, critiques                 │     │
│  │  ├── Morning planning  ──→ strategy overrides                 │     │
│  │  └── Cold start (day 0) ──→ bandit priors                    │     │
│  │                                                                │     │
│  │  Evolution (triggered by poor performance)                    │     │
│  │  ├── Planner evolution ──→ new Python planner class (LLM)    │     │
│  │  ├── Memory policy evolution ──→ new retrieval config (LLM)  │     │
│  │  └── Skill extraction ──→ tools, templates, strategies       │     │
│  │                                                                │     │
│  └────────────────────────────────────────────────────────────────┘     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 2. Information Flow (One Episode)

```
                    ┌───────────────┐
                    │  Market Data  │
                    │  (CSV/JSON)   │
                    └───────┬───────┘
                            │
    ┌───────────────────────▼───────────────────────┐
    │              12 Finance Tools                  │
    │                                               │
    │  Data Layer        Compute Layer    Score Layer│
    │  ┌────────────┐   ┌──────────────┐ ┌────────┐│
    │  │prices      │──→│technicals    │─→│risk    ││
    │  │fundamentals│──→│quant_risk    │ │score   ││
    │  │analyst     │──→│momentum      │─→│composite│
    │  │options     │   │correlations  │ │signal  ││
    │  │earnings    │   │dcf_model     │ └────────┘│
    │  └────────────┘   └──────────────┘           │
    └───────────────────────┬───────────────────────┘
                            │ tool_outputs
                            │ (only compute_*/score_*/run_dcf
                            │  passed to LLM)
    ┌───────────────────────▼───────────────────────┐
    │              LLM Predictor                    │
    │         (Bedrock Claude Haiku)                │
    │                                               │
    │  Input (~1400 tokens):                       │
    │  ├── Ticker + date + horizons                │
    │  ├── Computed signals (flat key-value)        │
    │  └── Memory context (if policy selects any)  │
    │                                               │
    │  Output: JSON per horizon                    │
    │  ├── direction: up/down/flat                 │
    │  ├── magnitude: 0.03 (3%)                    │
    │  ├── confidence: 0.7                         │
    │  └── reasoning: "..."                        │
    └───────────────────────┬───────────────────────┘
                            │ PredictionResult
                            │
    ┌───────────────────────▼───────────────────────┐
    │              Evaluation                       │
    │                                               │
    │  vs. Realized Returns (from yfinance):       │
    │  ├── 1-day:  actual close → direction match? │
    │  ├── 7-day:  actual close → direction match? │
    │  └── 14-day: actual close → direction match? │
    │                                               │
    │  Metrics:                                    │
    │  ├── Directional Accuracy (DA)               │
    │  ├── MAPE (magnitude error)                  │
    │  ├── Weighted Accuracy (confidence-weighted) │
    │  └── Outcome Score ∈ [-1, 1]                 │
    └───────────────────────────────────────────────┘
```

## 3. Temporal Structure

### Prediction Horizons

Each episode produces 3 simultaneous predictions:

```
        As-of Date
            │
            ├── 1-day horizon  → "daily"   → next trading day close
            ├── 7-day horizon  → "weekly"  → price 7 calendar days out
            └── 14-day horizon → "biweekly"→ price 14 calendar days out
```

- **1-day (daily)**: Most volatile, hardest to predict. Tests short-term signal processing.
- **7-day (weekly)**: Medium-term trend. Tests fundamental + momentum synthesis.
- **14-day (biweekly)**: Longer-term direction. Tests the agent's ability to filter noise.

Flat threshold: < 0.5% change = "flat" direction.

### Day-Level Loop

The agent operates on a **daily cycle** that mirrors a human trader:

```
═══════════════════════════════════════════════════════════
Day 1 (e.g., 2025-01-02)
═══════════════════════════════════════════════════════════

  ☀ MORNING (once per day)
  │  └── LLM morning planning: analyze yesterday's results,
  │      set today's strategy overrides
  │
  ├── Episode 1: AAPL @ 2025-01-02
  │   └── select planner → run tools → LLM predict → evaluate
  ├── Episode 2: NVDA @ 2025-01-02
  │   └── select planner → run tools → LLM predict → evaluate
  ├── ...
  └── Episode 10: NEE @ 2025-01-02
      └── select planner → run tools → LLM predict → evaluate

  🌙 EVENING (once per day)
  │  └── LLM daily reflection: what worked? what failed?
  │      per-ticker tool accuracy, planner critique
  │
  🌑 NIGHT (once per day)
     ├── Memory consolidation (episodic → semantic → procedural)
     ├── Skill extraction (every 3 days)
     ├── Memory policy evolution (every 5 days if reward < 0.4)
     └── Planner evolution (if 3+ consecutive day failures)

═══════════════════════════════════════════════════════════
Day 2 (e.g., 2025-01-03)
═══════════════════════════════════════════════════════════
  ... repeat ...
```

### Week-Level Patterns

Over a typical week (5 trading days × 10 tickers = 50 episodes):

```
Week N
├── Mon: 10 episodes + morning + evening + night
├── Tue: 10 episodes + morning + evening + night
├── Wed: 10 episodes + morning + evening + night (memory consolidation)
├── Thu: 10 episodes + morning + evening + night (skill extraction)
└── Fri: 10 episodes + morning + evening + night (memory consolidation)
                                                   (memory policy evolution check)
```

- **Memory consolidation** runs every 2 nights → ~2-3 times per week
- **Skill extraction** runs every 3 days → ~1-2 times per week
- **Memory policy evolution** checked every 5 days → ~1 per week
- **Planner evolution** only on structural failure → 0-1 per week

### Month-Level / Phase-Level Structure

The experiment time span is divided into **three temporal phases**:

```
Dataset D2 (Oct 2024 – Mar 2025, ~125 trading days)

│◄────── TRAIN ──────►│◄── VAL ──►│◄──── TEST ────►│
│ Oct 2024 — Jan 17    │Jan 20—Feb 21│ Feb 24—Mar 28 │
│ ~75 trading days     │~25 days     │~25 days        │
│                      │             │                │
│ Agent learns:        │ Agent learns│ Agent FROZEN:  │
│ • Bandits update     │ (same as    │ • No bandit    │
│ • Memory writes      │  train)     │   updates      │
│ • Planners evolve    │             │ • No evolution  │
│ • Skills extracted   │             │ • No memory     │
│ • Credit assignment  │             │   writes        │
│                      │             │ • Pure eval     │
└──────────────────────┘─────────────┘────────────────┘
```

**Train (~60% of days)**: Full evolution — all bandits update, memory writes, reflection, nightly evolution.

**Val (~20%)**: Still learning — used later for hyperparameter tuning (e.g., "what's the best Shapley interval?"). Identical to train in the current codebase.

**Test (~20%)**: Agent is **frozen** via `loop.freeze()`:
- `select()` still picks configurations, but bandits don't update
- Tools execute, LLM predicts, but no credit assignment
- No memory writes, no evolution, no reflection
- This measures: **does what the agent learned generalize to unseen days?**

### Per-Dataset Temporal Coverage

```
Dataset  │ Date Range           │ Train Days │ Val Days │ Test Days │ Total Episodes
─────────┼──────────────────────┼────────────┼──────────┼───────────┼───────────────
D1       │ Jan 2 – Mar 28 2025 │ ~30        │ ~15      │ ~15       │ 620
D2       │ Oct 1 – Mar 28      │ ~75        │ ~25      │ ~25       │ 1,290
D3       │ Oct 1 – Mar 28      │ ~75        │ ~25      │ ~25       │ 1,290
D4       │ Jul 1 – Mar 28      │ ~110       │ ~30      │ ~55       │ 1,949
```

D4's longer train period (5 months) tests whether more experience → better test performance.

---

## 4. Evolution Across Timescales

```
Timescale        │ What Evolves                          │ Mechanism
─────────────────┼───────────────────────────────────────┼──────────────────────
Per-episode      │ Bandit posteriors (planner/tool/mem)  │ Thompson/LinUCB update
(~7s)            │ Episodic memory entries               │ Direct write
                 │                                       │
Per-day          │ Morning strategy overrides            │ LLM planning
(10 episodes)    │ Daily reflection insights             │ LLM analysis
                 │                                       │
Every 2 days     │ Semantic memory (patterns)            │ auto_distill()
                 │ Procedural memory (strategies)        │ promote_to_procedural()
                 │ Planner prompts                       │ inject procedural rules
                 │                                       │
Every 3 days     │ Skills (tools, templates, strategies) │ LLM extraction
                 │                                       │
Every 5 days     │ Memory policy registry                │ LLM evolution
                 │                                       │
On failure       │ Planner pool (new planners)           │ LLM code generation
(streak ≥ 3)     │                                       │
                 │                                       │
Every 20 ep      │ Shapley-calibrated credit             │ Counterfactual replay
```

### What Each Method Evolves

```
                  │ Episode │ Day    │ Multi-Day │ On-Failure │
Method            │ Bandits │ Reflect│ Consolidate│ Evolve    │
──────────────────┼─────────┼────────┼───────────┼───────────│
M1: stateless     │    ✗    │   ✗    │     ✗     │    ✗      │
M2: memory_only   │  mem ✓  │   ✗    │   mem ✓   │    ✗      │
M3: tool_only     │ tool ✓  │   ✗    │     ✗     │    ✗      │
M4: ael_full      │  all ✓  │   ✗    │   mem ✓   │    ✗      │
M5: credit_ablat  │  all ✓  │   ✗    │   mem ✓   │    ✗      │
M6: eael_full     │  all ✓  │   ✓    │   all ✓   │  plan+mem │
M7: no_reflect    │  all ✓  │   ✗    │   all ✓   │  plan+mem │
M8: no_coldstart  │  all ✓  │   ✓    │   all ✓   │  plan+mem │
M9: no_planevolve │  all ✓  │   ✓    │   all ✓   │    mem    │
M10: preset_tools │  all ✓  │   ✓    │   all ✓   │  plan+mem │
M11: no_skills    │  all ✓  │   ✓    │   mem ✓   │  plan+mem │
```

---

## 5. Metrics

### Per-Episode Metrics

| Metric | Formula | Range | Meaning |
|--------|---------|-------|---------|
| Directional Accuracy | `predicted_dir == actual_dir` | 0 or 1 | Binary: did we get the direction right? |
| MAPE | `|predicted_mag - actual_return| / |actual_return|` | [0, ∞) | How close was the magnitude? |
| Weighted Accuracy | `dir_correct × confidence` | [0, 1] | Confidence-weighted direction accuracy |
| Outcome Score | blend of DA + MAPE bonus | [-1, 1] | Single scalar for credit assignment |

### Aggregate Metrics (Per Horizon)

Reported separately for 1d, 7d, 14d:
- **Directional Accuracy**: fraction of correct direction predictions
- **MAPE**: mean magnitude error
- **AULC**: Area Under Learning Curve (cumulative DA over episodes)
- **Learning Gain**: last-quarter DA minus first-quarter DA

### Per-Phase Metrics

Metrics are computed separately for train, val, and test phases. **Paper reports test-phase metrics only** to measure generalization.

---

## 6. Experiment Matrix

### E1: Main Comparison

**Question**: Does evolving more components improve prediction accuracy?

```
              D1 (tech)  D2 (diverse)  D3 (small-cap)  D4 (regime)
stateless      ✓×3         ✓×3           ✓×3             ✓×3
memory_only    ✓×3         ✓×3           ✓×3             ✓×3
tool_only      ✓×3         ✓×3           ✓×3             ✓×3
ael_full       ✓×3         ✓×3           ✓×3             ✓×3
credit_ablat   ✓×3         ✓×3            —               —
eael_full      ✓×3         ✓×3           ✓×3             ✓×3
```

66 runs total. ×3 = seeds {42, 123, 456} for statistical significance.

### E2: EAEL Ablation

**Question**: Which EAEL component contributes most?

```
                          D2 (diverse)
eael_full                   ✓×3
eael_no_reflect             ✓×3    ← remove daily reflection
eael_no_coldstart           ✓×3    ← remove LLM cold-start priors
eael_no_plannerevolve       ✓×3    ← remove LLM planner generation
eael_preset_tools           ✓×3    ← replace per-tool with preset Thompson
eael_no_skills              ✓×3    ← remove skill extraction system
```

18 runs. All on D2 (most diverse dataset) for cost efficiency.

### Statistical Testing

- 3 seeds per cell → paired bootstrap p-values
- Holm-Bonferroni correction for multiple comparisons
- Report: mean ± std across seeds
