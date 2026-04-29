# AEL: Agent Evolving Learning

Code and data for the paper *"AEL: Agent Evolving Learning for Open-Ended Environments"*.

## Overview

AEL is a two-timescale self-improving agent framework for financial portfolio allocation. It jointly evolves three components across trading episodes:

- **Planner selection** via LinUCB contextual bandit (slow timescale)
- **Tool selection** via Thompson Sampling (fast timescale)
- **Memory retrieval** with episodic/semantic/procedural tiers and bandit-based relevance

A core finding: the simplest variant (reflection + memory, no credit assignment, no per-tool selection, no skills) achieves the best performance (Sharpe 2.13), establishing a "less is more" result where every additional component degrades the system.

## Installation

```bash
pip install -e .
# or
pip install -r requirements.txt
```

**Dependencies:** numpy, pandas, pyyaml, yfinance, scipy, matplotlib, litellm, tenacity, boto3, requests

## Quick Start

```bash
# Smoke test with cheap GPT model (~$0.01, ~2 minutes)
python experiments/run_portfolio.py \
  --config experiments/configs/test_smoke_gpt.yaml \
  --seed 42

# Run a single stateless baseline on D-full
python experiments/run_portfolio.py \
  --config experiments/configs/stateless.yaml \
  --dataset experiments/configs/dataset_d_full.yaml \
  --seed 42

# Run full AEL
python experiments/run_portfolio.py \
  --config experiments/configs/incremental/step3_tool_memory_reflect.yaml \
  --dataset experiments/configs/dataset_d_full.yaml \
  --seed 42
```

## Reproducing Paper Results

```bash
# Run all 5-seed experiments (Table 3 + Table 4)
bash experiments/run_paper_5seed.sh

# Extract metrics table from backtest logs
python experiments/extract_table3.py

# Generate paper figures from pre-computed data
python figures/generate_paper_figures.py

# Statistical significance tests
python experiments/statistical_tests.py
```

Pre-computed results are in `results/paper_results.json`. Raw backtest logs from our experiments are in `logs/`.

## Baselines

Six baseline methods are implemented in `baselines/`:

| Method | File | Reference |
|--------|------|-----------|
| Reflexion | `reflexion.py` | Shinn et al., 2023 |
| ExpeL | `expel.py` | Zhao et al., 2024 |
| FactorMiner | `factorminer.py` | Li et al., 2026 |
| Meta-Reflexion | `meta_reflexion.py` | Qu et al., 2025 |
| EvoTool | `evotool.py` | Chen et al., 2026 |
| HyperAgent | `hyperagent.py` | Murthy et al., 2025 |

```bash
# Run all baselines on D-full with 5 seeds
python baselines/run_portfolio_baselines.py --dataset d_full --seeds 42,123,456,789,1024
```

See `baselines/README.md` for details.

## Configuration

Experiments are configured via YAML files in `experiments/configs/`:

- **Method configs**: `stateless.yaml`, `ael_full.yaml`, `eael_full.yaml`, etc.
- **Dataset configs**: `dataset_d_full.yaml` (paper), `dataset_d_small.yaml` (quick test)
- **Incremental build**: `incremental/step2_tool_memory.yaml` through `step5_eael_full.yaml`
- **Ablation configs**: `incremental/step3_*.yaml` (7 ablation variants from AEL)

Configs are composable: method config + dataset config are merged at runtime.

## LLM Backend

Uses [litellm](https://github.com/BerriAI/litellm) for unified LLM calling. Set your API key:

```bash
# For Anthropic models (used in paper)
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=us-east-1

# For OpenAI models (smoke test)
export OPENAI_API_KEY=...

# For local models via Ollama
# No API key needed, just run: ollama serve
```

The paper uses `bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0`. For quick testing, `gpt-4o-mini` works well.

## Data

Pre-cached market data (2.1 MB) in `data/`:
- `prices/`, `prices_1h/` : OHLCV price data
- `fundamentals/` : quarterly financial metrics
- `analyst/` : analyst ratings and target prices
- `earnings/` : earnings surprise data
- `options/` : implied volatility and options data
- `metadata/` : ticker information

To re-collect data: `python scripts/collect_data.py`

## Project Structure

```
AEL/
├── ael/                    # Core framework
│   ├── benchmarks/         # Backtesting and portfolio runner
│   ├── evolution/          # Bandits, credit assignment, reflection
│   ├── memory/             # Multi-tier memory store
│   ├── planner/            # Strategy pool (sequential, decompose, adaptive, ...)
│   ├── skills/             # Skill extraction and storage
│   └── tools/              # Finance tools (prices, fundamentals, technicals)
├── baselines/              # 6 baseline implementations
├── experiments/            # Experiment runners and configs
│   └── configs/            # YAML configuration files
├── data/                   # Pre-cached market data
├── logs/                   # Experiment backtest logs
├── results/                # Pre-computed paper results
├── figures/                # Figure generation scripts
├── tests/                  # Smoke tests
├── scripts/                # Data collection utilities
└── docs/                   # Architecture documentation
```

## License

MIT License. See `LICENSE` for details.
