# Paper Results

## Pre-computed Results

`paper_results.json` contains all metrics reported in Table 3 and Table 4 of the paper, computed from 5-seed evaluations on D-full (commit `13ba204`).

## Reproducing Results

To reproduce the paper results from scratch:

```bash
# 1. Run all 5-seed experiments (requires LLM API key)
bash experiments/run_paper_5seed.sh

# 2. Extract metrics table
python experiments/extract_table3.py

# 3. Generate paper figures
python figures/generate_paper_figures.py
```

## Backtest Logs

The `logs/` directory contains the raw backtest JSON files from our experiments. Each file includes:
- `metrics`: aggregate portfolio metrics (Sharpe, Sortino, Calmar, etc.)
- `phase_metrics`: train/val/test phase breakdowns
- `score_history`: per-bar returns and portfolio states
- `config`: experiment configuration used

To verify paper values match the logs:
```bash
python experiments/extract_table3.py 13ba204
```
