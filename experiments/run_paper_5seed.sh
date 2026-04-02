#!/usr/bin/env bash
# Run all paper experiments with 5 seeds (42, 123, 456, 789, 1024) on D-full.
#
# Total: ~95 runs
#   65 AEL variants (incremental build + credit study + ablation)
#   25 prior baselines (reflexion, expel, factorminer, meta_reflexion, evotool)
#    5 HyperAgent
#
# Usage:
#   bash experiments/run_paper_5seed.sh          # full run
#   bash experiments/run_paper_5seed.sh --parallel 4   # parallel workers
set -euo pipefail

cd "$(dirname "$0")/.."
# Activate your virtual environment here

PARALLEL="${1:---parallel}"
NWORKERS="${2:-3}"

echo "=== Paper 5-seed experiments on D-full ==="
echo "Workers: $NWORKERS"
echo ""

# 1. AEL variants: incremental build + credit study + ablation (65 runs)
echo "--- Phase 1: AEL variants (65 runs) ---"
python experiments/run_all.py --experiment paper_all --parallel "$NWORKERS" 2>&1 | tee logs/paper_ael_5seed.log

# 2. Prior baselines: portfolio mode (25 runs)
echo "--- Phase 2: Prior baselines (25 runs) ---"
python -m baselines.run_portfolio_baselines --dataset d_full \
    --seeds 42 123 456 789 1024 --parallel "$NWORKERS" 2>&1 | tee logs/paper_baselines_5seed.log

# 3. HyperAgent (5 runs)
echo "--- Phase 3: HyperAgent (5 runs) ---"
python -m baselines.run_hyperagent --dataset d_full \
    --seeds 42 123 456 789 1024 2>&1 | tee logs/paper_hyperagent_5seed.log

echo ""
echo "=== All paper 5-seed experiments complete ==="
echo "Run 'python figures/generate_paper_figures.py' to regenerate figures."
