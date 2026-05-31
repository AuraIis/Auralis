#!/usr/bin/env bash
set -euo pipefail

cd /workspace/v2data
mkdir -p logs
export PYTHONUNBUFFERED=1

LOG="logs/pretrain_mix_v4_boosted_500m.log"
echo "started pretrain_mix_v4_boosted_500m $(date -Is)" > "$LOG"

set +e
python scripts/pretrain/train_phase1.py \
  --config configs/training/pretrain_mix_v4_boosted_500m.yaml \
  --no-wandb >> "$LOG" 2>&1
rc=$?
set -e

echo "EXIT_CODE=$rc" >> "$LOG"
exit "$rc"
