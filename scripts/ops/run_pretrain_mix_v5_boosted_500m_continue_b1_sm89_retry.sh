#!/usr/bin/env bash
set -euo pipefail

cd /workspace/v2data
mkdir -p logs
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TRITON_OVERRIDE_ARCH="${TRITON_OVERRIDE_ARCH:-sm89}"

LOG="logs/pretrain_mix_v5_boosted_500m_continue_b1_sm89_retry.log"
echo "started pretrain_mix_v5_boosted_500m_continue_b1_sm89_retry $(date -Is)" > "$LOG"
echo "PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF" >> "$LOG"
echo "TRITON_OVERRIDE_ARCH=$TRITON_OVERRIDE_ARCH" >> "$LOG"

set +e
python scripts/pretrain/train_phase1.py \
  --config configs/training/pretrain_mix_v5_boosted_500m_continue_b1_sm89_retry.yaml \
  --init-weights checkpoints/pretrain_mix_v4_boosted_500m/best.pt \
  --no-wandb >> "$LOG" 2>&1
rc=$?
set -e

echo "EXIT_CODE=$rc" >> "$LOG"
exit "$rc"
