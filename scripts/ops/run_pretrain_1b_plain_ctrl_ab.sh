#!/usr/bin/env bash
# Architecture A/B control run: all-plain-attention model, same training config
# as the de55/en45 ramp. Fresh start (no warm-start). Compare its curves to the
# Helix ramp at matching steps.
set -euo pipefail

cd /workspace/v2data

TRAIN_CONFIG="${TRAIN_CONFIG:-configs/training/pretrain_1b_plain_ctrl_ab.yaml}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export AURALIS_USE_FLASH_ATTN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "== plain-attention control: training dry-run =="
python scripts/pretrain/train_phase1.py \
  --config "$TRAIN_CONFIG" \
  --dry-run \
  --no-wandb \
  --no-compile

echo "== plain-attention control: start fresh run =="
python -u scripts/pretrain/train_phase1.py \
  --config "$TRAIN_CONFIG" \
  --no-wandb \
  --no-compile
