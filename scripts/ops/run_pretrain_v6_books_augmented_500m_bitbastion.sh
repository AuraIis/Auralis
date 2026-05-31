#!/usr/bin/env bash
set -euo pipefail
cd /workspace/v2data

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export TRITON_OVERRIDE_ARCH="${TRITON_OVERRIDE_ARCH:-sm89}"
export AURALIS_USE_CUDA_KERNELS=1
export AURALIS_USE_MAMBA_KERNEL=1
export AURALIS_USE_GLA_KERNEL=1
export AURALIS_USE_FLASH_ATTN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

python -u scripts/pretrain/train_phase1.py \
  --config configs/training/pretrain_v6_books_augmented_500m_from_v5_best_bitbastion.yaml \
  --init-weights /workspace/v2data/checkpoints/pretrain_mix_v5_boosted_500m_a100/best.pt \
  --no-wandb \
  --no-compile
