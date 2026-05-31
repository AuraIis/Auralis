#!/usr/bin/env bash
# Multi-GPU (data-parallel / DDP) pretraining launcher.
#
# The training code is single-process by default; DDP activates only when
# launched via torchrun (WORLD_SIZE>1). This script does that.
#
# Usage (on a RunPod multi-GPU pod):
#   NPROC=4 TRAIN_CONFIG=configs/training/pretrain_1b_bilingual_foundation_warmstart.yaml \
#     bash scripts/ops/run_pretrain_multigpu.sh --warm-start checkpoints/.../best.pt
#
# IMPORTANT — keep the GLOBAL batch constant when adding GPUs:
#   global_batch_tokens = batch_size_per_device * gradient_accumulation * NPROC * seq_length
#   A single-GPU config with gradient_accumulation=32 should use 32/NPROC
#   (16 for 2 GPUs, 8 for 4 GPUs). Otherwise the effective batch grows ~NPROC×
#   and convergence/LR behaviour changes. Throughput scales ~NPROC; wall-clock
#   drops ~NPROC (minus ~10-15% all-reduce overhead).
#
# Notes:
#   - Checkpoints are written DDP-agnostically (no "module." prefix) → they load
#     back fine on a single GPU (e.g. on BITBASTION).
#   - If gradient checkpointing + DDP errors ("marked ready twice" / unused
#     params), set use_reentrant=False in the checkpoint wrapper or pass
#     find_unused_parameters via the model; flagged in train_phase1.py.
set -euo pipefail
cd /workspace/v2data

NPROC="${NPROC:-2}"
TRAIN_CONFIG="${TRAIN_CONFIG:?set TRAIN_CONFIG=configs/training/<run>.yaml}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export AURALIS_USE_FLASH_ATTN="${AURALIS_USE_FLASH_ATTN:-1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

echo "== preflight (single-process dry-run) =="
python scripts/pretrain/train_phase1.py --config "$TRAIN_CONFIG" --dry-run --no-wandb

echo "== multi-GPU pretrain: torchrun --nproc_per_node=$NPROC | config=$TRAIN_CONFIG =="
torchrun --standalone --nproc_per_node="$NPROC" \
  scripts/pretrain/train_phase1.py \
  --config "$TRAIN_CONFIG" \
  "$@"
