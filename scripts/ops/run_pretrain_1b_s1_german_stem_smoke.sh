#!/usr/bin/env bash
set -euo pipefail

cd /workspace/v2data

TRAIN_CONFIG="${TRAIN_CONFIG:-configs/training/pretrain_1b_s1_german_stem_smoke.yaml}"
DATE_TAG="${DATE_TAG:-$(date -u +%Y-%m-%d)}"
RUN_NAME="${RUN_NAME:-pretrain_1b_s1_german_stem_smoke_${DATE_TAG}}"
LOG_DIR="${LOG_DIR:-logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_NAME}.log}"
PID_FILE="${PID_FILE:-${LOG_DIR}/${RUN_NAME}.pid}"
RESULT_DIR="${RESULT_DIR:-data/eval/checkpoint_tests/pretrain_1b_s1_german_stem_smoke}"
CKPT="${CKPT:-checkpoints/pretrain_1b_s1_german_stem_smoke/best.pt}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export TRITON_OVERRIDE_ARCH="${TRITON_OVERRIDE_ARCH:-sm89}"
export AURALIS_USE_CUDA_KERNELS=1
export AURALIS_USE_MAMBA_KERNEL=1
export AURALIS_USE_GLA_KERNEL=1
export AURALIS_USE_FLASH_ATTN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

mkdir -p "$LOG_DIR" "$RESULT_DIR" reports

echo "== S1 German stem dry-run =="
python scripts/pretrain/train_phase1.py \
  --config "$TRAIN_CONFIG" \
  --dry-run \
  --no-wandb \
  --no-compile

echo "== start S1 German stem smoke =="
echo $$ > "$PID_FILE"
python -u scripts/pretrain/train_phase1.py \
  --config "$TRAIN_CONFIG" \
  --no-wandb \
  --no-compile 2>&1 | tee "$LOG_FILE"

echo "== post-run language samples =="
python scripts/eval/run_capability_probes.py \
  --model-config configs/model/helix_v2_1b.yaml \
  --checkpoint "$CKPT" \
  --probes eval/auralis_1b_readiness_gate_v1.yaml \
  --results-dir "$RESULT_DIR" \
  --tag "auralis_1b_readiness_gate_${DATE_TAG}" \
  --max-new-tokens 96

echo "== S1 smoke complete =="
