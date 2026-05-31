#!/usr/bin/env bash
set -euo pipefail

cd /workspace/v2data

PREFLIGHT_CONFIG="${PREFLIGHT_CONFIG:-configs/eval/auralis_1b_readiness_preflight.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/training/pretrain_1b_canary_readiness.yaml}"
REPORT_DIR="${REPORT_DIR:-reports}"
DATE_TAG="${DATE_TAG:-$(date -u +%Y-%m-%d)}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export TRITON_OVERRIDE_ARCH="${TRITON_OVERRIDE_ARCH:-sm89}"
export AURALIS_USE_CUDA_KERNELS=1
export AURALIS_USE_MAMBA_KERNEL=1
export AURALIS_USE_GLA_KERNEL=1
export AURALIS_USE_FLASH_ATTN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "== 1B readiness preflight =="
python scripts/eval/one_b_readiness_preflight.py \
  --config "$PREFLIGHT_CONFIG" \
  --output-json "${REPORT_DIR}/auralis_1b_readiness_preflight_${DATE_TAG}.json" \
  --output-md "${REPORT_DIR}/auralis_1b_readiness_preflight_${DATE_TAG}.md"

echo "== training dry-run preflight =="
python scripts/pretrain/train_phase1.py \
  --config "$TRAIN_CONFIG" \
  --dry-run \
  --no-wandb \
  --no-compile

echo "== start guarded 1B canary =="
python -u scripts/pretrain/train_phase1.py \
  --config "$TRAIN_CONFIG" \
  --no-wandb \
  --no-compile

echo "== post-run readiness gates =="
CKPT="${CKPT:-checkpoints/pretrain_1b_canary_readiness/best.pt}"
RESULT_DIR="${RESULT_DIR:-data/eval/checkpoint_tests/auralis_1b_canary_readiness}"
mkdir -p "$RESULT_DIR"

python scripts/eval/run_capability_probes.py \
  --model-config configs/model/helix_v2_1b.yaml \
  --checkpoint "$CKPT" \
  --probes eval/auralis_1b_readiness_gate_v1.yaml \
  --results-dir "$RESULT_DIR" \
  --tag "auralis_1b_readiness_gate_${DATE_TAG}" \
  --max-new-tokens 96

python scripts/eval/run_capability_probes.py \
  --model-config configs/model/helix_v2_1b.yaml \
  --checkpoint "$CKPT" \
  --probes eval/sft_response_frozen_target_retention_v2.yaml \
  --results-dir "$RESULT_DIR" \
  --tag "sft_response_frozen_target_retention_v2_${DATE_TAG}" \
  --max-new-tokens 96

python scripts/eval/frozen_response_gate.py \
  --probes eval/sft_response_frozen_target_retention_v2.yaml \
  --input "${RESULT_DIR}/sft_response_frozen_target_retention_v2_${DATE_TAG}.json" \
  --output-json "${REPORT_DIR}/frozen_response_gate_sft_response_v2_${DATE_TAG}.json" \
  --output-md "${REPORT_DIR}/frozen_response_gate_sft_response_v2_${DATE_TAG}.md"
