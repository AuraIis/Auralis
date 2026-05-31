#!/usr/bin/env bash
set -euo pipefail

cd /workspace/v2data

TRAIN_CONFIG="${TRAIN_CONFIG:-configs/training/pretrain_1b_bilingual_de55_en45_foundation_warmstart.yaml}"
WARM_START_CKPT="${WARM_START_CKPT:-checkpoints/pretrain_1b_bilingual_de55_en45_ramp/best.pt}"
DATE_TAG="${DATE_TAG:-$(date -u +%Y-%m-%d)}"
RUN_NAME="${RUN_NAME:-pretrain_1b_bilingual_foundation_warmstart_${DATE_TAG}}"
LOG_DIR="${LOG_DIR:-logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_NAME}.log}"
PID_FILE="${PID_FILE:-${LOG_DIR}/${RUN_NAME}.pid}"
RESULT_DIR="${RESULT_DIR:-data/eval/checkpoint_tests/pretrain_1b_bilingual_foundation_warmstart}"
CKPT="${CKPT:-checkpoints/pretrain_1b_bilingual_de55_en45_foundation_warmstart/best.pt}"

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

echo "== 1B bilingual foundation warm-start dry-run =="
python scripts/pretrain/train_phase1.py \
  --config "$TRAIN_CONFIG" \
  --dry-run \
  --no-wandb \
  --no-compile

echo "== start 1B bilingual foundation warm-start =="
echo $$ > "$PID_FILE"
echo "config: $TRAIN_CONFIG"
echo "warm_start: $WARM_START_CKPT"
python -u scripts/pretrain/train_phase1.py \
  --config "$TRAIN_CONFIG" \
  --warm-start "$WARM_START_CKPT" \
  --no-wandb \
  --no-compile 2>&1 | tee "$LOG_FILE"

if [[ -f "$CKPT" ]]; then
  echo "== post-run capability probes =="
  python scripts/eval/run_capability_probes.py \
    --model-config configs/model/helix_v2_1b.yaml \
    --checkpoint "$CKPT" \
    --results-dir "$RESULT_DIR" \
    --tag "foundation_warmstart_capability_${DATE_TAG}" \
    --device cuda \
    --max-new-tokens 64

  python scripts/eval/diagnose_checkpoint_generation.py \
    --model-config configs/model/helix_v2_1b.yaml \
    --checkpoint "$CKPT" \
    --tokenizer tokenizer/helix_v2_tokenizer.model \
    --output "${RESULT_DIR}/foundation_warmstart_diag_${DATE_TAG}.json" \
    --device cuda \
    --max-new-tokens 64 \
    --top-k 12
else
  echo "warn: expected checkpoint not found after run: $CKPT"
fi

echo "== foundation warm-start complete =="
