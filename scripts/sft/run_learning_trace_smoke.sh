#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/workspace/v2data}"
cd "$ROOT_DIR"

TAG="${TAG:-learning_trace_de_core_smoke}"
BASE_CKPT="${BASE_CKPT:-checkpoints/sft_response_fix_de_v8_stable_from_v6_40_20/sft_smoke_step_20.pt}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/model/helix_v2_mid_500m_smart.yaml}"
TOKENIZER="${TOKENIZER:-tokenizer/helix_v2_tokenizer.model}"
TRAIN="${TRAIN:-data/training/sft_response_fix_de_v8_stable_mix/core_train.helix.jsonl}"
VAL="${VAL:-data/training/sft_response_fix_de_v8_stable_mix/val.helix.jsonl}"
PROBES="${PROBES:-eval/learning_trace_de_core.yaml}"
OUT_DIR="${OUT_DIR:-checkpoints/${TAG}}"
REPORT_DIR="${REPORT_DIR:-reports/learning_trace}"

mkdir -p "$OUT_DIR" "$REPORT_DIR"

python scripts/sft/smoke_sft_de.py \
  --model-config "$MODEL_CONFIG" \
  --checkpoint "$BASE_CKPT" \
  --tokenizer "$TOKENIZER" \
  --train "$TRAIN" \
  --val "$VAL" \
  --output-dir "$OUT_DIR" \
  --steps "${STEPS:-8}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --grad-accum "${GRAD_ACCUM:-4}" \
  --max-length "${MAX_LENGTH:-512}" \
  --train-limit "${TRAIN_LIMIT:-0}" \
  --val-limit "${VAL_LIMIT:-0}" \
  --lr "${LR:-5e-8}" \
  --warmup-steps "${WARMUP_STEPS:-4}" \
  --eval-every "${EVAL_EVERY:-4}" \
  --category-weights "${CATEGORY_WEIGHTS:-facts_de=1.3,hallucination_guard=1.3,qa_de=1.4,honesty=1.4,instruction_de=0.9}" \
  --family-balanced-sampler \
  --eos-loss-weight "${EOS_LOSS_WEIGHT:-8}" \
  --learning-probes "$PROBES" \
  --learning-trace-json "${REPORT_DIR}/${TAG}.json" \
  --learning-trace-html "${REPORT_DIR}/${TAG}.html" \
  --learning-neuro-html "${REPORT_DIR}/${TAG}_neuro.html" \
  --learning-html-auto-refresh "${LEARNING_HTML_AUTO_REFRESH:-10}" \
  --learning-trace-every "${LEARNING_TRACE_EVERY:-4}" \
  --diag-json "${REPORT_DIR}/${TAG}_diag.json" \
  --save-final

echo "learning trace JSON: ${REPORT_DIR}/${TAG}.json"
echo "learning trace HTML: ${REPORT_DIR}/${TAG}.html"
echo "learning neuro HTML: ${REPORT_DIR}/${TAG}_neuro.html"
