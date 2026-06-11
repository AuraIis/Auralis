#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/workspace/v2data}"
cd "$ROOT_DIR"

BASE_CKPT="${BASE_CKPT:-checkpoints/sft_response_fix_de_v6_bridge_from_v5guardbal_40/sft_smoke_step_40.pt}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/model/helix_v2_mid_500m_smart.yaml}"
TOKENIZER="${TOKENIZER:-tokenizer/helix_v2_tokenizer.model}"
TRAIN="${TRAIN:-data/training/sft_response_fix_de_v7_bonn_photo_patch/core_train.helix.jsonl}"
VAL="${VAL:-data/training/sft_response_fix_de_v7_bonn_photo_patch/val.helix.jsonl}"
RESULT_ROOT="${RESULT_ROOT:-data/eval/checkpoint_tests}"
REPORT_DIR="${REPORT_DIR:-reports}"

mkdir -p "$RESULT_ROOT" "$REPORT_DIR"
python scripts/data/build_sft_response_fix_de_v7_bonn_photo_patch.py

for steps in ${STEPS:-15 30 45}; do
  tag="sft_response_fix_de_v7_bonn_photo_from_v6_40_${steps}"
  out_dir="checkpoints/${tag}"
  diag="${REPORT_DIR}/${tag}_2026-05-28.json"
  echo "== train ${tag} =="
  python scripts/sft/smoke_sft_de.py \
    --model-config "$MODEL_CONFIG" \
    --checkpoint "$BASE_CKPT" \
    --train "$TRAIN" \
    --val "$VAL" \
    --output-dir "$out_dir" \
    --steps "$steps" \
    --batch-size 1 \
    --grad-accum 4 \
    --max-length 512 \
    --train-limit 0 \
    --val-limit 0 \
    --lr "${LR:-3e-7}" \
    --warmup-steps 8 \
    --eval-every "$steps" \
    --category-weights "${CATEGORY_WEIGHTS:-hallucination_guard=1.8,qa_de=1.8,facts_de=0.9,honesty=1.2}" \
    --family-balanced-sampler \
    --eos-loss-weight 10 \
    --diag-json "$diag" \
    --save-final

  ckpt="$(ls -1t "${out_dir}"/sft_smoke_step_*.pt | head -n 1)"
  result_dir="${RESULT_ROOT}/${tag}"
  for gate in sft_response_fix_chat_gate_v2 sft_response_fix_chat_gate_v3_holdout sft_response_fix_chat_gate_v4_fresh_holdout sft_response_fix_chat_gate_v5_fresh_holdout; do
    probe="eval/${gate}.yaml"
    rtag="${tag}_${gate}"
    echo "== gate ${rtag} =="
    python scripts/eval/run_capability_probes.py \
      --model-config "$MODEL_CONFIG" \
      --checkpoint "$ckpt" \
      --tokenizer "$TOKENIZER" \
      --probes "$probe" \
      --results-dir "$result_dir" \
      --tag "$rtag"
    python scripts/eval/semantic_response_gate.py \
      --input "${result_dir}/${rtag}.json" \
      --output-json "${REPORT_DIR}/semantic_gate_${rtag}_2026-05-28.json" \
      --output-md "${REPORT_DIR}/semantic_gate_${rtag}_2026-05-28.md"
  done
done

python - <<'PY'
import json
from pathlib import Path

print("== v7 sweep summary ==")
for path in sorted(Path("reports").glob("semantic_gate_sft_response_fix_de_v7_bonn_photo_from_v6_40_*_2026-05-28.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    s = data["summary"]
    print(f"{path.name}: semantic={s['semantic_score']:.3f} ({s['passed']}/{s['total']})")
PY
