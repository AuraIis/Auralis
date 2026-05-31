#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/workspace/v2data}"
cd "$ROOT_DIR"

BASE_CKPT="${BASE_CKPT:-checkpoints/sft_response_fix_de_v8_stable_from_v6_40_20/sft_smoke_step_20.pt}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/model/helix_v2_mid_500m_smart.yaml}"
TOKENIZER="${TOKENIZER:-tokenizer/helix_v2_tokenizer.model}"
TRAIN="${TRAIN:-data/training/sft_response_fix_de_v9_stable_reinforce/core_train.helix.jsonl}"
VAL="${VAL:-data/training/sft_response_fix_de_v9_stable_reinforce/val.helix.jsonl}"
RESULT_ROOT="${RESULT_ROOT:-data/eval/checkpoint_tests}"
REPORT_DIR="${REPORT_DIR:-reports}"

mkdir -p "$RESULT_ROOT" "$REPORT_DIR"
python scripts/data/build_sft_response_fix_de_v9_stable_reinforce.py

for steps in ${STEPS:-8 16 28}; do
  tag="sft_response_fix_de_v9_stable_from_v8_20_${steps}"
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
    --lr "${LR:-7e-8}" \
    --warmup-steps 4 \
    --eval-every "$steps" \
    --category-weights "${CATEGORY_WEIGHTS:-facts_de=1.3,hallucination_guard=1.2,qa_de=1.6,honesty=1.5,instruction_de=0.8}" \
    --family-balanced-sampler \
    --eos-loss-weight 8 \
    --diag-json "$diag" \
    --save-final

  ckpt="$(ls -1t "${out_dir}"/sft_smoke_step_*.pt | head -n 1)"
  result_dir="${RESULT_ROOT}/${tag}"
  for gate in sft_response_fix_chat_gate_v2 sft_response_fix_chat_gate_v3_holdout sft_response_fix_chat_gate_v4_fresh_holdout sft_response_fix_chat_gate_v5_fresh_holdout sft_response_fix_chat_gate_v6_fresh_holdout; do
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

print("== v9 stable sweep summary ==")
for path in sorted(Path("reports").glob("semantic_gate_sft_response_fix_de_v9_stable_from_v8_20_*_2026-05-28.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    s = data["summary"]
    print(f"{path.name}: semantic={s['semantic_score']:.3f} ({s['passed']}/{s['total']})")
PY
