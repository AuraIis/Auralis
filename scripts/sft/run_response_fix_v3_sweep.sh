#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/workspace/v2data}"
cd "$ROOT_DIR"

BASE_CKPT="${BASE_CKPT:-checkpoints/sft_response_fix_de_v2_core_phase_a2/sft_smoke_step_220.pt}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/model/helix_v2_mid_500m_smart.yaml}"
TOKENIZER="${TOKENIZER:-tokenizer/helix_v2_tokenizer.model}"
TRAIN="${TRAIN:-data/training/sft_response_fix_de_v3/core_train.helix.jsonl}"
VAL="${VAL:-data/training/sft_response_fix_de_v3/val.helix.jsonl}"
PROBES="${PROBES:-eval/sft_response_fix_chat_gate_v2.yaml}"
RESULT_DIR="${RESULT_DIR:-data/eval/checkpoint_tests/sft_response_fix_de_v3_sweep}"
REPORT_DIR="${REPORT_DIR:-reports}"

mkdir -p "$RESULT_DIR" "$REPORT_DIR"

for steps in ${STEPS:-20 40 60}; do
  tag="sft_response_fix_de_v3_from_a2_sweep_${steps}"
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
    --lr 8e-7 \
    --warmup-steps 20 \
    --eval-every "$steps" \
    --category-weights facts_de=2,hallucination_guard=2,qa_de=2,honesty=2 \
    --eos-loss-weight 8 \
    --diag-json "$diag" \
    --save-final

  ckpt="$(ls -1t "${out_dir}"/sft_smoke_step_*.pt | head -n 1)"
  echo "== keyword gate ${tag}: ${ckpt} =="
  python scripts/eval/run_capability_probes.py \
    --model-config "$MODEL_CONFIG" \
    --checkpoint "$ckpt" \
    --tokenizer "$TOKENIZER" \
    --probes "$PROBES" \
    --results-dir "$RESULT_DIR" \
    --tag "${tag}_chat_gate_v2"

  echo "== semantic gate ${tag} =="
  python scripts/eval/semantic_response_gate.py \
    --input "${RESULT_DIR}/${tag}_chat_gate_v2.json" \
    --output-json "${REPORT_DIR}/semantic_gate_${tag}_2026-05-28.json" \
    --output-md "${REPORT_DIR}/semantic_gate_${tag}_2026-05-28.md"
done

python - <<'PY'
import json
from pathlib import Path

rows = []
for path in sorted(Path("reports").glob("semantic_gate_sft_response_fix_de_v3_from_a2_sweep_*_2026-05-28.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = data["summary"]
    rows.append((path.name, summary["semantic_score"], summary["passed"], summary["total"]))

print("== sweep semantic summary ==")
for name, score, passed, total in rows:
    print(f"{name}: {score:.3f} ({passed}/{total})")
PY
