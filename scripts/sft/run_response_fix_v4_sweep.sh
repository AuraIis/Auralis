#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/workspace/v2data}"
cd "$ROOT_DIR"

BASE_CKPT="${BASE_CKPT:-checkpoints/sft_response_fix_de_v2_core_phase_a2/sft_smoke_step_220.pt}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/model/helix_v2_mid_500m_smart.yaml}"
TOKENIZER="${TOKENIZER:-tokenizer/helix_v2_tokenizer.model}"
TRAIN="${TRAIN:-data/training/sft_response_fix_de_v4/core_train.helix.jsonl}"
VAL="${VAL:-data/training/sft_response_fix_de_v4/val.helix.jsonl}"
PROBES="${PROBES:-eval/sft_response_fix_chat_gate_v2.yaml}"
RESULT_DIR="${RESULT_DIR:-data/eval/checkpoint_tests/sft_response_fix_de_v4_sweep}"
REPORT_DIR="${REPORT_DIR:-reports}"

mkdir -p "$RESULT_DIR" "$REPORT_DIR"

for steps in ${STEPS:-40 80 120}; do
  tag="sft_response_fix_de_v4_from_a2_sweep_${steps}"
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
    --lr 3e-7 \
    --warmup-steps 30 \
    --eval-every "$steps" \
    --category-weights facts_de=1.4,hallucination_guard=1.0,qa_de=2.0,honesty=2.0,instruction_de=1.5 \
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

print("== v4 sweep summary ==")
for path in sorted(Path("reports").glob("semantic_gate_sft_response_fix_de_v4_from_a2_sweep_*_2026-05-28.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    s = data["summary"]
    tag = path.name.replace("semantic_gate_", "").replace("_2026-05-28.json", "")
    gate_path = Path("data/eval/checkpoint_tests/sft_response_fix_de_v4_sweep") / f"{tag}_chat_gate_v2.json"
    keyword = None
    if gate_path.exists():
        keyword = json.loads(gate_path.read_text(encoding="utf-8")).get("aggregate_score")
    keyword_s = "n/a" if keyword is None else f"{keyword:.3f}"
    print(f"{tag}: keyword={keyword_s} semantic={s['semantic_score']:.3f} ({s['passed']}/{s['total']})")
PY
