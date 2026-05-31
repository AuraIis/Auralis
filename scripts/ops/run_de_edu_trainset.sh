#!/usr/bin/env bash
# Large training-label annotation for the German edu classifier.
#
# Per-source sample counts are boundary-weighted: more fineweb2_de / german_commons
# (where the keep/drop decision actually lives) and less saturated-high wikipedia_de.
# Same judge (Gemini) + config as the proof run, but a fresh seed and deeper scan so
# the docs don't overlap the proof set in eval/results/de_edu/train/.
#
# Combine with the proof labels at train time:
#   --labels eval/results/de_edu/train/*.jsonl eval/results/de_edu/train2/*.jsonl
set -uo pipefail
cd /workspace/v2data

: "${OPENAI_API_BASE:=https://generativelanguage.googleapis.com/v1beta/openai}"
: "${OPENAI_API_KEY:?set OPENAI_API_KEY}"
: "${OPENAI_MODEL:=gemini-3.5-flash}"
export OPENAI_API_BASE OPENAI_API_KEY OPENAI_MODEL

CONC="${CONC:-20}"
REASONING="${REASONING:-}"   # empty = no reasoning_effort (non-thinking models like qwen3-2507); set 'low' for thinking models
SEED="${SEED:-20260601}"
SCAN_LINES="${SCAN_LINES:-1000000}"
OUT="${OUT:-eval/results/de_edu/train2}"
PAIRS="${PAIRS:-fineweb2_de:4500 german_commons:4500 wikipedia_de:1500}"
mkdir -p "$OUT"

echo "trainset annotation | model $OPENAI_MODEL | conc $CONC | seed $SEED | scan $SCAN_LINES | out $OUT"
echo "pairs: $PAIRS"
for pair in $PAIRS; do
  s="${pair%%:*}"; n="${pair##*:}"
  echo "=========================================================="
  echo "== $s (n=$n) =="
  python scripts/data/score_german_edu.py \
    --input "cleaned/$s.filtered.txt" --source "$s" --sample "$n" \
    --concurrency "$CONC" --reasoning-effort "$REASONING" \
    --scan-lines "$SCAN_LINES" --seed "$SEED" \
    --output-jsonl "$OUT/$s.jsonl" || echo "WARN: $s failed (continuing)"
done
echo "TRAINSET_DONE"
