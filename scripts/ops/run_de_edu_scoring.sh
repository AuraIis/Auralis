#!/usr/bin/env bash
# German educational-quality scoring sweep (FineWeb-Edu methodology).
#
# Annotates a sample of each German source 0-5 on educational value via a strong
# local LLM (Ollama, OpenAI-compatible endpoint), then prints the per-source
# score distribution. English fineweb_edu is scored too, as a calibration
# reference (it was already edu-filtered upstream, so it should skew high).
#
# Run inside the auralis-blackwell container. The Ollama host endpoint must be
# reachable from the container (host LAN IP, not localhost):
#
#   docker exec \
#     -e OPENAI_API_BASE=http://192.168.178.5:11434/v1 \
#     -e OPENAI_API_KEY=ollama \
#     -e OPENAI_MODEL=gemma4:31b \
#     auralis-blackwell bash -lc 'cd /workspace/v2data && bash scripts/ops/run_de_edu_scoring.sh'
set -euo pipefail
cd /workspace/v2data

: "${OPENAI_API_BASE:=http://192.168.178.5:11434/v1}"
: "${OPENAI_API_KEY:=ollama}"
: "${OPENAI_MODEL:=gemma4:31b}"
export OPENAI_API_BASE OPENAI_API_KEY OPENAI_MODEL

SAMPLE="${SAMPLE:-400}"
CONC="${CONC:-10}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
REASONING="${REASONING:-low}"   # empty string = provider default thinking
SEED="${SEED:-20260530}"        # change to draw a fresh, non-overlapping sample
SCAN_LINES="${SCAN_LINES:-200000}"
SOURCES="${SOURCES:-fineweb2_de german_commons wikipedia_de fineweb_edu}"
OUT="${OUT:-eval/results/de_edu}"
mkdir -p "$OUT"

echo "endpoint: $OPENAI_API_BASE | model: $OPENAI_MODEL | sample: $SAMPLE | conc: $CONC | reasoning: '${REASONING}' | seed: $SEED"
for s in $SOURCES; do
  echo "=========================================================="
  echo "== scoring $s =="
  python scripts/data/score_german_edu.py \
    --input "cleaned/$s.filtered.txt" \
    --source "$s" \
    --sample "$SAMPLE" \
    --concurrency "$CONC" \
    --max-tokens "$MAX_TOKENS" \
    --reasoning-effort "$REASONING" \
    --scan-lines "$SCAN_LINES" \
    --seed "$SEED" \
    --output-jsonl "$OUT/$s.jsonl" || echo "WARN: scoring $s failed (continuing)"
done
echo "ALL DONE"
