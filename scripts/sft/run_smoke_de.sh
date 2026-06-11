#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

python scripts/sft/smoke_sft_de.py \
  --checkpoint checkpoints/phase1_pretrain/best.pt \
  --train data/training/sft_rescued/balanced/de_strict/train.helix.jsonl \
  --val data/training/sft_rescued/balanced/de_strict/val.helix.jsonl \
  --output-dir checkpoints/sft_smoke_de \
  --steps 50 \
  --batch-size 1 \
  --grad-accum 8 \
  --max-length 1536 \
  --train-limit 512 \
  --val-limit 64 \
  --lr 2.0e-5 \
  --eval-every 10 \
  --save-final
