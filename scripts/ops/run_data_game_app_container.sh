#!/usr/bin/env bash
set -euo pipefail

NAME="${NAME:-auralis-data-game}"
IMAGE="${IMAGE:-auralis-blackwell:cu13}"
PORT="${PORT:-8777}"
REPO_HOST="${REPO_HOST:-/mnt/user/Auralis/AuralisV2}"
SOURCE="${SOURCE:-/workspace/v2data/data/training/curated_40b/german.txt}"
OUTPUT="${OUTPUT:-/workspace/v2data/data/human_feedback/auralis_data_game_v2.jsonl}"
SCAN_LINES="${SCAN_LINES:-30000}"
QUEUE_SIZE="${QUEUE_SIZE:-400}"
# Model-in-the-loop: review the edu classifier's borderline calls (built by
# score_corpus_edu.py --review-pool). If the file is missing the app falls back
# to the raw --source scan.
POOL="${POOL:-/workspace/v2data/eval/results/de_edu/review_pool.jsonl}"
BOUNDARY="${BOUNDARY:-2.0}"

docker rm -f "$NAME" >/dev/null 2>&1 || true

docker run -d \
  --name "$NAME" \
  --restart unless-stopped \
  -p "${PORT}:8777" \
  -v "${REPO_HOST}:/workspace/v2data" \
  "$IMAGE" \
  bash -lc "cd /workspace/v2data && python scripts/monitor/data_game_app.py --host 0.0.0.0 --port 8777 --root /workspace/v2data --source '${SOURCE}' --output '${OUTPUT}' --scan-lines '${SCAN_LINES}' --queue-size '${QUEUE_SIZE}' --pool '${POOL}' --boundary '${BOUNDARY}'"

echo "Auralis data game container started: ${NAME}"
echo "URL: http://BITBASTION:${PORT}"
echo "Output: ${OUTPUT}"
