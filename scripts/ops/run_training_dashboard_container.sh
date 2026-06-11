#!/usr/bin/env bash
set -euo pipefail

NAME="${NAME:-auralis-dashboard}"
IMAGE="${IMAGE:-auralis-blackwell:cu13}"
PORT="${PORT:-8765}"
REPO_HOST="${REPO_HOST:-/mnt/user/Auralis/AuralisV2}"
V2DATA_HOST="${V2DATA_HOST:-/mnt/user/Auralis/NEWGPT/v2data}"
CACHE_HOST="${CACHE_HOST:-/mnt/cache/auralis-blackwell-cache}"
CHECKPOINTS_HOST="${CHECKPOINTS_HOST:-/mnt/user/Auralis/checkpoints}"

docker rm -f "$NAME" >/dev/null 2>&1 || true

docker run -d \
  --name "$NAME" \
  --restart unless-stopped \
  -p "${PORT}:8765" \
  -v "${REPO_HOST}:/workspace/v2data" \
  -v "${V2DATA_HOST}/logs:/workspace/v2data/logs" \
  -v "${V2DATA_HOST}/checkpoints:/workspace/v2data/checkpoints" \
  -v "${V2DATA_HOST}/data:/workspace/v2data/data" \
  -v "${V2DATA_HOST}/tokenized:/workspace/v2data/tokenized" \
  -v "${V2DATA_HOST}/tokenizer:/workspace/v2data/tokenizer:ro" \
  -v "${CHECKPOINTS_HOST}:/checkpoints" \
  -v "${CACHE_HOST}:/cache" \
  "$IMAGE" \
  bash -lc "cd /workspace/v2data && python scripts/monitor/training_dashboard.py --host 0.0.0.0 --port 8765 --root /workspace/v2data --log-dir /workspace/v2data/logs"

echo "Auralis dashboard container started: ${NAME}"
echo "URL: http://BITBASTION:${PORT}"
