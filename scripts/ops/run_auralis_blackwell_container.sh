#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-auralis-blackwell:cu13}"
CONTAINER_NAME="${CONTAINER_NAME:-auralis-blackwell}"
CACHE_DIR="${CACHE_DIR:-/mnt/cache/auralis-blackwell-cache}"
CODE_REPO_DIR="${CODE_REPO_DIR:-/mnt/user/Auralis/AuralisV2}"
DATA_REPO_DIR="${DATA_REPO_DIR:-/mnt/user/Auralis/NEWGPT/v2data}"
SHM_SIZE="${SHM_SIZE:-16g}"

mkdir -p "$CACHE_DIR/pip" "$CACHE_DIR/tmp"

if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  docker rm -f "$CONTAINER_NAME" >/dev/null
fi

docker run -d \
  --name "$CONTAINER_NAME" \
  --runtime=nvidia \
  --shm-size="$SHM_SIZE" \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e PIP_CACHE_DIR=/cache/pip \
  -e TMPDIR=/cache/tmp \
  -v "$CODE_REPO_DIR:/workspace/v2data" \
  -v "$DATA_REPO_DIR/checkpoints:/workspace/v2data/checkpoints" \
  -v "$DATA_REPO_DIR/data:/workspace/v2data/data" \
  -v "$DATA_REPO_DIR/tokenized:/workspace/v2data/tokenized" \
  -v "$DATA_REPO_DIR/tokenizer:/workspace/v2data/tokenizer:ro" \
  -v "$DATA_REPO_DIR/logs:/workspace/v2data/logs" \
  -v /mnt/user/Auralis/checkpoints:/checkpoints \
  -v "$CACHE_DIR:/cache" \
  "$IMAGE_NAME" \
  bash -lc "sleep infinity"

echo "Started $CONTAINER_NAME from $IMAGE_NAME"
echo "Enter with:"
echo "  docker exec -it $CONTAINER_NAME bash"
