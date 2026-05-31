#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-auralis-blackwell:cu13}"
CONTAINER_NAME="${CONTAINER_NAME:-auralis-blackwell}"
REPO_DIR="${REPO_DIR:-/mnt/user/Auralis/AuralisV2}"
BUILD_ROOT="${BUILD_ROOT:-/mnt/cache/auralis-blackwell-build}"
BUILD_CONTEXT="$BUILD_ROOT/context"
CACHE_DIR="${CACHE_DIR:-/mnt/cache/auralis-blackwell-cache}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-1}"
MAX_JOBS="${MAX_JOBS:-8}"

if [ ! -f "$REPO_DIR/pyproject.toml" ]; then
  echo "Repo not found at $REPO_DIR (missing pyproject.toml)" >&2
  exit 1
fi

mkdir -p "$BUILD_CONTEXT" "$CACHE_DIR/pip" "$CACHE_DIR/tmp"
rm -rf "$BUILD_CONTEXT"
mkdir -p "$BUILD_CONTEXT"

for item in pyproject.toml README.md src scripts configs eval tokenizer docker; do
  if [ -e "$REPO_DIR/$item" ]; then
    rsync -a --delete \
      --exclude "__pycache__" \
      --exclude "*.pyc" \
      "$REPO_DIR/$item" "$BUILD_CONTEXT/"
  fi
done

echo "Building $IMAGE_NAME from $BUILD_CONTEXT"
DOCKER_BUILDKIT=1 docker build \
  --build-arg INSTALL_FLASH_ATTN="$INSTALL_FLASH_ATTN" \
  --build-arg MAX_JOBS="$MAX_JOBS" \
  -f "$BUILD_CONTEXT/docker/blackwell/Dockerfile" \
  -t "$IMAGE_NAME" \
  "$BUILD_CONTEXT"

echo
echo "Image built: $IMAGE_NAME"
echo "Run a shell with:"
echo "  docker run --rm -it --name $CONTAINER_NAME --runtime=nvidia --shm-size=16g \\"
echo "    -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \\"
echo "    -v /mnt/user/Auralis/NEWGPT:/workspace \\"
echo "    -v /mnt/user/Auralis/checkpoints:/checkpoints \\"
echo "    -v $CACHE_DIR:/cache \\"
echo "    $IMAGE_NAME bash"
