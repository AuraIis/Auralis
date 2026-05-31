#!/usr/bin/env bash
set -euo pipefail

export VIRTUAL_ENV="${VIRTUAL_ENV:-/opt/auralis-venv}"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-all}"
export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}"

if [ -d /workspace/v2data/src ]; then
  export PYTHONPATH="/workspace/v2data/src:/workspace/v2data:${PYTHONPATH:-}"
else
  export PYTHONPATH="/opt/auralis/src:/opt/auralis:${PYTHONPATH:-}"
fi

exec "$@"
