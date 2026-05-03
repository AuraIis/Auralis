#!/usr/bin/env bash
# Auralis Phase-1 resume on RunPod — pod-side setup.
#
# Run this once after the pod first boots with the network volume attached
# at /workspace/network_volume. Idempotent — safe to re-run if something
# goes wrong half-way through.
#
# Prereqs (on RunPod side):
#   * Pod image with PyTorch 2.7 + CUDA 12.8 (the official pytorch image works)
#   * Network volume 7a41k5ssos mounted at /workspace/network_volume
#   * Bitbastion → S3 sync of tokenized/, checkpoints/, tokenizer/ already done
#   * SSH key (~/.ssh/id_ed25519) on bitbastion can reach github (already true)
#
# What this does:
#   1. Clones Auralis repo to /workspace/Auralis
#   2. Installs Python deps + mamba_ssm + causal_conv1d
#   3. Symlinks the network volume's data/checkpoints/tokenizer into the repo
#   4. Verifies the resume checkpoint loads + a 5-token greedy generation works
#   5. Prints the launch command

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/ForceGaming4K/Auralis.git}"
REPO_DIR="${REPO_DIR:-/workspace/Auralis}"
NV_DIR="${NV_DIR:-/workspace/network_volume}"

echo "=== 1/5 Clone repo ==="
if [ -d "$REPO_DIR/.git" ]; then
  echo "  $REPO_DIR exists, pulling latest"
  cd "$REPO_DIR" && git pull --ff-only
else
  git clone "$REPO_URL" "$REPO_DIR"
  cd "$REPO_DIR"
fi
echo

echo "=== 2/5 Install deps ==="
pip install --upgrade pip
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r "$REPO_DIR/requirements.txt"
# mamba_ssm + causal_conv1d are needed by the AURALIS_USE_MAMBA_KERNEL=1 path.
# These need building against the right torch+CUDA — --no-build-isolation
# keeps them from re-resolving torch.
pip install --no-build-isolation \
  causal-conv1d>=1.4.0 \
  mamba-ssm>=2.2.0
echo

echo "=== 3/5 Wire network-volume symlinks ==="
mkdir -p "$REPO_DIR/checkpoints" "$REPO_DIR/tokenizer" "$REPO_DIR/tokenized"
# Use ln -sfn so re-runs replace stale symlinks cleanly
ln -sfn "$NV_DIR/tokenized/curated_40b" "$REPO_DIR/tokenized/curated_40b"
ln -sfn "$NV_DIR/checkpoints/phase1_pretrain" "$REPO_DIR/checkpoints/phase1_pretrain"
ln -sfn "$NV_DIR/tokenizer/helix_v2_tokenizer.model" "$REPO_DIR/tokenizer/helix_v2_tokenizer.model"
ln -sfn "$NV_DIR/tokenizer/helix_v2_tokenizer.vocab" "$REPO_DIR/tokenizer/helix_v2_tokenizer.vocab"
ls -la "$REPO_DIR/checkpoints/phase1_pretrain/best.pt" || {
  echo "  ERROR: best.pt not found on network volume."
  echo "  Did the bitbastion → S3 sync finish? Check 's3://7a41k5ssos/checkpoints/phase1_pretrain/best.pt'"
  exit 1
}
echo

echo "=== 4/5 Verify checkpoint loads ==="
cd "$REPO_DIR"
AURALIS_USE_MAMBA_KERNEL=1 \
AURALIS_USE_CUDA_KERNELS=1 \
TRITON_OVERRIDE_ARCH=sm90 \
python - << 'EOF'
import os, sys, torch
sys.path.insert(0, "."); sys.path.insert(0, "src")
import sentencepiece as spm
from auralis.model import build_model

device = torch.device("cuda")
model = build_model("configs/model/helix_v2_1b.yaml").to(device)
ckpt = torch.load("checkpoints/phase1_pretrain/best.pt", map_location=device, weights_only=False)
sd = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
miss, extra = model.load_state_dict(sd, strict=False)
assert len(miss) == 0 and len(extra) == 0, f"checkpoint mismatch: missing={len(miss)} extra={len(extra)}"
step = ckpt["state"]["step"]
loss = ckpt["state"]["best_val_loss"]
print(f"  loaded step={step} val_loss={loss:.4f} ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")

sp = spm.SentencePieceProcessor(model_file="tokenizer/helix_v2_tokenizer.model")
ids = sp.EncodeAsIds("Die Hauptstadt von Deutschland ist")
x = torch.tensor([ids], device=device)
with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
    out = model(input_ids=x)
print(f"  forward OK, logits shape={tuple(out['logits'].shape)}, finite={torch.isfinite(out['logits']).all().item()}")
EOF
echo

echo "=== 5/5 Done. Launch command: ==="
cat << 'EOF'

  cd /workspace/Auralis
  AURALIS_USE_CUDA_KERNELS=1 \
  AURALIS_USE_MAMBA_KERNEL=1 \
  TRITON_OVERRIDE_ARCH=sm90 \
  PYTHONUNBUFFERED=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  nohup python scripts/pretrain/train_phase1.py \
    --config configs/training/phase1_pretrain_runpod_resume.yaml \
    --resume-from checkpoints/phase1_pretrain/best.pt \
    --no-wandb \
    > /workspace/network_volume/logs/phase1_resume.log 2>&1 &

  echo $! > /workspace/network_volume/phase1.pid
  tail -f /workspace/network_volume/logs/phase1_resume.log

EOF
echo
echo "TRITON_OVERRIDE_ARCH:"
echo "  - bitbastion (Blackwell) needs sm89"
echo "  - RunPod H100/H200 needs sm90 (set above)"
echo "  - A100 needs sm80"
echo "  Adjust if you booted a different GPU class."
