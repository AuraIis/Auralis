# 250M test run on Unraid (RTX Pro 5000 Blackwell)

This run validates the full pipeline on a workstation GPU before
a multi-hour RunPod run is booked. 250 M model, `bf16`,
heterogeneous stack (Mamba + GLA + Sparse).

**Expectation (Blackwell 48 GB, pure-python kernel):**

| Metric | 3090 measured | Pro 5000 Blackwell expected |
|---|--:|--:|
| Peak VRAM | 13.0 GB | 13-15 GB |
| Tokens/s | 97 | 300-500 (Blackwell TC + FP8-ready) |
| Loss Δ over 50 steps | +0.87 | same (same model / same data) |

**With `mamba-ssm` + `flash-attn` + `flash-linear-attention` installed:**
10,000-30,000 tok/s expected. These are the libraries that are needed
on RunPod anyway. Recommendation: install them here right away.

---

## Variant A — Python venv (simplest)

Prerequisite: SSH to the Unraid host, user tools installed.

```bash
# 1. Pull repo (or rsync from the PC):
git clone <repo-url> /mnt/user/auralis_v2_repo
cd /mnt/user/auralis_v2_repo

# 2. venv with Python 3.12 (Blackwell needs CUDA 12.6+):
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# 3. PyTorch CUDA 12.4+ (Blackwell-ready):
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install numpy pyyaml tqdm sentencepiece

# 4. Optional (but dramatically faster):
pip install mamba-ssm flash-attn flash-linear-attention

# 5. Smoke test 250 M, bf16, synthetic:
PYTHONPATH=src python scripts/pretrain/smoke_test.py \
  --device cuda --dtype bf16 \
  --model-config configs/model/helix_v2_250m.yaml \
  --steps 100 --batch-size 8 --seq-length 512 \
  --warmup-steps 10 --lr 1e-3

# 6. Smoke test with REAL tokens (when tokenized/phase1/*.bin is on the NAS):
PYTHONPATH=src python scripts/pretrain/smoke_test.py \
  --device cuda --dtype bf16 \
  --model-config configs/model/helix_v2_250m.yaml \
  --use-real-data \
  --steps 200 --batch-size 8 --seq-length 512 \
  --warmup-steps 20 --lr 3e-4
```

## Variant B — Docker (if Unraid Docker GPU passthrough)

```bash
docker run --rm --gpus all \
  -v /mnt/user/auralis_v2_repo:/workspace \
  -v /mnt/user/Auralis/AuralisV2:/data/auralis \
  -w /workspace \
  nvcr.io/nvidia/pytorch:25.01-py3 \
  bash -c "pip install -q sentencepiece pyyaml tqdm && \
           PYTHONPATH=src python scripts/pretrain/smoke_test.py \
             --device cuda --dtype bf16 \
             --model-config configs/model/helix_v2_250m.yaml \
             --use-real-data \
             --steps 200 --batch-size 8 --seq-length 512 \
             --warmup-steps 20 --lr 3e-4"
```

**Important:** `data_paths.yaml` expects `//BITBASTION/...`. If the
Unraid host is the same machine: `cp configs/data_paths.yaml
configs/data_paths.local.yaml`, set `data_root` there to
`/mnt/user/Auralis/AuralisV2`, and pass `--data-config` on
the call.

## What the report has to show

On success (at the end of the output):

```
  peak VRAM         : 10-15 GB
  loss first        : ~12.2     (≈ ln(200k) = uniform prior)
  loss last         : clearly < first  (at least Δ 0.5 at 100+ steps)
  loss delta        : +0.5+   ✓ learning
  checkpoint        :  ... (reloaded OK)
```

If this runs through without `RuntimeError` / `CUDA OOM` / `NaN`:
**Green light for booking RunPod.** Definitely not before.

## Troubleshooting

| Symptom | Cause / Fix |
|---|---|
| `CUDA OOM` | Lower batch size, halve seq-length |
| `NaN loss` | Try `--dtype fp32`; if stable then → bf16-specific problem |
| `303 tok/s` instead of >5k | Pure-python scan active. Install `pip install mamba-ssm flash-linear-attention` and hook the code later. |
| `FileNotFoundError: .../english.bin` with `--use-real-data` | Check data dir path, adjust `configs/data_paths.yaml` → `data_root` |
| `CUDA kernel image` error | Torch doesn't match the CUDA version. Use `cu124` for Blackwell, possibly `cu126`. |
