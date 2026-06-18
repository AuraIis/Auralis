# Phase 1 — Pretraining Launch Guide

This document is the operational checklist for the actual
Phase-1 pretraining run on a RunPod host (H200 or 4×A40).
All code paths are already implemented and validated locally end-to-end via
`scripts/pretrain/smoke_test.py`.

---

## 1. Prerequisites (one-time)

- [ ] `helix_v2_tokenizer.model` committed in `tokenizer/` (✓ Phase 0)
- [ ] `//BITBASTION/Auralis/AuralisV2/tokenized/phase1/{english,german,code}.bin`
  fully generated via `scripts/data/tokenize_for_pretraining.py`
  (runs ~4 h, 88 GB input → ~40-50 GB output)
- [ ] RunPod credit ≥ **$800** (Phase 1 costs ~$500-800)
- [ ] `HF_TOKEN` in the pod environment (for code fetch, not for data —
  all Phase-1 data is already on the NAS)
- [ ] `WANDB_API_KEY` in the pod environment (monitoring)
- [ ] SSH access + SMB mount: the pod must be able to read
  `//BITBASTION/Auralis/AuralisV2/tokenized/phase1/`
  (alternative: rsync to the pod volume beforehand)

## 2. Pod configuration

**Recommendation: 1 × H200 SXM (143 GB VRAM)**

| Setup             | VRAM     | $/h       | Phase-1 cost |
|-------------------|---------:|----------:|-------------:|
| 1 × H200          | 143 GB   | $3.50-4.50 | $600-750     |
| 4 × A40 48 GB     | 192 GB   | $1.80-2.40 | $300-400     |
| 1 × H100 80 GB    | 80 GB    | $2.50-3.50 | $450-600     |

**4×A40 is currently the sweet spot** (see `Doc/SPECs/SPEC_MULTI_GPU_TRAINING.md`).
H200 is simpler (no FSDP/DeepSpeed setup).

## 3. Setup on the pod

```bash
# 1. Clone repo
git clone <repo-url> auralis-v2 && cd auralis-v2

# 2. venv + deps
python -m venv .venv && source .venv/bin/activate
# all-linux pulls pretrain + posttrain + lora + inference + dev with all
# CUDA kernels (mamba-ssm, flash-attn, flash-linear-attention via pretrain-extra).
pip install -e ".[all-linux]"
# Triton ≥ 3.6 is required for mamba-ssm / fla on cu128:
pip install --upgrade triton

# 3. HF + WandB login
huggingface-cli login
wandb login

# 4. Check NAS data
ls -la /mnt/nas/Auralis/AuralisV2/tokenized/phase1/
# Expected: english.bin, german.bin, code.bin + *.idx + *.manifest.json
```

### 3a. Enable kernels (important for speed)

```bash
# H100 / H200 (Hopper): all kernels on directly
export AURALIS_USE_CUDA_KERNELS=1

# RTX PRO 5000 / 6000 Blackwell: Triton doesn't fully know sm_120 yet,
# the workaround is emulation as sm89 (Ada). Clean speedup, no
# accuracy loss (loss verified byte-identical to native).
export TRITON_OVERRIDE_ARCH=sm89
export AURALIS_USE_CUDA_KERNELS=1
```

## 4. Preflight

```bash
# Dry run: preflight check without loading weights
python scripts/pretrain/train_phase1.py --dry-run
```

Must run through without errors — prints `preflight ok`.

Additionally: **end-to-end smoke test** (if the pod is fresh):
```bash
python scripts/pretrain/smoke_test.py
# ~30s, PASS = pipeline wired up correctly
```

## 5. Start training

```bash
# Single-GPU (H200/H100):
python scripts/pretrain/train_phase1.py \
  --config configs/training/phase1_pretrain.yaml

# Multi-GPU (4×A40) via torchrun + FSDP (config extension needed):
torchrun --nproc_per_node=4 scripts/pretrain/train_phase1.py \
  --config configs/training/phase1_pretrain.yaml
```

**Monitoring:**

- WandB dashboard: `project=auralis-v2`, tags `phase1 pretrain helix-v2`
- Log fields: `train/loss`, `train/grad_norm`, `train/lr`, `train/tokens_per_second`
- Eval every 1,000 steps: `eval/val_loss`
- Alert when `val_loss` rises for three evals in a row

## 6. Checkpoints & Resume

- Last 3 step checkpoints + `best.pt` in `checkpoints/phase1_pretrain/`
- Every 10k steps: external backup to NAS
  (`//BITBASTION/Auralis/AuralisV2/checkpoints/phase1/`)
- Resume from step N:
  ```bash
  python scripts/pretrain/train_phase1.py \
    --resume checkpoints/phase1_pretrain/step_<N>.pt
  ```

## 7. Expected milestones

(from `Doc/SPECs/SPEC_PHASE_1_PRETRAINING.md` §6)

| Step    | val_loss | Benchmark (approx.)          |
|--------:|---------:|:-----------------------------|
| 1,000   | -        | Loss falls steadily          |
| 5,000   | < 7.0    | First visible learning curves |
| 25,000  | < 5.0    | Baseline score > 10%         |
| 50,000  | < 4.0    | HellaSwag > 40%              |
| 80,000  | < 3.5    | MMLU > 30%, TRAINING END     |

## 8. What AFTER Phase 1

1. Hand over `best.pt` to the NAS and into the Phase-2 spec
2. Baseline against the 50 questions in `eval/baseline_questions.yaml`
3. Fill out the manifest (`MANIFEST.yaml` per run — see `configs/MANIFEST_TEMPLATE.yaml`) and commit
4. Set up Phase 2 (Continued Bilingual with KL distillation): see
   `Doc/SPECs/SPEC_PHASE_2_CONTINUED_BILINGUAL.md`

## 9. Rollback

- Pretraining diverges (NaN loss, train_loss explodes, grad_norm > 1000):
  - Stop the pod, **do not terminate** (keep data on the volume)
  - Resume from the last healthy step checkpoint with
    `--config <copied-config-with-lr-halved.yaml>`
- Val loss rises persistently:
  - Check the data mix (tokenization manifest in `tokenized/phase1/*.manifest.json`)
  - If needed, LR warm-restart (switch scheduler to `constant_with_warmup`)

---

**Pipeline status (as of 2026-04-26):** all code paths implemented + review-validated,
**105/105 unit tests green** on the BITBASTION server, end-to-end smoke test (CPU)
and Blackwell GPU validation PASS, tokenization done (~21B tokens in
`tokenized/curated_40b/`), canary rounds 2 + 3 completed, 1B batch sweep
delivers the final main-run config (seq=2048, batch=4, gradient_checkpointing=on).
Only open blocker: target-host setup (local Blackwell run vs. RunPod
H200/A40) + go/no-go decision. See [STATUS.md](../STATUS.md).
