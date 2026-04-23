# Phase 1 — Pretraining Launch Guide

Dieses Dokument ist die operative Checkliste für den eigentlichen
Phase-1-Pretraining-Lauf auf einem RunPod-Host (H200 oder 4×A40).
Alle Code-Pfade sind bereits implementiert und lokal end-to-end per
`scripts/pretrain/smoke_test.py` validiert.

---

## 1. Voraussetzungen (einmalig)

- [ ] `helix_v2_tokenizer.model` committed in `tokenizer/` (✓ Phase 0)
- [ ] `//BITBASTION/Auralis/AuralisV2/tokenized/phase1/{english,german,code}.bin`
  vollständig erzeugt via `scripts/data/tokenize_for_pretraining.py`
  (läuft ~4 h, 88 GB Input → ~40-50 GB Output)
- [ ] RunPod-Guthaben ≥ **$800** (Phase 1 kostet ~$500-800)
- [ ] `HF_TOKEN` in der Pod-Environment (für Code-Fetch, nicht für Daten —
  alle Phase-1-Daten liegen schon auf dem NAS)
- [ ] `WANDB_API_KEY` in der Pod-Environment (Monitoring)
- [ ] SSH-Zugriff + SMB-Mount: der Pod muss
  `//BITBASTION/Auralis/AuralisV2/tokenized/phase1/` lesen können
  (Alternative: vorab auf das Pod-Volume rsync-en)

## 2. Pod-Konfiguration

**Empfehlung: 1 × H200 SXM (143 GB VRAM)**

| Setup             | VRAM     | $/h       | Phase-1-Kosten |
|-------------------|---------:|----------:|---------------:|
| 1 × H200          | 143 GB   | $3.50-4.50 | $600-750      |
| 4 × A40 48 GB     | 192 GB   | $1.80-2.40 | $300-400      |
| 1 × H100 80 GB    | 80 GB    | $2.50-3.50 | $450-600      |

**4×A40 ist aktuell sweet-spot** (siehe `Doc/SPECs/SPEC_MULTI_GPU_TRAINING.md`).
H200 ist einfacher (kein FSDP/DeepSpeed-Setup).

## 3. Setup auf dem Pod

```bash
# 1. Repo klonen
git clone <repo-url> auralis-v2 && cd auralis-v2

# 2. venv + deps
python -m venv .venv && source .venv/bin/activate
pip install -e ".[train]"
# Zusätzlich auf GPU: optimierte Kernel (große Speedups, Interface bleibt gleich)
pip install mamba-ssm flash-attn
# Optional: flash-linear-attention (GLA CUDA-Kernel, ~20x Speedup ggü. pure-torch)
pip install flash-linear-attention

# 3. HF + WandB Login
huggingface-cli login
wandb login

# 4. NAS-Daten prüfen
ls -la /mnt/nas/Auralis/AuralisV2/tokenized/phase1/
# Erwartet: english.bin, german.bin, code.bin + *.idx + *.manifest.json
```

## 4. Preflight

```bash
# Dry-Run: preflight-check ohne weights zu laden
python scripts/pretrain/train_phase1.py --dry-run
```

Muss ohne Fehler durchlaufen — gibt `preflight ok` aus.

Zusätzlich: **End-to-End Smoke Test** (falls Pod frisch ist):
```bash
python scripts/pretrain/smoke_test.py
# ~30s, PASS = pipeline wired up correctly
```

## 5. Training starten

```bash
# Single-GPU (H200/H100):
python scripts/pretrain/train_phase1.py \
  --config configs/training/phase1_pretrain.yaml

# Multi-GPU (4×A40) via torchrun + FSDP (Config-Erweiterung nötig):
torchrun --nproc_per_node=4 scripts/pretrain/train_phase1.py \
  --config configs/training/phase1_pretrain.yaml
```

**Monitoring:**

- WandB Dashboard: `project=auralis-v2`, Tags `phase1 pretrain helix-v2`
- Log-Felder: `train/loss`, `train/grad_norm`, `train/lr`, `train/tokens_per_second`
- Eval alle 1 000 Steps: `eval/val_loss`
- Alert wenn `val_loss` drei Evals in Folge steigt

## 6. Checkpoints & Resume

- Letzte 3 Step-Checkpoints + `best.pt` in `checkpoints/phase1_pretrain/`
- Alle 10k Steps: externes Backup nach NAS
  (`//BITBASTION/Auralis/AuralisV2/checkpoints/phase1/`)
- Resume von Step N:
  ```bash
  python scripts/pretrain/train_phase1.py \
    --resume checkpoints/phase1_pretrain/step_<N>.pt
  ```

## 7. Erwartete Meilensteine

(aus `Doc/SPECs/SPEC_PHASE_1_PRETRAINING.md` §6)

| Step    | val_loss | Benchmark (ca.)              |
|--------:|---------:|:-----------------------------|
| 1 000   | -        | Loss fällt stetig            |
| 5 000   | < 7.0    | Erste sichtbare Lern-Kurven  |
| 25 000  | < 5.0    | Baseline-Score > 10 %        |
| 50 000  | < 4.0    | HellaSwag > 40 %             |
| 80 000  | < 3.5    | MMLU > 30 %, TRAINING ENDE   |

## 8. Was NACH Phase 1

1. `best.pt` auf NAS und ins Phase-2-Spec übergeben
2. Baseline gegen die 50 Fragen aus `eval/baseline_questions.yaml`
3. Manifest (`MANIFEST.yaml` pro Run — siehe `configs/MANIFEST_TEMPLATE.yaml`) ausfüllen und committen
4. Phase 2 (Continued Bilingual mit KL-Distillation) aufsetzen: siehe
   `Doc/SPECs/SPEC_PHASE_2_CONTINUED_BILINGUAL.md`

## 9. Rollback

- Pre-Training divergiert (NaN loss, train_loss explodiert, grad_norm > 1000):
  - Pod stoppen, **nicht terminieren** (Daten auf Volume behalten)
  - Resume vom letzten gesunden Step-Checkpoint mit
    `--config <kopierter-config-mit-lr-halbiert.yaml>`
- Val-Loss steigt persistent:
  - Data-Mix prüfen (Tokenization-Manifest in `tokenized/phase1/*.manifest.json`)
  - Ggf. LR-Warm-Restart (scheduler auf `constant_with_warmup` schalten)

---

**Pipeline-Status heute:** alle Code-Pfade implementiert, 64/64 Unit-Tests grün,
end-to-end Smoke-Test auf CPU validiert. Einziger offener Blocker vor dem
Launch-Button ist die Tokenization (läuft aktuell im Hintergrund) und der
RunPod-Setup.
