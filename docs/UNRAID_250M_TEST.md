# 250M-Testlauf auf Unraid (RTX Pro 5000 Blackwell)

Dieser Lauf validiert die volle Pipeline auf einer Workstation-GPU, bevor
ein mehrstündiger RunPod-Lauf gebucht wird. 250 M Modell, `bf16`,
heterogener Stack (Mamba + GLA + Sparse).

**Erwartung (Blackwell 48 GB, pure-python Kernel):**

| Metrik | 3090 gemessen | Pro 5000 Blackwell erwartet |
|---|--:|--:|
| Peak VRAM | 13.0 GB | 13-15 GB |
| Tokens/s | 97 | 300-500 (Blackwell TC + FP8-ready) |
| Loss Δ über 50 Steps | +0.87 | gleich (gleiches Modell / gleiche Daten) |

**Mit `mamba-ssm` + `flash-attn` + `flash-linear-attention` installiert:**
10 000-30 000 tok/s erwartet. Das sind die Libraries, die ohnehin auf
RunPod gebraucht werden. Empfehlung: hier gleich mitinstallieren.

---

## Variante A — Python venv (am einfachsten)

Voraussetzung: SSH auf den Unraid-Host, User-Tools installiert.

```bash
# 1. Repo ziehen (oder rsync vom PC):
git clone <repo-url> /mnt/user/auralis_v2_repo
cd /mnt/user/auralis_v2_repo

# 2. venv mit Python 3.12 (Blackwell braucht CUDA 12.6+):
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# 3. PyTorch CUDA 12.4+ (Blackwell-ready):
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install numpy pyyaml tqdm sentencepiece

# 4. Optional (aber dramatisch schneller):
pip install mamba-ssm flash-attn flash-linear-attention

# 5. Smoke-Test 250 M, bf16, synthetisch:
PYTHONPATH=src python scripts/pretrain/smoke_test.py \
  --device cuda --dtype bf16 \
  --model-config configs/model/helix_v2_250m.yaml \
  --steps 100 --batch-size 8 --seq-length 512 \
  --warmup-steps 10 --lr 1e-3

# 6. Smoke-Test mit ECHTEN Tokens (wenn tokenized/phase1/*.bin auf dem NAS steht):
PYTHONPATH=src python scripts/pretrain/smoke_test.py \
  --device cuda --dtype bf16 \
  --model-config configs/model/helix_v2_250m.yaml \
  --use-real-data \
  --steps 200 --batch-size 8 --seq-length 512 \
  --warmup-steps 20 --lr 3e-4
```

## Variante B — Docker (wenn Unraid-Docker GPU-Passthrough)

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

**Wichtig:** `data_paths.yaml` erwartet `//BITBASTION/...`. Wenn der
Unraid-Host dieselbe Maschine ist: `cp configs/data_paths.yaml
configs/data_paths.local.yaml`, dort `data_root` auf
`/mnt/user/Auralis/AuralisV2` setzen, und `--data-config` beim Aufruf
übergeben.

## Was der Report zeigen muss

Bei Erfolg (am Ende des Outputs):

```
  peak VRAM         : 10-15 GB
  loss first        : ~12.2     (≈ ln(200k) = uniform prior)
  loss last         : deutlich < first  (mind. Δ 0.5 bei 100+ Steps)
  loss delta        : +0.5+   ✓ learning
  checkpoint        :  ... (reloaded OK)
```

Wenn das durchläuft ohne `RuntimeError` / `CUDA OOM` / `NaN`:
**Grünes Licht für RunPod-Buchung.** Vorher definitiv nicht.

## Troubleshooting

| Symptom | Ursache / Fix |
|---|---|
| `CUDA OOM` | Batch-Size senken, seq-length halbieren |
| `NaN loss` | `--dtype fp32` probieren; wenn dann stabil → bf16-spezifisches Problem |
| `303 tok/s` statt >5k | Pure-python-Scan aktiv. `pip install mamba-ssm flash-linear-attention` installieren und Code-Hook später. |
| `FileNotFoundError: .../english.bin` bei `--use-real-data` | Daten-Dir-Pfad prüfen, `configs/data_paths.yaml` → `data_root` anpassen |
| `CUDA kernel image` error | Torch nicht passend zur CUDA-Version. `cu124` für Blackwell nehmen, ggf. `cu126`. |
