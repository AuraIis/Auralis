# STATUS — Auralis v2

**Letzte Aktualisierung:** 2026-04-23
**Aktive Phase:** Phase 1 **Code + Infrastruktur fertig**, Tokenization läuft, **Blackwell-GPU-Validation PASS** — RunPod-Launch-ready
**Modellgröße:** 1B (final, Konfig ~954 M Params)
**Phase-1-Token-Budget:** 25B geplant → **21B tatsächlich** bereitgestellt (84 % Deckung; Lücke in Phase 2 schließbar)

## Phase 0 — Tokenizer ✓

**Artefakte** (in `tokenizer/`, versioniert):
- `helix_v2_tokenizer.model` (3.8 MB, 200 k Unigram)
- `helix_v2_tokenizer.vocab` (4.0 MB)
- `training_manifest.yaml`
- `quality_report.md` (alle Gates ✓, Status: PASS)

**Qualitätsprofil** (2 000 Samples pro Sprache):

| Sprache | Tokens/100 Wörter | Tokens/KB | Unknown-Rate | Target |
|---|--:|--:|--:|:-:|
| EN  | 123.0  | 203.4  | 0 % | ≤135 ✓ |
| DE  | 133.8  | 188.7  | 0 % | ≤150 ✓ (v1 GPT-2: ~220) |
| Code | 272.2  | 313.6  | 0 % | ≤350 tok/KB ✓ |

**Chat-Template-Roundtrip:** byte-exakt ✓ — v1-L-001-Bug (Prompt-Format-Konsistenz) architektonisch verhindert.

## Phase-1-Datenlage (auf `//BITBASTION/Auralis/AuralisV2/`)

| Datei | Größe | Tokens est. | Quelle |
|---|--:|--:|---|
| `cleaned/german.txt` | 23.70 GB | ~4.7 B | v1-Reuse (`all_deduped` + `fineweb2_de`) |
| `raw/english/fineweb_edu.txt` | 40.00 GB | ~10.0 B | FineWeb-Edu sample-10BT |
| `raw/english/wikipedia_en.txt` | 12.00 GB | ~3.0 B | wikimedia/wikipedia 20231101.en |
| `raw/english/openmath.txt` | 8.00 GB | ~2.0 B | NVIDIA OpenMathInstruct-2 |
| `raw/code/starcoderdata.txt` | 3.50 GB | ~1.0 B | BigCode StarCoderData (9 Sprachen) |
| `raw/code/open_web_math.txt` | 0.88 GB | ~0.25 B | open-web-math/open-web-math |
| **Total** | **88.08 GB** | **~21 B** | |

Nicht eingeflossen: SlimPajama (entfernt), Dolma (script-basiert), Proof-Pile-2 (script-basiert). Lücke ~4 B EN-Tokens → Phase 2.

**Tokenizer-Korpus** (`tokenizer_corpus/corpus_clean.txt`): 15.5 GB (NUL-bereinigt), Mix 50/40/10 EN/DE/Code.

## Baseline-Eval ✓

- [eval/baseline_questions.yaml](eval/baseline_questions.yaml) — 50 Fragen, 8 Kategorien, EN+DE
- [scripts/eval/run_baseline.py](scripts/eval/run_baseline.py) — läuft gegen jede beliebige `Callable[[str], str]`
- Dry-Run-Smoke-Test grün (6 % Zufalls-Score mit Dummy-Generator)

## Erledigt insgesamt

- Projekt-Skelett, `pyproject.toml`, `.gitignore`, Verzeichnisbaum
- Git-Repo, aktuell ~15 Commits auf `main`
- Byte-exakter Chat-Template-Builder + 12/12 Unit-Tests
- Data-Pipeline: `configs/data_paths.yaml`, atomare Writes, Manifests
- Download-Scripts (englisch/deutsch/code) + v1-Reuse-Script + Inventory
- Tokenizer-Pipeline (`prepare_corpus` → `train_tokenizer` → `report_quality`)
- `LESSONS.md` erweitert um L-007..L-012 (SP-Fallstricke aus Phase 0)

## Phase 0.5 — Modell-Architektur ✓

**Module** (in `src/auralis/model/`):
- `config.py` — `AuralisConfig` + Sub-Configs (Layer/FFN/MoE/MTP/RoPE/Init/Dropout/Advanced)
- `layers/norm.py` — `RMSNorm`
- `layers/ffn.py` — `DenseFFN` (SwiGLU) + `MoEFFN`-Placeholder + `build_ffn` Factory
- `layers/mamba_layer.py` — Mamba-2 Pure-PyTorch Referenz (selective scan)
- `layers/gla_layer.py` — Gated Linear Attention Pure-PyTorch Referenz
- `layers/sparse_attn_layer.py` — Sliding-Window + Global-Tokens Attention
- `utils/rotary.py` — RoPE mit Cache
- `utils/init.py` — Scaled-Normal Init mit Output-Scale-Trick
- `utils/kv_cache.py` — KVCache Dataclass (für Inference später)
- `helix_model.py` — `HelixBlock` + `HelixModel` + `build_model(yaml_path)` Factory

**Configs** (`configs/model/`):
- `helix_v2_100m.yaml` — 8-Layer Test-Modell (2 Mamba + 4 GLA + 2 Sparse, d=512)
- `helix_v2_1b.yaml` — 28-Layer Production (6 Mamba + 16 GLA + 6 Sparse, d=1280, ~954 M Params)

**Tests** (50/50 grün, ~3 s):
- `tests/model/test_config.py` — YAML-Load, Validation, Param-Estimates (10 Tests)
- `tests/model/test_layers.py` — RMSNorm, SwiGLU, RoPE-Roundtrip, Mamba/GLA Forward+Backward, Sparse-Attention Causal-/Window-/Global-Masking (14 Tests)
- `tests/model/test_helix_model.py` — Build, Forward, Backward, Loss, Layer-Reihenfolge, Tied-Embeddings (10 Tests)
- Plus 16 Tests aus Phase 0 (Tokenizer + Baseline + Atomic-Writer)

**Forward-Loss** auf frisch-initialisiertem 100M-Modell: **12.37 ≈ ln(200 000) = 12.20** → uniformer Prior über Vocab, genau wie erwartet. Keine NaN/Inf in Logits oder Gradienten.

## Phase 1 — Pretraining-Pipeline ✓ (Code-ready, Launch ausstehend)

**Scripts**:
- [scripts/data/tokenize_for_pretraining.py](scripts/data/tokenize_for_pretraining.py) — batched SP-Encode auf NAS, atomare Writes, Resume-safe
- [scripts/pretrain/train_phase1.py](scripts/pretrain/train_phase1.py) — CLI mit Preflight, Resume, Device-Override
- [scripts/pretrain/smoke_test.py](scripts/pretrain/smoke_test.py) — 30 s End-to-End-Validation

**Training-Module** (`src/auralis/training/`):
- [dataset.py](src/auralis/training/dataset.py) — `PretrainDataset` (memmap uint32) + `MixedDataLoader` mit largest-remainder Partitionierung
- [optimizer.py](src/auralis/training/optimizer.py) — `build_optimizer` (AdamW + decay-Split für Normen/Biases) + `build_scheduler` (cosine / constant_with_warmup)
- [trainer.py](src/auralis/training/trainer.py) — `PretrainTrainer` mit Grad-Accum, Clip, Checkpoint-Rotation, NaN-Detection, Val-Alarm nach 3 Regressions
- [utils.py](src/auralis/training/utils.py) — `load_yaml`, `set_seed`, `preflight_check`

**Configs**:
- [configs/training/phase1_pretrain.yaml](configs/training/phase1_pretrain.yaml) — 80 k Steps, cosine, bf16, 75/20/5 Mix

**Tests: 64/64 grün in 4 s** (+14 neue: 7 dataset, 4 optimizer, 3 trainer-smoke inkl. NaN-Guard + Checkpoint-Roundtrip).

**CPU-Smoke-Test** (`scripts/pretrain/smoke_test.py`): 20 Steps auf 100M-Modell mit synthetischen Tokens, **PASS in 30 s** — Loss stabil bei 12.28, Checkpoint geschrieben + neu geladen. End-to-End-Wiring bestätigt.

**Launch-Guide**: [docs/PHASE_1_LAUNCH.md](docs/PHASE_1_LAUNCH.md) — alle RunPod-Setup-Schritte + Preflight + Monitoring + Rollback.

## GPU-Validation auf RTX PRO 5000 Blackwell (2026-04-23)

Testumgebung: Unraid-Docker `auralis-training`, CUDA 12.8, torch 2.7.0, Triton 3.6,
RTX PRO 5000 Blackwell 47 GB.

**Installierte Libraries (alle cu128-kompatibel):**
- `flash-linear-attention` (GLA chunk kernels, Triton)
- `mamba-ssm 2.3.1` + `causal-conv1d 1.6.1` (Triton; funktioniert mit Triton 3.6, nicht 3.3)
- `flash-attn 2.8.3` (eigene CUDA-Kernels, kein Triton)

**Ergebnisse 250M-Modell, bf16, batch=4:**

| Config | seq | Backend | tok/s | VRAM | Loss-Korrektheit |
|---|--:|---|--:|--:|:-:|
| Baseline 3090 | 256 | native | 97 | 13.0 GB | ✓ |
| Blackwell | 256 | native | 147-151 | 13.0 GB | ✓ |
| Blackwell | 512 | native | 82 | **24.85 GB** | ✓ |
| Blackwell | 512 | gla-kernel | 88 | **21.27 GB** (-14 %) | ✓ identisch |
| Blackwell | 512 | gla + flash-attn | 88 | 21.27 GB | ✓ identisch |

**Key Takeaways:**
- Native Backend läuft stabil auf Blackwell (≈1.6× schneller als 3090 bei gleichem Shape).
- Kernel-Swap numerisch korrekt — Loss-Werte byte-identisch mit native.
- **Haupt-Win ist VRAM-Ersparnis (-14 %)** durch fla chunk-wise state, nicht tok/s.
  Bei seq=2048 (Phase-1-Config) steigt der Speedup.
- **Mamba-Kernel aktuell auf Blackwell problematisch** — Triton-Compile-Fehler in mamba_ssm trotz
  Triton 3.6 Upgrade. Auf H100/H200 geht er. Für Blackwell-Runs: `AURALIS_USE_GLA_KERNEL=1
  AURALIS_USE_FLASH_ATTN=1` (Mamba bleibt native).

## Offene Blocker vor GPU-Launch

1. **Tokenization läuft** (88 GB → ~40-50 GB binär, Fortschritt: english.bin + german.bin fertig, code.bin läuft noch). Kann ich nicht beschleunigen — ist Disk-I/O-bound.
2. **RunPod-Pod-Setup** (Guthaben, SSH, NAS-Mount, `pip install mamba-ssm flash-attn flash-linear-attention`). User-Aufgabe.
3. **Entscheidung 1 × H200 vs. 4 × A40** — siehe Launch-Guide §2.

## Nächster Schritt nach Phase 1

Phase 2 — Bilingual Continued Pretraining mit KL-Distillation ([SPEC_PHASE_2_CONTINUED_BILINGUAL.md](Doc/SPECs/SPEC_PHASE_2_CONTINUED_BILINGUAL.md)). Grobplan: Teacher-Phase-1-Checkpoint frieren, Student weitertrainieren auf 60/30/10 DE/EN/Code mit `lambda_kd = 0.5`.

## Offene Entscheidungen

- Multi-GPU-Setup für Phase 1 (`1×H200` einfacher, `4×A40` günstiger — siehe [SPEC_MULTI_GPU_TRAINING.md](Doc/SPECs/SPEC_MULTI_GPU_TRAINING.md)).
- Phase-2-Daten-Ergänzung: Ersatz für Dolma/SlimPajama (Cosmopedia, RedPajama-V2?) oder synthetisch auffüllen.
- Open-Weights vs. proprietär für Release (Brief §10.4).
