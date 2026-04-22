# STATUS — Auralis v2

**Letzte Aktualisierung:** 2026-04-23
**Aktive Phase:** Phase 0.5 **abgeschlossen** → bereit für Phase 1 (Pretraining)
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

## Nächster Schritt

**Phase 1 — EN-Heavy Pretraining.** Siehe [Doc/SPECs/SPEC_PHASE_1_PRETRAINING.md](Doc/SPECs/SPEC_PHASE_1_PRETRAINING.md). Wichtige Bausteine vor dem ersten Run:

1. **Tokenizer-Integration in die Datenpipeline** — `scripts/data/tokenize_for_pretraining.py` konvertiert `cleaned/*.txt` → `tokenized/phase1/*.bin` (uint32 memmap-ready)
2. **Data-Loader** mit Streaming aus den `.bin`-Files + Mix-Ratios (75 EN / 20 DE / 5 Code)
3. **Training-Loop** (Torch + FSDP/DeepSpeed auf RunPod) mit Sanity-Check vor dem ersten Step
4. **Monitoring** (WandB) + 50-Baseline-Fragen pro Checkpoint

## Offene Entscheidungen

- Multi-GPU-Setup für Phase 1 (`1×H200` vs. `4×A40`) — siehe [SPEC_MULTI_GPU_TRAINING.md](Doc/SPECs/SPEC_MULTI_GPU_TRAINING.md).
- Phase-1 Backends: Pure-Python-Mamba/GLA reicht für CPU-Tests. Für echtes GPU-Pretraining **zusätzlich** `mamba_ssm` und `flash-linear-attention` installieren und per Config-Flag aktivieren.
- Phase-2-Ergänzung: Ersatz für Dolma/SlimPajama/Proof-Pile-2 suchen (Cosmopedia? RedPajama-V2?) oder synthetisch auffüllen.
- Open-Weights vs. proprietär für Release (Brief §10.4).
