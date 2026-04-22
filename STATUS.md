# STATUS — Auralis v2

**Letzte Aktualisierung:** 2026-04-22
**Aktive Phase:** Phase 0 **abgeschlossen** → bereit für Phase 0.5 (Modell-Architektur)
**Modellgröße:** 1B (final)
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

## Nächster Schritt

**Phase 0.5 — Modell-Architektur.** Siehe [Doc/SPECs/SPEC_PHASE_0.5_MODEL_ARCHITECTURE.md](Doc/SPECs/SPEC_PHASE_0.5_MODEL_ARCHITECTURE.md):

1. `src/auralis/model/config.py` — `AuralisConfig` dataclass
2. `src/auralis/model/layers/` — Mamba-2, GLA, Sparse Attention, FFN, RMSNorm, Rotary
3. `src/auralis/model/helix_model.py` — heterogener 28-Layer-Stack
4. `configs/model/helix_v2_100m.yaml` (Test) und `helix_v2_1b.yaml` (Produktion)
5. Unit-Tests für jeden Layer + kompletter Forward/Backward auf 100 M-Variante

## Offene Entscheidungen

- Multi-GPU-Setup für Phase 1 (`1×H200` vs. `4×A40`) — siehe [SPEC_MULTI_GPU_TRAINING.md](Doc/SPECs/SPEC_MULTI_GPU_TRAINING.md).
- Phase-2-Ergänzung: Ersatz für Dolma/SlimPajama/Proof-Pile-2 suchen (Cosmopedia? RedPajama-V2?) oder synthetisch auffüllen.
- Open-Weights vs. proprietär für Release (Brief §10.4).
