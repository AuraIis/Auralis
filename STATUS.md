# STATUS — Auralis v2

**Letzte Aktualisierung:** 2026-04-22
**Aktive Phase:** Phase 0 — Tokenizer (Vorbereitung)

## Aktueller Stand

- [x] Projektstruktur angelegt (`data/`, `src/`, `scripts/`, `tests/`, `configs/`, `checkpoints/`, `eval/`, `docs/`)
- [x] `pyproject.toml` + `.gitignore` + `README.md`
- [x] Git-Repo initialisiert (branch `main`, 3 Commits)
- [x] 50 Baseline-Fragen als YAML + Runner-Script (`scripts/eval/run_baseline.py --dry` ✓)
- [x] Prompt-Builder + byte-gleiche Training/Inference-Tests (12/12 pytest ✓)
- [ ] Tokenizer-Korpus Download-Script
- [ ] SentencePiece Training + Quality-Report

## Nächster Schritt

Phase 0 Tokenizer: Korpus-Prep (`scripts/tokenizer/prepare_corpus.py`) → SentencePiece-Training
(`train_tokenizer.py`) → Quality-Report mit Token-Effizienz pro Sprache.

## Offene Entscheidungen

- Modellgröße endgültig: **2B** oder **3B**? (Brief: „2–3B"). Einfluss v.a. auf Pretrain-Kosten.
- Multi-GPU-Setup für Phase 1: `1×H200` vs. `4×A40` — siehe
  [SPEC_MULTI_GPU_TRAINING.md](Doc/SPECs/SPEC_MULTI_GPU_TRAINING.md).
- Open-Weights vs. proprietär für Release (Brief §10.4).
