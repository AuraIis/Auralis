# STATUS — Auralis v2

**Letzte Aktualisierung:** 2026-04-22
**Aktive Phase:** Phase 0 — Tokenizer (Vorbereitung)

## Aktueller Stand

- [x] Projektstruktur angelegt (`data/`, `src/`, `scripts/`, `tests/`, `configs/`, `checkpoints/`, `eval/`, `docs/`)
- [x] `pyproject.toml` + `.gitignore` + `README.md`
- [ ] Git-Repo initialisieren + erster Commit
- [ ] 50 Baseline-Fragen als YAML
- [ ] Prompt-Builder + byte-gleicher Training/Inference-Test
- [ ] Tokenizer-Korpus Download-Script
- [ ] SentencePiece Training + Quality-Report

## Nächster Schritt

Git initialisieren, dann `eval/baseline_questions.yaml` anlegen — vor dem ersten Training muss das
Test-Set committed sein, damit jede spätere Regression sichtbar wird.

## Offene Entscheidungen

- Modellgröße endgültig: **2B** oder **3B**? (Brief: „2–3B"). Einfluss v.a. auf Pretrain-Kosten.
- Multi-GPU-Setup für Phase 1: `1×H200` vs. `4×A40` — siehe
  [SPEC_MULTI_GPU_TRAINING.md](Doc/SPECs/SPEC_MULTI_GPU_TRAINING.md).
- Open-Weights vs. proprietär für Release (Brief §10.4).
