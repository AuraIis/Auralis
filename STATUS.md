# STATUS — Auralis v2

**Letzte Aktualisierung:** 2026-04-22
**Aktive Phase:** Phase 0 — Tokenizer (Vorbereitung) + Daten-Pipeline
**Modellgröße:** **1B** (final)
**Phase-1-Token-Budget:** 25B (75 % EN / 20 % DE / 5 % Code)

## Aktueller Stand

- [x] Projektstruktur + `pyproject.toml` + `.gitignore` + `README.md`
- [x] Git-Repo (`main`, aktuell 5+ Commits)
- [x] 50 Baseline-Fragen + Runner (`scripts/eval/run_baseline.py --dry` ✓)
- [x] Prompt-Builder + byte-gleiche Tests (12/12 ✓)
- [x] Daten-Config `configs/data_paths.yaml` (NAS `//BITBASTION/Auralis/AuralisV2/`, 25 TB frei)
- [x] `scripts/data/_common.py` — atomare Writes, Manifest, Free-Space-Check + 4/4 Tests ✓
- [x] Download-Scripts Phase 1: `download_english.py`, `download_german.py`, `download_code.py`
- [x] `scripts/data/inventory_v1.py` — inventarisiert `I:/Auralis/NEWGPT/data`
- [x] `Doc/SPEC_DATASETS.md` ins Repo aufgenommen
- [ ] `python scripts/data/inventory_v1.py` ausführen (erster Überblick v1-Bestand)
- [ ] Downloads starten (`download_english.py` zuerst — größte Last)
- [ ] Tokenizer-Korpus sampeln (Phase 0)
- [ ] SentencePiece 200k Unigram trainieren + Quality-Report

## Nächster Schritt

1. `pip install -e ".[dev]"` im venv, HF-Token setzen (`huggingface-cli login`)
2. `python scripts/data/inventory_v1.py` — JSON-Report über v1-Datenbestand
3. `python scripts/data/download_english.py --sources wikipedia_en` als kleiner Smoke-Test
   (Wikipedia ist mit ~3B Tokens / ~10 GB der überschaubarste Einstieg)
4. Danach vollständige Phase-1-Downloads (alle drei Scripts parallel möglich)

## Offene Entscheidungen

- Multi-GPU-Setup für Phase 1: `1×H200` vs. `4×A40` — siehe
  [SPEC_MULTI_GPU_TRAINING.md](Doc/SPECs/SPEC_MULTI_GPU_TRAINING.md).
- Open-Weights vs. proprietär für Release (Brief §10.4).
- Zusätzliche synthetische Generierung (DeepSeek V3 / Qwen lokal) bei Lücken — wie in v1 erlaubt & erwünscht.
