# Auralis v2 — Helix v2 LLM

Modularer deutscher/englischer Assistent. Kleines Basismodell (2–3B dense) + LoRAs + Tools.

**Status:** Phase 0 (Tokenizer-Vorbereitung). Siehe [STATUS.md](STATUS.md).

## Dokumentation

- **Master-Einweisung:** [Doc/AURALIS_V2_PROJECT_BRIEF.md](Doc/AURALIS_V2_PROJECT_BRIEF.md)
- **Phasen-Specs:** [Doc/SPECs/](Doc/SPECs/)
- **Aktueller Stand:** [STATUS.md](STATUS.md)
- **Erkenntnisse:** [LESSONS.md](LESSONS.md)
- **Milestones:** [HISTORY.md](HISTORY.md)

## Projekt-Struktur

```
/data/            Datensätze (raw/cleaned/training/eval) — nicht im Git
/src/auralis/     Python-Paket
  tokenizer/      SentencePiece Wrapper + Chat-Template
  model/          Helix-Modell (Mamba + GLA + Sparse Attention)
  training/       Pretrain / SFT / ORPO / KL-Distillation
  inference/      vLLM / llama.cpp Adapter
  lora/           MoRA / DoRA / GaLore
/scripts/         CLI-Scripts pro Phase
/configs/         YAML-Hyperparameter (modell, training, lora)
/tests/           Pytest-Suites
/checkpoints/     Gewichte — nicht im Git
/eval/            Baseline-Fragen + Ergebnisse
/docs/archive/    Ältere Dokumente
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
.venv\Scripts\activate         # Windows

pip install -e ".[dev]"
pytest
```

## Grundregeln

1. Alles modular — keine hardcoded Werte, alles über Configs
2. Type Hints überall (Python 3.11+)
3. Docstrings für jede Funktion
4. EIN Prompt-Builder für Training = Inference = Eval = API
5. Jede Funktionseinheit bekommt einen atomaren Commit
6. Pro Experiment eine MANIFEST.yaml (Config + Git-Hash + Daten-Hash + Metrics)
