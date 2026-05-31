# Auralis v2 / Helix v2

Auralis ist das Assistenz-System. Helix v2 ist das eigene LLM darunter.

Der aktuelle Arbeitsstand steht in [STATUS.md](STATUS.md). Die grosse
Projektidee und Modellphilosophie stehen in
[Doc/AURALIS_V2_PROJECT_BRIEF.md](Doc/AURALIS_V2_PROJECT_BRIEF.md). Die
technische Architektur-Spec steht in
[Doc/SPECs/SPEC_PHASE_0.5_MODEL_ARCHITECTURE.md](Doc/SPECs/SPEC_PHASE_0.5_MODEL_ARCHITECTURE.md).

## Quick Links

- [Aktueller Stand](STATUS.md)
- [Doku-Index](docs/DOCS_INDEX.md)
- [Projekt-Brief / Grundidee](Doc/AURALIS_V2_PROJECT_BRIEF.md)
- [Modell-Architektur](Doc/SPECs/SPEC_PHASE_0.5_MODEL_ARCHITECTURE.md)
- [Data Cleaning Pipeline V3](docs/data_cleaning_pipeline_v3.md)
- [Dataset Market App](docs/dataset_market_app.md)
- [Evaluation](eval/README.md)
- [Lessons](LESSONS.md)
- [History](HISTORY.md)

## Aktueller Fokus

Der alte Base-Run hat gezeigt: Pipeline, Checkpointing, Tokenizer-Roundtrip
und Training funktionieren, aber die damaligen Trainingsdaten waren zu noisy.
Der Fokus liegt deshalb jetzt auf:

1. sauberen Pretraining-Daten statt roher Web-/Archiv-Fragmente,
2. kleinen Canary-Runs vor teuren 1B-Runs,
3. klaren Capability-Evals statt nur Loss,
4. optionalen Boostern wie Knowledge-DNA nur nach messbarem Signal.

## Projekt-Struktur

```text
configs/          YAML-Configs fuer Modell, Training, Daten und Experimente
data/             lokale Daten, Audits und Zwischenartefakte
Doc/              urspruengliche Master-Specs und Phasen-Spezifikationen
docs/             aktuelle Arbeitsdoku und Experimente
eval/             Probes, Benchmarks und Eval-Dokumentation
scripts/          Download, Cleaning, Tokenize, Training, Eval, Experimente
src/auralis/      Python-Paket: Tokenizer, Modell, Training, Inference
tests/            Pytest-Suites
tokenizer/        Helix-v2-Tokenizer und Qualitaetsreport
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Auf dem Trainingsserver laufen viele Jobs im Docker-Container
`auralis-training`. Container-Pfade beginnen dort typischerweise mit
`/workspace/v2data`.

## Grundregeln

1. Der aktuelle Status steht in `STATUS.md`, nicht in alten Phasen-Specs.
2. Specs in `Doc/SPECs/` sind Designhistorie plus Referenz, aber nicht immer
   der heutige Run-Plan.
3. Kein grosser Run ohne Audit, Tokenize-Manifest und Capability-Probes.
4. Keine Tokenizer-Aenderung ohne bewusstes Tokenizer-v2-Experiment.
5. Neue Booster wie Knowledge-DNA bleiben experimentell, bis eine Ablation
   eindeutig positiv ist.
