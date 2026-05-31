# Auralis 1B Staged Training Plan - 2026-05-29

## Ziel

Der naechste 1B-Lauf soll nicht wieder ein grosser Blindflug werden. Wir
trainieren stufenweise und lassen Gates live mitlaufen:

1. breite Sprach-/Wissensbasis
2. Antwortformat und Instruction-Following
3. schwierige/kontrastive Reparaturen nur mit Retention-Guard
4. spaetere Evol-/Feedback-Daten erst nach stabiler Basis

## Phase 1: Fundament

Aktueller Kandidat:

- `tokenized/curated_40b/english.bin`
- `tokenized/curated_40b/german.bin`

Tokenbestand laut Manifest:

- Englisch: 11,792,854,409 Tokens
- Deutsch: 5,601,108,978 Tokens
- Code: vorhanden, aber fuer diesen ersten Antwort-/Schreib-Canary nicht im Mix

Canary-Mix:

- 55% Englisch
- 45% Deutsch
- 0% Code

Grund:

- Das Modell soll zuerst allgemeine Sprache, Fakten und Antwortstabilitaet
  verbessern.
- Code/Math koennen spaeter wieder rein, aber nicht bevor einfache Antworten,
  Deutsch und bekannte Fakten stabil sind.

Verdrahtete Dateien:

- `configs/data_paths_1b_samples_container.yaml`
- `configs/training/pretrain_1b_canary_readiness.yaml`
- `configs/curriculum/helix_1b_curriculum_v1.yaml`

## Phase 2: Format / Instruction

Adaptive Stage:

`s2_format_prompt`

Daten:

- `data/training/sft_clean_de_v1/train.helix.jsonl`
- `data/training/sft_response_fix_de_v8_stable_mix/core_train.helix.jsonl`

Ziel:

- Chat-/Tag-Format lernen
- knappe korrekte Antworten
- bekannte Fakten confident beantworten
- Refusal nur fuer unbekannte/erfundene Dinge

## Phase 3: Kontrastive Reparatur

Adaptive Stage:

`s3_contrastive_repair`

Daten:

- `data/training/sft_response_fix_de_v11_contrastive_corrections/core_train.helix.jsonl`
- `data/training/sft_response_fix_de_v12_photo_faust_bridge/core_train.helix.jsonl`

Wichtig:

- Diese Phase ist nicht automatisch "gut", nur weil Loss sinkt.
- Sie darf nur mit Frozen-Gate live laufen.
- Eine Retention-Regression stoppt den Lauf.

## Phase 4: Evol / Feedback

Noch nicht aktiv im ersten Canary.

Naechste Datenklasse, sobald Phase 1-3 stabil sind:

- Evol-Instruct Deutsch/QA
- schwierigere, aber source-disjunkte Varianten
- Execution-Feedback fuer Code erst spaeter
- reasoning traces nur wenn sie sauber und nicht uebervorsichtig machen

## Preflight-Status

Aktueller Report:

`reports/auralis_1b_readiness_preflight_curated_40b_v2_2026-05-29.md`

Ergebnis:

- `ready_to_launch: True`
- Eval-Prompts: 70
- train_units_scanned: 11,558
- fast_text_files_scanned: 2
- Hash-Kollisionen: 0
- Substring-Hits: 0

Hinweis:

- Grosse `.txt`-Korpora werden jetzt per schnellem Literal-Scan geprueft.
- Kleine/SFT-JSONL-Dateien bleiben auf dem strengeren normalisierten Hash-Pfad.

## Dry-Run

`train_phase1.py --dry-run` mit
`configs/training/pretrain_1b_canary_readiness.yaml` ist gruen.

Output:

`preflight ok - exiting (--dry-run)`

## Startregel

Noch kein langer Lauf ohne bewusste Entscheidung.

Wenn gestartet wird:

```bash
bash scripts/ops/run_pretrain_1b_canary_readiness.sh
```

Fuer adaptive Live-Gates:

```bash
python scripts/train/adaptive_curriculum.py \
  --model-config configs/model/helix_v2_1b.yaml \
  --curriculum configs/curriculum/helix_1b_curriculum_v1.yaml \
  --probes eval/adaptive_margin_probes_v1.yaml \
  --frozen-gate eval/sft_response_frozen_target_retention_v2.yaml \
  --output-dir runs/adaptive_1b_v1 \
  --batch-size 8 --seq-length 2048 --grad-accum 4 --max-steps 200000
```
