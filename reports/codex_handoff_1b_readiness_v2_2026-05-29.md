# Codex Handoff - Auralis 1B Readiness v2 - 2026-05-29

## Kurzfassung

Nicht mit 1B starten, bis der Preflight gruen ist. Die neue verbindliche
Promotion-Eval ist:

`eval/sft_response_frozen_target_retention_v2.yaml`

Sie ist source-disjunkt geprueft, nach zwei lokalen Leak-Fixes sauber, und auf
500M-Checkpoints als harte Baseline gelaufen. Kein 500M-Checkpoint ist
promotable. Das ist Absicht: v2 ist kein Score-Verschoenerer, sondern ein
Regression-Schutz fuer 1B.

## Wichtigste Dateien

- 1B Preflight:
  `scripts/eval/one_b_readiness_preflight.py`
- 1B Preflight Config:
  `configs/eval/auralis_1b_readiness_preflight.yaml`
- Frozen SFT Gate:
  `eval/sft_response_frozen_target_retention_v2.yaml`
- Frozen Gate Runner:
  `scripts/eval/frozen_response_gate.py`
- Guarded 1B Canary Runner:
  `scripts/ops/run_pretrain_1b_canary_readiness.sh`
- Adaptive Frozen-Gate Live Bridge:
  `src/auralis/adaptive/frozen_gate.py`
- Adaptive Trainer CLI:
  `scripts/train/adaptive_curriculum.py`
- Plan/Status:
  `reports/auralis_1b_readiness_plan_2026-05-29.md`
- Aktueller Preflight:
  `reports/auralis_1b_readiness_preflight_v2_2026-05-29.md`

## Was v2 korrigiert

Die v2-Datei aus der Gegenpruefung war gut gebaut, aber in dieser Workspace
nicht sofort leakfrei:

- `retention_spain_capital` kollidierte exakt mit SFT-Trainingsprompts.
  Neuer Prompt:
  `Welche Stadt ist Spaniens Regierungssitz und Hauptstadt?`
- `retention_france_capital` kam als Substring im Pretrain-Mix vor.
  Neuer Prompt:
  `Welche franzoesische Stadt ist Regierungssitz und Hauptstadt?`

Nach der Korrektur:

- SFT-Leakcheck gegen v1/v8/v10/v11/v12-SFT: 0 Kollisionen.
- 1B-Preflight gegen vorhandene Kandidaten: 0 Hash-Kollisionen, 0 Substring-Hits.
- `scripts/eval/one_b_readiness_preflight.py` prueft jetzt zusaetzlich, ob
  eingetragene `cleaned`- und `tokenized`-Pfade wirklich existieren. Bei
  `.bin`-Tokenfiles muss auch die passende `.idx` vorhanden sein.

## Aktueller Preflight-Status

`reports/auralis_1b_readiness_preflight_v2_2026-05-29.md`

- `ready_to_launch: False`
- Eval-Prompts: 70
- Trainings-Einheiten gescannt: 382,763
- Hash-Kollisionen: 0
- Substring-Hits: 0
- Einziger Blocker:
  `configs/data_paths_1b_samples_container.yaml` hat noch leere `cleaned`- und
  `tokenized`-Listen.

Das bedeutet: Die Gate-/Leak-Seite ist sauber. Der Start ist nur blockiert, weil
der finale 1B-Datenmix noch nicht eingetragen/tokenisiert ist.

## 500M Baseline gegen v2

| Checkpoint | Target | Retention | Promotable |
|---|---:|---:|---:|
| `v8_safe` | 8/25 | 18/25 | nein |
| `hybrid_v1_40` | 9/25 | 17/25 | nein |
| `hybrid_v12_bridge_60` | 10/25 | 17/25 | nein |
| `hybrid_v12_repair_v2_80` | 9/25 | 17/25 | nein |

Interpretation:

- v8 ist nur relativ am stabilsten, weil Retention weniger kippt.
- Hybrid/v12 verbessert Target minimal, verliert aber Retention.
- Keine 500M-Variante darf als geloest gelten.
- 500M nur noch als Diagnose nutzen, nicht weiter mit Mini-Patches promoten.

## Verbindliche Policy fuer 1B

- Nie gegen `eval/sft_response_frozen_target_retention_v2.yaml` trainieren.
- Nie bestehende v2-Probes lockern, um einen Lauf gruen zu machen.
- Neue Probes nur append-only ergaenzen.
- Promotable nur, wenn Target und Retention komplett gruen sind.
- Eine Retention-Regression = nicht promotable.
- Known facts muessen confident beantwortet werden.
- Refusal/Honesty nur bei unbekannten/erfundenen Entitaeten, nicht bei Faust,
  Berlin, Bonn historisch, Photosynthese oder einfachen Fakten.

## Adaptive Frozen-Gate Live Bridge

Die adaptive Trainingsschicht kann jetzt das v2-Frozen-Gate live bei jedem Eval
mitlaufen lassen:

```bash
python scripts/train/adaptive_curriculum.py \
  --model-config configs/model/helix_v2_1b.yaml \
  --curriculum configs/curriculum/helix_1b_curriculum_v1.yaml \
  --probes eval/adaptive_margin_probes_v1.yaml \
  --frozen-gate eval/sft_response_frozen_target_retention_v2.yaml \
  --output-dir runs/adaptive_1b_v1 \
  --batch-size 8 --seq-length 2048 --grad-accum 4 --max-steps 200000
```

Neue Live-Metriken im `LearningMonitor`:

- `frozen_target_pass`
- `frozen_retention_pass`
- `frozen_target_failures`
- `frozen_retention_failures`
- `frozen_promotable`

Zusatz-Trace:

`<output-dir>/frozen_gate_trace.jsonl`

Teststatus:

- lokale Gehirn-Tests: 17/17 gruen
- Container-Gehirn-Tests: 17/17 gruen
- Syntaxcheck fuer adaptive Trainer/Kalibrierung/Frozen-Bridge: gruen
- Mini-Smoke im Container mit 500M-v8-Checkpoint und v2-Gate:
  `reports/frozen_gate_live_smoke_2026-05-29.jsonl`

Der Mini-Smoke nutzte nur 2 neue Tokens pro Antwort. Die Scores daraus sind
nicht inhaltlich zu interpretieren; er beweist nur, dass Modell, Tokenizer,
Greedy-Generation, v2-Semantic-Gate und JSONL-Trace verdrahtet sind.

## Naechster technischer Schritt

Der erste 1B-Clean/Tokenized-Kandidatenmix ist eingetragen:

`configs/data_paths_1b_samples_container.yaml`

Aktuell:

- `cleaned.english`: `data/training/curated_40b/english.txt`
- `cleaned.german`: `data/training/curated_40b/german.txt`
- `tokenized.curated_1b_english`: `tokenized/curated_40b/english.bin`
- `tokenized.curated_1b_german`: `tokenized/curated_40b/german.bin`
- Code bleibt im ersten Antwort-/Schreib-Canary draussen.

Preflight:

`reports/auralis_1b_readiness_preflight_curated_40b_v2_2026-05-29.md`

- `ready_to_launch: True`
- Hash-Kollisionen: 0
- Substring-Hits: 0
- grosse Textdateien per `large_text_literal_grep`

Dry-Run:

`train_phase1.py --dry-run` mit
`configs/training/pretrain_1b_canary_readiness.yaml` ist gruen.

Stufenplan:

`reports/auralis_1b_staged_training_plan_2026-05-29.md`

Falls der Mix veraendert wird, erneut:

```bash
python scripts/eval/one_b_readiness_preflight.py \
  --config configs/eval/auralis_1b_readiness_preflight.yaml \
  --output-json reports/auralis_1b_readiness_preflight_v2_$(date -u +%F).json \
  --output-md reports/auralis_1b_readiness_preflight_v2_$(date -u +%F).md
```

Erst wenn `ready_to_launch: true`, den Canary starten:

```bash
bash scripts/ops/run_pretrain_1b_canary_readiness.sh
```

Der Runner startet absichtlich kein Training, solange der Preflight rot ist.
