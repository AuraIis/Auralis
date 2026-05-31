# Auralis Learning Trace System

Datum: 2026-05-29

## Ziel

Das neue Trace-System macht sichtbar, was das Modell waehrend SFT wirklich lernt:

- ob eine richtige Zielantwort wahrscheinlicher wird,
- ob gefaehrliche falsche Antworten unwahrscheinlicher werden,
- ob die generierte Antwort inhaltlich besser wird,
- welche Faehigkeiten stabil, schwach oder gefaehrlich sind.

Beispiel: Bei "Was ist die Hauptstadt von Deutschland?" wird nicht nur die Ausgabe betrachtet, sondern auch die Wahrscheinlichkeit fuer:

- Ziel: `Die Hauptstadt von Deutschland ist Berlin.`
- Negativ: `Die Hauptstadt von Deutschland ist Bonn.`

Der Abstand `margin = negative_nll - target_nll` ist positiv, wenn das Modell die richtige Antwort bevorzugt.

## Neue Dateien

- `eval/learning_trace_de_core.yaml`
- `scripts/eval/learning_trace_dashboard.py`
- `scripts/sft/run_learning_trace_smoke.sh`
- Erweiterung in `scripts/sft/smoke_sft_de.py`

## Nutzung

Im Container auf Bitbastion:

```bash
cd /workspace/v2data
TAG=learning_trace_de_core_run STEPS=24 EVAL_EVERY=4 LEARNING_TRACE_EVERY=4 \
  bash scripts/sft/run_learning_trace_smoke.sh
```

Output:

- JSON: `reports/learning_trace/<TAG>.json`
- HTML: `reports/learning_trace/<TAG>.html`
- Diag: `reports/learning_trace/<TAG>_diag.json`
- Checkpoint: `checkpoints/<TAG>/sft_smoke_step_<N>.pt`

## Was das Dashboard zeigt

- Target NLL: Wie teuer ist die richtige Antwort?
- Negative NLL: Wie teuer ist eine gefaehrliche falsche Antwort?
- Margin: Positiv ist gut, negativ ist schlecht.
- Delta Margin: Ob sich die Entscheidung waehrend SFT verbessert.
- Antwort-Snapshots: Was das Modell tatsaechlich generiert.
- Forbidden Hits: harte Warnungen, wenn die Antwort verbotene Muster enthaelt.
- Top Next Tokens: Welche Token direkt nach dem Prompt am wahrscheinlichsten sind.

## Testlauf

Getestet mit:

```bash
TAG=learning_trace_de_core_test_v2 STEPS=1 EVAL_EVERY=1 LEARNING_TRACE_EVERY=1 LR=1e-8 \
  bash scripts/sft/run_learning_trace_smoke.sh
```

Ergebnisdateien:

- `reports/learning_trace/learning_trace_de_core_test_v2.json`
- `reports/learning_trace/learning_trace_de_core_test_v2.html`

Letzter Step:

- `de_capital_current`: OK, Berlin bevorzugt.
- `bonn_current_trap`: Watch, richtige Antwort, aber Margin noch klein.
- `bern_positive`: Weak, generiert richtig, aber Negativ-Variante ist noch nicht klar genug unterdrueckt.
- `water_not_element`: OK.
- `photosynthesis_core`: Danger, Antwort enthaelt noch den Fehler `licht aus licht`.
- `faust_author`: Watch, korrekt generiert, aber Margin noch klein.
- `unknown_entity_honesty`: Danger, beginnt ehrlich, driftet aber in erfundene Details.

## Interpretation

Das System zeigt bereits, warum kleine SFT-Patches riskant waren: Manche Antworten sehen korrekt aus, aber die Margin ist klein oder eine verbotene Fortsetzung bleibt wahrscheinlich. Genau solche Faelle sollen vor groesseren Trainings sichtbar werden.

## Naechster Schritt

Vor dem naechsten groesseren Datenlauf sollte jede neue Datenmischung mit diesem Trace laufen. Gute Promotion-Regel:

- keine `Danger`-Probes,
- keine negativen Margins bei Kernwissen,
- Bonn/Berlin und Wasser/Element duerfen nicht regressieren,
- Honest-Answer-Probes muessen ohne erfundene Nachsaetze stoppen.
