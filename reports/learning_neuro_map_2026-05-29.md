# Auralis Live Neuro Map

Datum: 2026-05-29

## Was gebaut wurde

Es gibt jetzt zusaetzlich zum Learning-Trace eine Live-Neuro-Map:

- `scripts/eval/learning_neuro_map.py`
- Erweiterung in `scripts/sft/smoke_sft_de.py`
- Erweiterung in `scripts/sft/run_learning_trace_smoke.sh`

Die Karte ist ein Interpretability-Proxy. Sie zeigt nicht wortwoertlich echte Neuronen, sondern sichtbare Wissensverbindungen aus:

- Zielantwort-Likelihood,
- gefaehrlicher Falschantwort-Likelihood,
- Margin zwischen richtig/falsch,
- generierter Antwort,
- verbotenen Mustern,
- Kategorie und Probe.

## Live-Nutzung

```bash
cd /workspace/v2data
TAG=learning_neuro_live_run STEPS=24 EVAL_EVERY=4 LEARNING_TRACE_EVERY=4 LEARNING_HTML_AUTO_REFRESH=5 \
  bash scripts/sft/run_learning_trace_smoke.sh
```

Waehrend des Trainings werden diese Dateien laufend aktualisiert:

- `reports/learning_trace/<TAG>.json`
- `reports/learning_trace/<TAG>.html`
- `reports/learning_trace/<TAG>_neuro.html`

Die Neuro-HTML hat Auto-Refresh, wenn `LEARNING_HTML_AUTO_REFRESH` groesser 0 ist.

## Testlauf

Getestet mit:

```bash
TAG=learning_neuro_live_test STEPS=1 EVAL_EVERY=1 LEARNING_TRACE_EVERY=1 LR=1e-8 LEARNING_HTML_AUTO_REFRESH=5 \
  bash scripts/sft/run_learning_trace_smoke.sh
```

Output:

- `reports/learning_trace/learning_neuro_live_test.json`
- `reports/learning_trace/learning_neuro_live_test.html`
- `reports/learning_trace/learning_neuro_live_test_neuro.html`

## Was man sieht

Beispiel aus dem Test:

- `de_capital_current`: stark, Berlin wird gegen Bonn bevorzugt.
- `bonn_current_trap`: korrekt, aber noch Watch, weil der Margin klein ist.
- `bern_positive`: generiert korrekt, aber Margin ist negativ; das ist ein verstecktes Risiko.
- `water_not_element`: stark.
- `photosynthesis_core`: Danger, weil die Antwort den Fehler `licht aus licht` enthaelt.
- `faust_author`: korrekt, aber Watch.
- `unknown_entity_honesty`: Danger, weil die Antwort ehrlich startet, danach aber erfundene Details/Abkuerzungs-Drift erzeugt.

## Interpretation

Gruene Kanten bedeuten: Zielantwort ist klar besser als die falsche Alternative.
Gelb/orange bedeutet: Antwort kann richtig aussehen, ist intern aber noch nicht stabil.
Rot bedeutet: verbotene Muster oder falsche Verbindung sind sichtbar.

Das hilft uns vor groesseren SFT-Runs zu sehen, ob Daten wirklich Wissen stabilisieren oder nur ein einzelnes Probe-Pattern bedienen.
