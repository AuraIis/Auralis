# SFT Response Fix DE v3 - Bericht 2026-05-28

## Ziel

Nach v2 war klar: Das Modell kippt zwischen zwei Fehlern:

- zu viel Guard: wahre Ja/Nein-Fragen werden mit "Nein" beantwortet
- zu viel positive Fakten: False-Premise-Guard bricht

v3 baut deshalb einen kleinen, streng paarweise balancierten Mikro-Datensatz.

## Dateien

- Builder: `scripts/data/build_sft_response_fix_de_v3.py`
- Daten: `data/training/sft_response_fix_de_v3`
- Diagnose: `reports/sft_response_fix_de_v3_from_a2_balanced_2026-05-28.json`
- Checkpoint: `checkpoints/sft_response_fix_de_v3_from_a2_balanced/sft_smoke_step_120.pt`

## Daten

Core train:

- 117 Beispiele
- facts_de: 56
- hallucination_guard: 42
- honesty: 11
- qa_de: 8

Val:

- 12 source-disjunkte Beispiele

Blocks:

- paired_capitals
- science_and_facts
- goethe_faust_mein_kampf
- honesty

## Training

Init:

- `checkpoints/sft_response_fix_de_v2_core_phase_a2/sft_smoke_step_220.pt`

Settings:

- LR: 8e-7
- Steps: 120
- EOS weight: 8
- Kategoriegewichtung: facts/guard/qa/honesty jeweils 2

Val-Loss:

- Start: 0.5145
- Ende: 0.4166

Smoke nach Training:

- Berlin: korrekt
- Wasser: korrekt
- Goethe/Mein Kampf: korrekt
- Bonn: falsch, kippt zu "Ja. Bonn ist die Hauptstadt von Deutschland."

Hard Chat Gate v2:

- A2 vor v3: 68.3%
- v3 nach 120 Steps: 63.3%

## Fazit

Der v3-Datensatz verbessert Goethe/Wasser und senkt die source-disjunkte Val-Loss, aber er verschlechtert den wichtigsten Guard-Fall Bonn. Damit ist v3 als weiterer Finetune-Zweig aktuell **verworfen**.

Bester aktueller Kandidat bleibt:

- `checkpoints/sft_response_fix_de_v2_core_phase_a2/sft_smoke_step_220.pt`

Warum das wichtig ist:

- Der Loss allein ist nicht ausreichend.
- Selbst ein balancierter kleiner Datensatz kann bei diesem Checkpoint semantische Entscheidungsgrenzen verschieben.
- Das harte Chat-Gate v2 ist noetig, weil einfache Loss-Verbesserung irrefuehrend sein kann.

## Naechster Schritt

Nicht weiter auf dem 500M-Checkpoint herumdruecken, bis bessere Daten vorhanden sind.

Stattdessen:

1. Ein dediziertes Ja/Nein-Eval bauen, das semantisch bewertet, nicht nur Keyword-Gates.
2. Training pro Mini-Block mit sehr fruehem Stop testen: 20, 40, 60 Steps statt 120+.
3. Fuer Hauptmodell lieber A2 als Basis nehmen und erst weitertrainieren, wenn das neue Eval zeigt, dass Bonn/positive Ja-Fakten gemeinsam stabil bleiben.
