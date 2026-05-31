# SFT Response Fix v3 Sweep - 2026-05-28

## Ziel

Kurzer Early-Stop-Sweep von `sft_response_fix_de_v2_core_phase_a2/sft_smoke_step_220.pt`
auf dem kleinen v3-Micro-Curriculum. Ziel war zu testen, ob 20/40/60 Schritte
die positiven Ja/Nein-Fakten verbessern, ohne die Guards wie Bonn/Hamburg zu
zerstoeren.

## Neues Diagnose-Werkzeug

Hinzugefuegt:

- `scripts/eval/semantic_response_gate.py`
- `scripts/sft/run_response_fix_v3_sweep.sh`

Der semantische Gate wertet die bestehenden Capability-Probe-JSONs nach. Er
markiert besonders:

- falsche explizite Polaritaet, z. B. `Nein. Wien ist die Hauptstadt...`
- falsche Praemisse akzeptiert, z. B. `Ja. Bonn ist die Hauptstadt...`
- richtige Stichwoerter, aber falsche Aussage

Das ist strenger und aussagekraeftiger als das reine Keyword-Gate.

## Baseline-Vergleich

| Checkpoint | Keyword-Gate v2 | Semantic-Gate | Befund |
| --- | ---: | ---: | --- |
| A2 `sft_smoke_step_220.pt` | 68.3% | 58.3% (7/12) | Beste bisherige Basis, aber mit Ja/Nein-Widerspruch und Wasser/Faust-Fehlern |
| v3 120 steps | 63.3% | 58.3% (7/12) | Repariert Wien, zerstoert aber Bonn |

## Early-Stop-Sweep

Trainingsbasis:

`checkpoints/sft_response_fix_de_v2_core_phase_a2/sft_smoke_step_220.pt`

Daten:

`data/training/sft_response_fix_de_v3/core_train.helix.jsonl`

Ergebnisse:

| Run | Val Loss | Keyword-Gate v2 | Semantic-Gate | Kurzbefund |
| --- | ---: | ---: | ---: | --- |
| v3 sweep 20 | 2.2340 | 22.5% | 16.7% (2/12) | Stark instabil, viele Wiederholungen |
| v3 sweep 40 | 1.2113 | 48.3% | 16.7% (2/12) | Keyword steigt, semantisch weiter schlecht |
| v3 sweep 60 | 0.9921 | 61.7% | 16.7% (2/12) | Keyword fast brauchbar, aber Aussagen widersprechen sich |

Beispiele:

- 40 steps: `Ja. Die Hauptstadt von Deutschland ist Berlin.` auf Bonn-Frage.
- 60 steps: `Nein. Bonn ist die Hauptstadt von Deutschland.`
- 60 steps: Wasser wird besser formuliert, aber andere Gates bleiben semantisch falsch.

## Schlussfolgerung

Der v3-Micro-Datensatz ist als direkter Finetune-Impuls nicht stabil genug. Er
senkt Loss und verbessert manche Keywords, aber er erzeugt weiter semantische
Widersprueche. Der reine Loss ist hier kein verlaessliches Auswahlkriterium.

A2 bleibt der beste Kandidat. Weder v3 120 noch die kurzen v3-Sweeps sollten
promoted werden.

## Naechster sinnvoller Schritt

Nicht weiter auf demselben Mini-Mix trainieren. Stattdessen:

1. Einen groesseren, source-disjunkten Response-Fix-v4 bauen.
2. Ja/Nein-Paare strikt symmetrisch halten: wahr/falsch pro Relation, nicht nur pro Thema.
3. Wasser/Faust/Honesty als eigene Guard-Familien mit mehreren Oberflaechenformen bauen.
4. Train nicht nur auf korrekten Antworten, sondern auf "Answer shape": erst Ja/Nein, dann kurze Begruendung.
5. Alle Kandidaten mit Keyword-Gate plus `semantic_response_gate.py` pruefen.

Empfohlene harte Promotion-Regel:

- Keyword-Gate >= 75%
- Semantic-Gate >= 75%
- Kein Fehler auf `no_bonn_current`, `water_not_element`, `goethe_faust_author`

