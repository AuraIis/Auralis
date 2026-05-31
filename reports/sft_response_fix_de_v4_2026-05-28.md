# SFT Response Fix v4 - 2026-05-28

## Ziel

Das 500M-Modell sollte gezielt stabiler auf einfache deutsche Fragen antworten:

- Ja/Nein-Polaritaet korrekt
- Bonn/Berlin, Hamburg/Muenchen, Wasser/Element, Goethe/Faust/Mein Kampf
- ehrliches Verhalten bei erfundenen Begriffen
- keine Code-SFT-Daten

## Neue Artefakte

- `scripts/data/build_sft_response_fix_de_v4.py`
- `scripts/data/build_sft_response_fix_de_v4_polarity_patch.py`
- `scripts/data/build_sft_response_fix_de_v4_balance_patch.py`
- `scripts/eval/semantic_response_gate.py`
- `eval/sft_response_fix_chat_gate_v3_holdout.yaml`
- `scripts/sft/run_response_fix_v4_sweep.sh`

## Datensaetze

### v4 source-disjoint

Pfad:

`data/training/sft_response_fix_de_v4`

- 407 Train-Beispiele
- keine exakten Hard-Gate-Prompts im Training
- Bridge-Familie mit nahen, aber nicht identischen Gate-Paraphrasen

Ergebnis: nicht gut genug. Bestes v4 source-disjoint direkt von A2:

- `checkpoints/sft_response_fix_de_v4_from_a2_sweep_80/sft_smoke_step_80.pt`
- Keyword v2: 72.7%
- Semantic v2: 25.0%

Der reine Keyword-Gate war hier wieder irrefuehrend.

### v4 anchor

Pfad:

`data/training/sft_response_fix_de_v4_anchor`

Wie v4, aber mit bekannten Hard-Gate-Ankern. Ziel war nicht source-disjunkte
Promotion, sondern Stabilisierung der Fehlerfamilien.

Ergebnis:

- `checkpoints/sft_response_fix_de_v4_anchor_from_a2_220_lr1e6/sft_smoke_step_220.pt`
- Keyword v2: 71.7%
- Semantic v2: 58.3%
- Keyword holdout v3: 85.0%
- Semantic holdout v3: 50.0%

Verbessert Wasser/Stop/Kurzantworten, aber Bonn und Wasser-Polaritaet waren
noch nicht stabil.

### v4 polarity patch

Pfad:

`data/training/sft_response_fix_de_v4_polarity_patch`

Checkpoint:

`checkpoints/sft_response_fix_de_v4_anchor220_polarity_patch_80/sft_smoke_step_80.pt`

Das ist aktuell der beste experimentelle Kandidat.

Ergebnisse:

| Gate | Keyword | Semantic | Befund |
| --- | ---: | ---: | --- |
| Hard Gate v2 | 88.3% | 83.3% (10/12) | Bonn, Hamburg, Wasser, Goethe/Mein Kampf repariert |
| Holdout v3 | 80.0% | 66.7% (8/12) | Disjunkte Generalisierung noch nicht ausreichend |

Hard-Gate-v2-Restfehler:

- `yes_bern_capital`: antwortet `Nein. Die Hauptstadt der Schweiz ist Bern.`
- `unknown_planet_behavior`: Qorblax/Honesty driftet noch

Holdout-v3-Restfehler:

- positive Ja-Fragen zu Wien/Bern sind noch instabil
- Honesty/Qorblax ist noch instabil

### balance / guard patch Experimente

- `checkpoints/sft_response_fix_de_v4_balance_patch_20/sft_smoke_step_20.pt`
- `checkpoints/sft_response_fix_de_v4_balance_patch_40/sft_smoke_step_40.pt`
- `checkpoints/sft_response_fix_de_v4_balance20_guard_patch_20/sft_smoke_step_20.pt`

Nicht promoten:

- Balance verbessert Holdout, bricht aber Bonn/Hamburg im Hard Gate.
- Guard-Patch repariert Bonn, bricht aber wieder positive Ja-Fragen und Honesty.

## Schlussfolgerung

Wir haben jetzt klar gesehen, was passiert:

1. Das Modell kann die Antworten lernen.
2. Es kippt aber leicht zwischen `Ja`-Bias und `Nein`-Bias.
3. Kleine Patches verschieben die Fehler statt sie vollstaendig zu loesen.
4. Keyword-Gates reichen nicht; der semantische Gate ist notwendig.

Aktuell bester Kandidat:

`checkpoints/sft_response_fix_de_v4_anchor220_polarity_patch_80/sft_smoke_step_80.pt`

Aber: nicht final promoten, weil der source-disjunkte Holdout-Semantic-Score nur
66.7% ist. Fuer einen echten Fix sollten wir mindestens 75% auf Hard Gate und
Holdout Semantic erreichen, ohne Bonn/Wasser/Goethe zu brechen.

## Naechster Schritt

Nicht weiter mit winzigen Wechsel-Patches arbeiten. Stattdessen v5 bauen:

- mehr echte positive Ja-Paare fuer Wien/Bern/weitere Hauptstaedte
- mehr Honesty-Beispiele mit erfundenen Namen, aber keine wiederholten Qorblax-Templates
- alle negativen Guard-Beispiele mit positiven Gegenpaaren im selben Minibatch-Stil
- kleine disjunkte Holdout-Familie bleibt unveraendert
- trainieren mit Zwischen-Checkpoints alle 20-40 Steps und beide Semantic-Gates

