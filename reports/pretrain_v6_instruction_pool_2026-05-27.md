# Pretrain v6 German Instruction Pool - 2026-05-27

## Ziel

Auf BITBASTION wurde ein kleiner, streng gefilterter deutscher Instruction-Pool fuer v6 aufgebaut. Er ist als sauberer SFT-/Instruction-Seed gedacht, nicht als alleiniger Continue-Pretrain-Datensatz fuer das 500M-Modell.

## Quellen

| Quelle | Lizenz | Nutzung | Ergebnis |
| --- | --- | --- | --- |
| FreedomIntelligence/alpaca-gpt4-deutsch | Apache-2.0 | deutsche Instruction/QA-Paare | 19,392 Train / 108 Holdout |
| OpenAssistant/OASST-DE | Apache-2.0 | deutsche Conversations zu Instruction-Paaren normalisiert | 5,339 Train / 27 Holdout |
| OpenAssistant/oasst1 | Apache-2.0 | nicht im Pool genutzt | ausgeschlossen, weil starke Ueberschneidung mit OASST-DE |

## Filter

- Exact-Dedupe ueber normalisierte Prompt/Response-Paare.
- Quellen-disjunkte Holdouts ueber `source_disjoint_manifest_v2.jsonl`.
- Rejects fuer zu kurze/zu lange Texte, URLs, schwaches Deutsch-Signal und schwache Task-Form.
- Gate-nahe Beispiele werden entfernt. Konkret wurde ein OASST-DE-Beispiel zu "Hauptstadt von Deutschland" verworfen, weil es unsere Capability-Gates beruehrt.

## Output

Container-Pfade:

- `data/training/pretrain_v6_instruction_de_strict/instruction_de_train.jsonl`
- `data/training/pretrain_v6_instruction_de_strict/instruction_de_holdout.jsonl`
- `data/training/pretrain_v6_instruction_de_strict/instruction_de_train.txt`
- `data/training/pretrain_v6_instruction_de_strict/manifest.json`
- `data/training/pretrain_v6_instruction_de_strict/contamination_gate_check.json`

Repo-Kopien:

- `reports/pretrain_v6_instruction_de_strict_manifest_2026-05-27.json`
- `reports/pretrain_v6_instruction_de_strict_contamination_2026-05-27.json`

## Zahlen

- Train-Records: 24,731
- Holdout-Records: 135
- Train-Text: 26,872,979 Bytes
- grobe Token-Schaetzung bei 4 Byte/Token: 6,718,244 Tokens
- Eval-Kontamination gegen aktuelle Gates: 0 Prompt-Hits

## Bewertung

Dieser Pool ist sauber genug fuer einen kleinen Instruction/SFT-Test oder fuer einen kontrollierten Mix-Anteil in v6. Er ist aber viel zu klein, um den 500M-Continue alleine sinnvoll zu tragen. Fuer einen neuen stabilen v6-Continue brauchen wir weiterhin:

- mehr source-disjunkte deutsche QA-/Instructiondaten,
- mehr saubere Code-Daten mit ausfuehrbaren disjunkten Evals,
- FineWeb-2 Deutsch nur nach weiterer Qualitaetsfilterung,
- OpenMathInstruct-2 nur gecappt und mit eigener Mathe-Holdout-Quelle,
- kein Training gegen dieselben Seeds/Templates, die in den Gates landen.

## Empfehlung

Als naechstes sollte ein v6-Mix gebaut werden, der diesen Instruction-Pool nur als kleinen Anteil einmischt und daneben deutlich mehr saubere Prosa, Code und Mathe enthaelt. Erst danach lohnt ein neuer 500M-Continue-Lauf; der letzte Mini-Canary war technisch stabil, aber datenmaessig zu klein und zu instruction-lastig.
