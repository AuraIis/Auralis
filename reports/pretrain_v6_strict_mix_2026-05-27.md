# Pretrain v6 Strict Mix - 2026-05-27

## Ziel

Auf BITBASTION wurde ein neuer, isolierter v6-Kandidatenmix gebaut. Er ersetzt den alten Mini-Canary-Mix nicht, sondern liegt separat fuer weitere Tests bereit.

## Quellen im Mix

| Anteil | Quelle | Geschrieben | Bemerkung |
| --- | --- | ---: | --- |
| Instruction-DE | `pretrain_v6_instruction_de_strict` | 24,731 | OASST-DE + Alpaca-GPT4-DE, dedupliziert und gate-bereinigt |
| Mathe | `nvidia/OpenMathInstruct-2` capped | 4,969 | Holdout respektiert, 1 zu langer Eintrag entfernt |
| Code | `codeparrot/codeparrot-clean` permissive Python | 3,491 | Secrets/HTML/URL-dichte und sehr lange Dateien entfernt |
| Prosa-DE | `HuggingFaceFW/fineweb-2` deu_Latn | 1,213 | starker Qualitaetsfilter; viel Boilerplate/List/Table/Commerce entfernt |

`OpenAssistant/oasst1` ist absichtlich nicht im Mix, weil es mit `OpenAssistant/OASST-DE` stark ueberlappt.

## Outputs

Container:

- `data/training/pretrain_v6_strict_mix/mix_full.txt`
- `data/training/pretrain_v6_strict_mix/manifest.json`
- `data/training/pretrain_v6_strict_mix/contamination_gate_check.json`
- `tokenized/pretrain_v6_strict_mix/german.bin`
- `tokenized/pretrain_v6_strict_mix/german.idx`
- `tokenized/pretrain_v6_strict_mix/german.bin.manifest.json`

Repo-Kopien:

- `reports/pretrain_v6_strict_mix_manifest_2026-05-27.json`
- `reports/pretrain_v6_strict_mix_contamination_2026-05-27.json`

## Zahlen

- Dokumente: 34,404
- Rohtext: 43,262,203 Bytes
- grobe 4-BpT-Schaetzung: 10,815,550 Tokens
- echte Tokenisierung mit Helix-v2-Tokenizer: 9,331,801 Tokens
- Tokenisierung: 8.2 Sekunden, keine leeren Zeilen
- Eval-Kontamination gegen aktuelle Gates: 0 Prompt-Hits

## Bewertung

Der Mix ist technisch sauberer als der alte Canary-Mix, weil er den OASST1-Duplikatpfad entfernt und die neuen deutschen Instructiondaten nutzt. Inhaltlich ist er aber weiterhin klein und instruction-lastig. Als Smoke-/Ablation-Datensatz ist er gut; als Fortsetzung fuer das 500M-Modell noch nicht.

## Naechste Schritte

- Mehr source-disjunkte Prosa-DE downloaden und mit dem strengen FineWeb-Filter pruefen.
- Code-Anteil vergroessern, aber nur mit ausfuehrbaren disjunkten Code-Evals.
- GermanQA-Ersatz suchen, weil `deepset/germanquad` per HF-Script/S3 aktuell nicht sauber ladbar war.
- Danach einen groesseren v6-Continue-Mix bauen; erst dann neuen 500M-Continue starten.
