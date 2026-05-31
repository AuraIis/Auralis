# Pretrain v6 Extra Data Test - 2026-05-27

## Ziel

Mehr Daten fuer v6 finden, laden, auditieren und in einem kleinen Trainings-Smoke testen, ohne den RunPod-Stand oder bestehende Mixe zu ueberschreiben.

## Neue Kandidaten

| Quelle | Lizenzstand | Ergebnis | Entscheidung |
| --- | --- | ---: | --- |
| CUI03/german-commons web/wikipedia | ODC-By + Row-Lizenzen | 3,000 Docs | in Expanded-Test-Mix |
| CUI03/german-commons web/wikidiscussions | ODC-By + Row-Lizenzen | 1,577 Docs | in Expanded-Test-Mix |
| CUI03/german-commons scientific/wikibooks | ODC-By + Row-Lizenzen | 3,000 Docs | in Expanded-Test-Mix |
| CUI03/german-commons scientific/wikiversity | ODC-By + Row-Lizenzen | 973 Docs | in Expanded-Test-Mix |
| codeparrot/codeparrot-clean permissive Python | permissive Row-Lizenzen | 8,000 Docs | in Expanded-Test-Mix, 4,496 neue nach Dedupe |
| oliverguhr/natural-questions-german | CC-BY-SA-3.0 | 8,000 Docs | nur Staging, nicht promoted |
| avemio German-RAG-SFT QA | CC-BY-SA-4.0 | 2,000 Docs | nur Staging, nicht promoted |
| avemio German-RAG-SFT summarizations | CC-BY-SA-4.0 | 2,000 Docs | nur Staging, nicht promoted |
| avemio German-RAG-SFT questions | CC-BY-SA-4.0 | 0 Docs | Filter verwarf alles wegen Listen-/Kontextstruktur |

`jblitzar/github-python` wurde geprueft, aber wegen Repo-Lizenz-Tag `gpl-3.0` nicht verwendet.

## Expanded-Test-Mix

Der neue Mix nutzt den bestehenden strict v6 Mix plus German Commons und den erweiterten permissiven Codepool. ShareAlike-Quellen wurden absichtlich nicht eingemischt.

- Dokumente: 47,411
- Rohtext: 75,129,864 Bytes
- Tokenisiert: 16,670,626 Tokens
- Gate-Kontamination: 0 Prompt-Hits
- Tokenizer: Helix v2, 200,000 Vocab
- Output: `data/training/pretrain_v6_expanded_test_mix/mix_full.txt`
- Tokenized: `tokenized/pretrain_v6_expanded_test_mix/german.bin`

## 100-Step Trainings-Smoke

Checkpoint-Init: `pretrain_mix_v5_boosted_500m_a100/best.pt` aus Step 14,500.

- Config: `configs/training/pretrain_v6_expanded_test_500m_from_v5_best_bitbastion.yaml`
- Output: `checkpoints/pretrain_v6_expanded_test_500m_from_v5_best_bitbastion`
- Status: completed
- Steps: 100
- Tokens gesehen: 6,553,600
- Peak tok/s: 18,304
- VRAM: ca. 8.8 / 13.9 GB
- Alerts: 0
- Val @ 50: 2.619
- Val @ 100: 2.474

Das ist nur ein Pipeline-/Datensmoke, kein finaler Continue.

## Capability-Probes

| Checkpoint | V4/V5 Gate | Clean v2 | Disjoint v1 |
| --- | ---: | ---: | ---: |
| v5 best step 14,500, vorheriger Stand | 28.2% | 27.7% | 11.7% |
| v6 strict mini-canary, vorheriger Stand | 18.2% | 24.4% | 8.3% |
| v6 expanded 100-step best | 52.2% | 47.5% | 22.7% |

Der Expanded-Test ist also klar besser als der alte Mini-Canary. Besonders wichtig: die disjunkten Gates steigen trotz 0 Prompt-Hits im Mix.

## Bewertung

Die Datenrichtung stimmt. German Commons + mehr permissiver Code reparieren den Mix sichtbar besser als der vorherige instruction-lastige Mini-Canary. Trotzdem ist der Mix mit 16.7M Tokens noch klein. Fuer einen echten 500M-Continue brauchen wir mindestens eine deutlich groessere Version davon.

## Empfehlung

1. German Commons groesser ziehen, aber weiter nur gute Splits/Filter: Wikipedia, Wikibooks, Wikiversity; Wikidiscussions nur selektiv.
2. Codeparrot weiter permissiv ausbauen, aber Dedupe gegen bisherigen Codepool behalten.
3. Natural Questions German und Avemio getrennt lassen, bis ShareAlike-Policy geklaert ist.
4. Danach `pretrain_v6_expanded_test_mix` zu einem echten v6 Mix hochskalieren und erst dann einen laengeren 500M-Continue starten.
