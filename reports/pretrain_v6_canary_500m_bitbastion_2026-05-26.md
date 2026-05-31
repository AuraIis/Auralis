# Pretrain v6 Canary 500M BITBASTION

Datum: 2026-05-26

## Setup

Checkpoint-Start:

- `/workspace/v2data/checkpoints/pretrain_mix_v5_boosted_500m_a100/best.pt`
- Source-Step: 14500

Canary-Mix:

- Input: `/workspace/v2data/data/training/pretrain_v6_candidates`
- Strict mix: `/workspace/v2data/data/training/pretrain_v6_canary_strict/mix_full.txt`
- Tokenized: `/workspace/v2data/tokenized/pretrain_v6_canary_strict/german.bin`

Mix nach Strict-Filter:

- 11,640 Dokumente
- 18.05 MB Text
- 4,742,719 Tokens
- grob 4.5M Tokens nach 4 bytes/token Schaetzung

Quellen im Mix:

- OASST-DE: 1,658 written
- FineWeb-2 deu_Latn strict: 1,522 written
- OpenMathInstruct-2: 4,969 written
- CodeParrot permissive: 3,491 written

Training:

- Config: `configs/training/pretrain_v6_canary_500m_from_v5_best_bitbastion.yaml`
- Runner: `scripts/ops/run_pretrain_v6_canary_500m_bitbastion.sh`
- Steps: 200
- Tokens seen: 13,107,200
- GPU: NVIDIA RTX PRO 5000 Blackwell
- Throughput peak: ca. 17.3k tok/s
- Exit: completed
- Alerts: none

## Loss

Der kleine Mix wurde technisch sauber gelernt:

- Step 50: val_loss 2.270, german 2.367
- Step 100: val_loss 2.314, german 2.396
- Step 150: val_loss 2.221, german 2.056
- Step 200: val_loss 2.168, german 1.849

Best checkpoint ist Step 200:

- `/workspace/v2data/checkpoints/pretrain_v6_canary_500m_from_v5_best_bitbastion/best.pt`

## Capability-Probes

Ergebnisse liegen unter:

- `/workspace/v2data/data/eval/checkpoint_tests/pretrain_v6_canary_500m_bitbastion`

Vergleich:

| Checkpoint | Probe Set | Aggregate | Bemerkung |
|---|---:|---:|---|
| v5 best step14500 | disjoint_pretrain_gate_v1 | 11.7% | Hallucination guard 70%, sonst 0 |
| v6 canary best step200 | disjoint_pretrain_gate_v1 | 8.3% | QA_DE 15%, Hallucination guard schlechter |
| v6 canary best step200 | v4/v5 gate | 18.2% | schlechter als frueher getestetes v5 best |
| v6 canary best step200 | clean_v2 | 24.4% | leicht unter v5 best clean_v2 final 27.7% |

## Bewertung

Der Canary ist als technischer Smoke erfolgreich, aber nicht als inhaltlicher
Promote-Kandidat.

Was gut ist:

- Training laeuft auf BITBASTION mit dem v5-best Checkpoint.
- Tokenisierung und Checkpointing funktionieren.
- Kein NaN/OOM/Health-Stop.
- Der kleine Mix ist lernbar.

Was schlecht ist:

- Disjunkte Probes verbessern sich nicht.
- Der neue disjunkte Gate sinkt von 11.7% auf 8.3%.
- Clean-v2 bleibt schwach.
- Mathe/Code/QA werden nicht robust besser, obwohl der Mix sie enthaelt.

Wahrscheinlichste Ursache:

- Der Mix ist viel zu klein und zu heterogen. 13.1M trainierte Tokens sind nur
  ein Micro-Finetune auf verrauschten Pretrain-Formaten, kein echter Continue.
- OASST-DE ist klein und teils noisy.
- FineWeb-2 ist trotz Strict-Filter noch nicht hochwertig genug als kleiner
  Signalgeber.
- OpenMath und CodeParrot helfen nicht in 200 Steps, solange die Base selbst
  noch keine stabile QA/Code-Instruktionsform hat.

## Empfehlung

Nicht weiter auf diesem Canary trainieren.

Naechster sinnvoller Schritt:

1. Quellen groesser, sauberer und source-disjunkt vorbereiten.
2. FineWeb-2 Filter deutlich verschaerfen oder durch bessere deutsche Quellen
   ersetzen.
3. OASST-DE durch bessere deutsche Instruction/QA-Daten ergaenzen.
4. Code separat als Code-Canary testen, nicht in denselben Mini-Mix werfen.
5. Erst ab mindestens einigen hundert Millionen sauberen Tokens einen echten
   Continue-Run starten.

