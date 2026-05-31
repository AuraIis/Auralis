# SFT Response Fix DE v1 - Diagnosebericht 2026-05-27

## Ziel

Code-SFT wurde bewusst ausgeklammert. Dieser Pfad soll nur testen, ob das 500M-Modell zuerst korrekt, knapp und ehrlich auf Deutsch antworten kann:

- deutsche QA
- kurze Instruktionen
- False-Premise-Korrektur
- Unsicherheit statt Halluzination
- sauberes Stoppen mit `<|end|>`

## Neue/angepasste Dateien

- `scripts/data/build_sft_response_fix_de.py`
- `scripts/sft/smoke_sft_de.py`
- `scripts/eval/run_capability_probes.py`
- `eval/sft_response_fix_chat_gate_v1.yaml`
- Daten: `data/training/sft_response_fix_de_v1`

## Datensatz

Build:

- Train: 31,589 Beispiele
- Val: 20 source-disjunkte, handgebaute Eval-Beispiele
- Code aktiv gefiltert: 4,981 Drops
- Hallucination-Guard/Core-Beispiele: 102 synthetische, kurze, kontrollierte Beispiele

Kategorien:

- response_de: 15,250
- qa_de: 10,320
- instruction_de: 5,951
- hallucination_guard: 68

Wichtig: Die Val-Beispiele werden nicht in den Train-Split geschrieben. Sie sind source-disjunkt, aber teilweise absichtlich faehigkeitsnah.

## Trainer-Fixes

Der vorhandene SFT-Smoke wurde zum Diagnose-Trainer erweitert:

- Kategoriegewichtung per `--category-weights`
- EOS-Loss-Boost per `--eos-loss-weight`
- Val-Loss nach Kategorie
- feste Generationen vor/nach SFT
- JSON-Diagnosebericht
- bf16-Autocast in Training, Eval und Generation

Gefundener echter Bug: Generation lief vorher ohne bf16-Autocast. Mit Flash-Attn kam dadurch `BFloat16 != float` in Linear-Layern. Das ist jetzt behoben.

Der Eval-Runner wurde ebenfalls gefixt:

- stoppt jetzt bei `<|end|>` statt SentencePiece-EOS
- unterstuetzt `prompt_style: chat`

## Experimente

### 1. Mixed weighted SFT

Checkpoint:

- `checkpoints/sft_response_fix_de_v1_diag_eos/sft_smoke_step_120.pt`

Einstellungen:

- Init: `pretrain_v6_expanded_test_500m_from_v5_best_bitbastion/best.pt`
- LR: 6e-6
- Warmup: 20
- EOS weight: 5
- Weights: `hallucination_guard=300,qa_de=3,instruction_de=3,response_de=0.5`

Befund:

- Val-Loss fiel von 4.99 auf 2.32
- Bonn/Stop besser
- Goethe blieb schlecht
- Berlin/Wasser hatten noch Wiederholung

### 2. Guard-only microfit

Checkpoint:

- `checkpoints/sft_response_fix_de_v1_guard_only_microfit_v2/sft_smoke_step_100.pt`

Befund:

- Guard lernt sauber.
- Goethe/Mein Kampf: korrekt
- Bonn: korrekt
- Berlin: korrekt
- Stop-Token: korrekt
- Aber allgemeine QA wird zu eng: Wasser -> “Ich bin Auralis.”

Schluss: Das Modell/Trainer-Format ist nicht grundsaetzlich kaputt. Das Problem ist Sampling/Curriculum, nicht reine Lernunfaehigkeit.

### 3. Core-only microfit

Checkpoint:

- `checkpoints/sft_response_fix_de_v1_core_only_microfit/sft_smoke_step_180.pt`

Befund:

- Chat-Gate: 100%
- Smoke:
  - Berlin korrekt
  - Wasser korrekt
  - Goethe korrekt
  - Bonn korrekt
  - Stop sauber

Heldout manuell:

- Hamburg/Bayern: korrekt
- Koeln/NRW: korrekt
- Faust: korrekt
- erfundener Planet: erkennt Unsicherheit
- Sauerstoff: verwechselt mit Wasser
- Photosynthese: falsch/zu duenn
- Computer: teilweise noch Musteruebertragung von Taschenrechner

Schluss: Sehr guter Kern-Fix, aber noch zu eng/overfit.

### 4. Stabilize from core

Checkpoint:

- `checkpoints/sft_response_fix_de_v1_stabilize_from_core/sft_smoke_step_80.pt`

Befund:

- Chat-Gate: 100%
- Disjunkte Val-Loss: 1.63
- Allgemeine QA etwas breiter:
  - Computer korrekt
  - Rhein/Australien korrekt
  - Koeln/NRW korrekt
  - Faust korrekt
- Aber:
  - Sauerstoff wird weiterhin mit Wasser verwechselt
  - Photosynthese ist falsch
  - Unsicherheitsantworten sind grammatisch/inhaltlich noch holprig
  - kurze Goethe-Variante `Schrieb Goethe Mein Kampf?` kann noch kippen, obwohl die laengere Gate-Form klappt

## Wichtigster Befund

Das Modell kann den Fehlerbereich lernen, aber der grosse gemischte SFT laesst das Signal schnell verwischen. Ein einzelner homogener SFT-Lauf ist fuer diesen Checkpoint riskant.

Was funktioniert:

- Curriculum statt alles auf einmal
- erst Guard/Core microfit
- dann sehr sanfte Stabilisierung
- niedrige LR
- EOS-Loss-Boost
- strenger System-Prompt
- Chat-formatierte Eval

Was noch nicht funktioniert:

- breite Sach-Generalisierung
- robuste kurze Antworten ohne passenden System-Prompt
- Wasser/Sauerstoff/Photosynthese-artige Begriffsfragen
- plain QA ohne Chat-Template

## Aktuelle Empfehlung

Nicht als finalen SFT freigeben.

Beste Diagnose-Checkpoints:

- Fuer Guard/False-Premise: `sft_response_fix_de_v1_guard_only_microfit_v2/sft_smoke_step_100.pt`
- Fuer Core-Demo: `sft_response_fix_de_v1_core_only_microfit/sft_smoke_step_180.pt`
- Fuer breiteren Chat-Gate-Test: `sft_response_fix_de_v1_stabilize_from_core/sft_smoke_step_80.pt`

Naechster sinnvoller Schritt:

1. Mehr source-disjunkte, einfache deutsche Basis-QA bauen: Wasser, Luft, Sauerstoff, Pflanzen, Computer, Tiere, Orte, Alltag.
2. Diese Basis-QA nicht als lange Antworten, sondern 1-2 Satz Antworten.
3. Curriculum:
   - Phase A: Guard/Core 100-150 Steps
   - Phase B: Basis-QA/Instruction 100-300 Steps
   - Phase C: gemischter SFT sehr niedrig, mit Guard-Replay
4. Eval immer in Chat-Format und zusaetzlich mit kurzen Varianten derselben Faehigkeit, nicht denselben Prompts.

Kurz: Wir haben jetzt das Testsystem, den Trainer-Bug gefunden und bewiesen, dass der Checkpoint korrigierbar ist. Fuer einen stabilen Fix fehlen noch mehr kurze, saubere, source-disjunkte deutsche Basis-QA-Daten.
