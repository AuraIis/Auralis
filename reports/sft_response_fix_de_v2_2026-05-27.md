# SFT Response Fix DE v2 - Bericht 2026-05-27

## Ziel

Das 500M-Modell soll vor Code-SFT erst sauber auf Deutsch antworten lernen:

- kurze deutsche Basis-QA
- richtige Ja/Nein-Antworten
- False-Premise-Korrektur
- Unsicherheit statt Halluzination
- sauberes Stoppen mit `<|end|>`

Code-Daten wurden weiterhin ausgeschlossen.

## Neue Daten

Builder:

- `scripts/data/build_sft_response_fix_de_v2.py`

Output:

- `data/training/sft_response_fix_de_v2/core_train.helix.jsonl`
- `data/training/sft_response_fix_de_v2/train.helix.jsonl`
- `data/training/sft_response_fix_de_v2/val.helix.jsonl`
- `data/training/sft_response_fix_de_v2/manifest.json`

Stand:

- Core train: 255 kurze, kontrollierte Beispiele
- Full train: 8,252 Beispiele
- Val: 17 source-disjunkte Chat-Eval-Beispiele
- Code: weiter ausgeschlossen

Core-Kategorien:

- hallucination_guard: 86
- facts_de: 86
- qa_de: 32
- instruction_de: 27
- honesty: 24

## Neue harte Eval

Gate:

- `eval/sft_response_fix_chat_gate_v2.yaml`

Dieses Gate wurde aus den gefundenen Fehlerklassen gebaut:

- wahre Ja/Nein-Fragen duerfen nicht mit "Nein" beantwortet werden
- Bonn/Hamburg-False-Premise muss weiter abgelehnt werden
- Goethe/Mein Kampf muss klar korrigiert werden
- Wasser darf nicht als chemisches Element bejaht werden
- Sauerstoff/Photosynthese duerfen nicht mit Wasser verwechselt werden
- Unsicherheit muss explizit genannt werden

## Kandidaten

### A2 Core Phase - aktuell bester v2-Kandidat

Checkpoint:

- `checkpoints/sft_response_fix_de_v2_core_phase_a2/sft_smoke_step_220.pt`

Training:

- Init: `pretrain_v6_expanded_test_500m_from_v5_best_bitbastion/best.pt`
- Train: `sft_response_fix_de_v2/core_train.helix.jsonl`
- LR: 4e-6
- Steps: 220
- EOS weight: 8

Befund:

- v1 Chat-Gate: 100%
- v2 Chat-Gate r2: 68.3%
- sehr gutes Stop-Verhalten
- gute Antworten fuer Bonn, Hamburg/Bayern, Sauerstoff, Photosynthese, Qorblax, Computer
- noch schlecht:
  - wahre Ja-Fragen wie Wien/Oesterreich und Bern/Schweiz werden oft mit "Nein, X ist Hauptstadt..." formuliert
  - `Ist Wasser ein chemisches Element?` kann noch falsch "Ja" sagen
  - `Wer schrieb Faust?` kann halluzinieren
  - Unsicherheitsfrage kann zu "Ich suche nach einer Antwort" kippen

Manueller Smoke nach A2:

- Berlin: korrekt
- Wasser: brauchbar, stoppt
- Goethe/Mein Kampf: meist richtig in laengerer Form
- Bonn: korrekt
- Sauerstoff: korrekt
- Photosynthese: korrekt
- Computer: korrekt

### A3 Positive-Facts-Finetune - verworfen

Checkpoint:

- `checkpoints/sft_response_fix_de_v2_core_phase_a3/sft_smoke_step_160.pt`

Befund:

- positive Ja-Fakten verbessert
- aber Guard bricht:
  - Bonn wird falsch bejaht
  - Hamburg/Bayern wird falsch bejaht
  - Wasser/Element bleibt falsch

Verwerfen als Hauptkandidat.

### A4 Balanced Correction - verworfen

Checkpoint:

- `checkpoints/sft_response_fix_de_v2_core_phase_a4_balanced/sft_smoke_step_100.pt`

Befund:

- Val-Loss etwas besser als A2
- aber v2 Gate schlechter als A2
- Guard/Fact-Balance nicht stabil genug

Verwerfen als Hauptkandidat.

### Full Stabilize from A2 - verworfen

Checkpoint:

- `checkpoints/sft_response_fix_de_v2_full_stabilize_from_a2/sft_smoke_step_60.pt`

Befund:

- Full short mix mit Mini-LR verschlechtert Val und Gate
- Goethe kippt zu "Mein Kampf ist nicht erfunden, sondern nur erfunden"
- breite vorhandene SFT-Zeilen sind fuer diesen Fix noch zu laut

Verwerfen als Hauptkandidat.

## Wichtigster Befund

Das Modell lernt korrektes Antworten, aber die Balance ist empfindlich:

- Zu viel Guard/Core: gute Korrektur, aber enge Muster.
- Zu viel positive Fakten: wahre Ja-Fragen besser, aber False-Premise-Schutz bricht.
- Breiter SFT-Mix: verwischt das fragile Signal.

Der beste aktuelle Punkt ist daher **A2 Core Phase**. Er ist kein finaler Assistant, aber ein klarer Fortschritt und ein guter Diagnoseanker.

## Empfehlung

Nicht mit dem breiten Full-SFT weitermachen, bis der harte v2-Gate stabil ueber 85-90% liegt.

Naechster Daten-/Trainingsschritt:

1. Mehr Paare bauen, die positive und negative Ja/Nein-Fragen paarweise balancieren:
   - `Ist Wien Hauptstadt von Oesterreich?` -> Ja
   - `Ist Salzburg Hauptstadt von Oesterreich?` -> Nein
   - jeweils gleicher Antwortstil
2. Speziell Wasser/Element, Faust/Goethe/Hitler, Unsicherheit und Quellenverhalten weiter ausbauen.
3. Kein grosser Full-Mix, sondern kleine Curriculum-Blöcke:
   - facts_yes_no_balanced
   - false_premise_guard
   - basic_science_qa
   - uncertainty_behavior
4. Nach jedem Block Chat-Gate v2 und manuellen Heldout laufen lassen.

Aktueller Kandidat zum Weiterarbeiten:

- `checkpoints/sft_response_fix_de_v2_core_phase_a2/sft_smoke_step_220.pt`
