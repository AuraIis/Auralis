# Auralis 500M v5 A100 - Root-Cause Notiz

Datum: 2026-05-26

## Kurzfazit

Der Run ist technisch gesund, aber die aktuelle Base ist trotz sinkendem Val-Loss
noch nicht robust SFT-ready. Die wahrscheinlichste Ursache ist kein einzelner
Checkpoint- oder Ladefehler, sondern ein Mismatch zwischen Trainings-/Val-Loss
und echten disjunkten Nutzfaehigkeiten:

1. Der Gesamtmix ist ca. 92% Rohtext/Base-Corpus.
2. Die letzten Validierungsdaten kommen aus einem kuenstlich gemischten Tail,
   dessen Muster im Training direkt davor ebenfalls vorkommen.
3. Die festen Probes sind disjunkt und pruefen QA, Code, einfache Fakten und
   saubere Antwortformate. Diese Faehigkeiten sind im Mix zu schwach gewichtet
   oder nicht im passenden Format gelernt.
4. Das 200k-Vocab nimmt bei 500M ca. 205M Parameter in der Embedding-Matrix ein
   (ca. 41% der Parameter). Das laesst weniger Modellkapazitaet fuer die eigentliche
   Transformation und macht seltene Tokens/Code-Patterns teuer.

## Belege

### Probe-Kurve

Vorhandene Reports zeigen:

- Step 5000/6000 A100-Import: ca. 22-28% auf den fruehen Capability-Probes.
- Step 9000: ca. 6.4%.
- Step 14500 best.pt: ca. 6.4% auf V4/V5-Gate, ca. 6.2% auf Clean-v2-Probes.

Der Val-Loss wurde gleichzeitig besser:

- Step 6000: val_loss 1.582
- Step 9000: val_loss 1.433
- Step 14500: val_loss 1.331
- Step 16000: val_loss 1.356, Alert wegen 3 Eval-Anstiegen

Das heisst: Val-Loss und disjunkte Nutzfaehigkeit laufen auseinander.

### Logit-/Decode-Diagnose

Checkpoint-Laden ist sauber:

- Step 14500 best.pt laedt mit missing=0, extra=0.
- mamba_ssm aktiv.
- Logits finite.
- Tokenizer/Chat-Roundtrip OK.

Aber:

- `Frage: Was ist die Hauptstadt von Deutschland?\nAntwort:` -> Zieltoken
  `Berlin` liegt nur etwa Rang 117 bei Step 14500. Das Modell ist also nicht
  wirklich QA-formattreu.
- `Berlin ist die Hauptstadt von` -> `Deutschland` ist bei Step 14500 Rang 2.
  Faktisches Wissen ist teilweise vorhanden, aber nicht stabil genug fuer Greedy.
- `Rechne exakt: 17 + 25 =` -> `42` ist bei Step 14500 nur Rang 3, nicht stabil.
- `def add(a, b):` -> `return` liegt bei Step 14500 grob Rang 56k. Code ist
  nicht ausreichend gelernt.

Damit ist es kein reines Sampling-/Greedy-Problem. Decode verschlechtert einiges,
aber Code und QA sind im Modell selbst schwach.

### Datenmix

Manifest `data/training/pretrain_mix_v5_boosted/manifest.json`:

- clean_v32_base: 26.211 GB, 92.28%
- large QA gesamt: ca. 0.697 GB, ca. 2.46%
- math gesamt: ca. 0.900 GB, ca. 3.17%
- wildchat_en: 0.160 GB, 0.56%
- reddit QA gesamt: ca. 0.117 GB, 0.41%
- balanced_validation_tail: 0.320 GB, 1.13%

Random Samples:

- Gesamtmix: ca. 74% plain, ca. 5% QA/Dialog-like, ca. 11% Math-Format,
  ca. 3% Code-like.
- Letzte 400MB: ca. 71% QA/Dialog-like, ca. 28% Math-Format.

Der Trainer nimmt die letzten `val_split_bytes` als Validation. Weil der Mix am
Ende einen `balanced_validation_tail` hat, ist Validation nicht source-disjunkt:
Train enthaelt den groessten Teil desselben Tail-Blocks, Val ist nur das Ende.
Das ist fuer Loss-Stabilitaet nuetzlich, aber kein guter Skill-Gate.

### Trainingsmenge

Tokens pro Step: 4 * 2048 * 64 = 524288.

- Step 14500: ca. 7.60B Tokens, ca. 1.36 Dataset-Epochen.
- Step 16400: ca. 8.60B Tokens, ca. 1.54 Dataset-Epochen.
- Step 20000: ca. 10.49B Tokens, ca. 1.88 Dataset-Epochen.

Der Run ist also fast bei zwei Durchlaeufen ueber diesen Mix. In den letzten
18% ist kein grosser qualitativer Sprung zu erwarten.

## Wahrscheinlichste Ursachen

1. **Val-Loss ist kein disjunkter Skill-Gate.**
   Er misst zu stark bekannte Source-/Formatverteilungen, nicht robuste QA/Code/
   Instruktionsfaehigkeit.

2. **Der Base-Corpus dominiert.**
   92% Rohtext druecken das Modell in Enzyklopaedie-/Web-Fortsetzungen. Das
   erklaert Antworten wie Stadt-/Bundesland-Fortsetzungen statt kurzer Antworten.

3. **QA/Instruction-Format ist zu schwach und oft Englisch/langform.**
   Die deutschen kurzen Probes passen nicht gut zu den gelernten Mustern.

4. **Code ist faktisch unterversorgt.**
   Der Code-Probe ist klar schlecht; `return` nach einer simplen Funktion ist
   kein naheliegendes Token.

5. **200k Vocab ist fuer 500M teuer.**
   Rund 41% der Parameter sitzen im Embedding. Das kann bei kleinem Modell die
   Generalisierung auf Code/QA/Mathe spuerbar erschweren.

## Empfehlung

Den Run bis 20k fertig laufen lassen, aber nicht automatisch als SFT-Basis
promoten. Danach testen:

- `best.pt`
- `step_20000.pt`
- optional `step_16000.pt` und `step_18000.pt`, falls vorhanden

Gates:

- disjunkte QA/Fact/Math/Code-Probes
- manuelle Sampling-Probes
- Logit-Rank fuer erwartete Zieltoken
- getrennte Source-PPL auf Base, QA, Math, Code, Deutsch

Wenn Step 20000 nicht deutlich besser wird, fuer 1B nicht denselben Mix einfach
skalieren. Vor 1B sollten wir:

- Val source-disjunkt machen.
- Einen separaten disjunkten Skill-Gate fixieren.
- Code- und QA-Daten deutlich staerker und sauberer gewichten.
- Deutsch-spezifische QA/Instruction-Daten ergaenzen.
- 200k Vocab gegen kleineres Vocab oder groesseres Modellbudget abwaegen.
