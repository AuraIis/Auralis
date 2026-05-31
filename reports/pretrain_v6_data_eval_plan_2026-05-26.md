# Auralis v6 Daten- und Eval-Plan

Datum: 2026-05-26

## Entscheidung

Wir sollten die naechste Runde nicht durch selbst geschriebene Trainingsdaten
retten. Selbst geschriebene Daten sind gut fuer kleine, feste Gates und Canary-
Tests, aber als Haupttraining wuerden sie zu schnell Template-Muster erzeugen.

Fuer Training nehmen wir externe, lizenzierbare Quellen mit klarer Source-
Trennung. Fuer Eval/Gates schreiben und pflegen wir eigene kleine Tests, plus
offizielle Test-Splits, die nie in Training gehen.

## Was jetzt angelegt wurde

- `configs/data/pretrain_v6_source_disjoint_plan.yaml`
- `configs/data/pretrain_v6_vocab_budget_notes.yaml`
- `eval/disjoint_pretrain_gate_v1.yaml`

Diese Dateien sind noch kein finaler 1B-Mix. Sie sind die Leitplanken, damit wir
den naechsten Canary/Mix nicht wieder mit Tail-Validation und vermischten Quellen
bewerten.

## Externe Kandidaten

### Sofort sinnvoll

- **OpenAssistant/oasst1**: Apache-2.0, 35 Sprachen, auch Deutsch. Nutzen als
  kleiner, human annotierter Deutsch/Englisch-Instruktionsblock. Nur reviewed,
  non-spam, beste Pfade.
- **deepset/germanquad**: CC-BY-4.0, deutsch, train/test vorhanden. Train kann
  in QA-Training, Test bleibt Gate/Holdout.
- **codeparrot/codeparrot-clean**: Python, per-row Lizenzfeld. Batch 1 ist bei
  uns bereits permissiv gefiltert und syntaktisch sauber. Naechster Schritt:
  groesseren Batch ziehen, aber nur nach disjunktem Code-Gate.
- **HuggingFaceFW/fineweb-2 deu_Latn**: guter Kandidat fuer deutschen Web-Prose-
  Top-up. Vor Produktion Lizenz/Config exakt pruefen und Holdout-Shards fixieren.
- **nvidia/OpenMathInstruct-2**: CC-BY-4.0, als Mathe-Reasoning capped sinnvoll.

### Nur mit Vorsicht

- **bigcode/the-stack-v2**: inhaltlich passend fuer breiteres Code-Pretraining,
  aber unser Downloader hatte noch einen Shutdown/Finalizer-Crash. Erst reparieren
  und dann per Sprache/Repo disjunkt ziehen.
- **nvidia/OpenCodeReasoning-2**: CC-BY-4.0, aber unser Smoke zeigte bei den
  geprueften Records kein brauchbares `question`-Feld. Nicht direkt SFT-ready.
- **nvidia/ChatQA-Training-Data**: enthaelt nuetzliche QA-Subsets, aber der
  synthetic conversational QA Teil ist laut Dataset Card non-commercial/OpenAI-
  ToU-gebunden. Nur einzelne non-synthetic Unterquellen nutzen, wenn deren
  Original-Lizenzen sauber sind.
- **wikimedia/wikipedia de**: guter Faktenanker, aber CC-BY-SA/GFDL. Nur bewusst
  und capped einsetzen, weil Share-Alike/Attribution fuer Releases wichtig ist.

## Source-disjunkte Validation

Der wichtigste Wechsel:

- Kein `val_split_bytes` als Tail des finalen Mixes mehr.
- Jede Quelle bekommt feste Train/Holdout-Regeln.
- Code wird repo-disjunkt getrennt.
- QA wird dataset-/split-disjunkt getrennt.
- Gates werden nie in Training geschrieben.

Neue Gate-Datei:

- `eval/disjoint_pretrain_gate_v1.yaml`

Bestehendes Code-Gate bleibt:

- `eval/disjoint_code_eval_v1.jsonl`

## Vocab/Modellbudget

Der aktuelle 500M-Lauf nutzt `vocab_size=200000`. Bei hidden size 1024 sind das
204.8M tied embedding parameters, also ca. 41% des 500M-Modells. Das ist fuer
Deutsch/Code bequem, aber fuer ein kleines Modell sehr teuer.

Vor 1B sollten wir 100k, 128k und 200k Tokenizer/Modelle kurz gegeneinander
messen: Tokens/Byte auf Deutsch, Code und QA, plus disjunkte Gates. 200k kann
bei 1B Sinn machen, aber es sollte nicht mehr ungeprueft gesetzt sein.

## Naechste konkrete Schritte

1. Script fuer Source-Manifeste bauen: Train/Holdout nach Source, Repo, Shard
   und offiziellen Splits.
2. OASST1 Deutsch + GermanQuAD klein ziehen und filtern.
3. CodeParrot Batch 2 ziehen, aber gegen `eval/disjoint_code_eval_v1.jsonl`
   kontaminationspruefen.
4. The-Stack-v2-Downloader reparieren, danach nur Smoke pro Sprache.
5. Canary-Mix bauen und nur promoten, wenn Source-PPL und disjunkte Gates beide
   besser werden.

