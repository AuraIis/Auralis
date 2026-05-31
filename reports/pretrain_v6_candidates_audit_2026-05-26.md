# Pretrain v6 Candidate Download/Audit

Datum: 2026-05-26

## Ergebnis

Vier von fuenf vorgeschlagenen Quellen wurden erfolgreich als capped Kandidaten
geladen und source-getrennt abgelegt. GermanQuAD ist aktuell blockiert, weil der
HF-Dataset-Loader ein Legacy-Script nutzt und die im Script referenzierte
offizielle S3-Datei aus der BITBASTION/Container-Umgebung HTTP 403 liefert.

Output auf BITBASTION:

- Daten: `/mnt/user/Auralis/NEWGPT/v2data/data/training/pretrain_v6_candidates`
- Repo-kopiertes Manifest: `reports/pretrain_v6_candidates_manifest_combined_2026-05-26.json`
- Combined manifest: `/mnt/user/Auralis/NEWGPT/v2data/data/training/pretrain_v6_candidates/manifest_combined.json`
- Source-disjoint manifest: `/mnt/user/Auralis/NEWGPT/v2data/data/training/pretrain_v6_candidates/source_disjoint_manifest.jsonl`

## Quellen

### OpenAssistant/oasst1 Deutsch

- Pfad: `data/training/pretrain_v6_candidates/oasst1_de/oasst1_de.jsonl`
- Records: 1,705
- TXT: 1.29 MB
- Duplikate: 0
- Lizenz: Apache-2.0

Bewertung: brauchbar als kleiner deutscher Instruction-/Dialogblock, aber nicht
blind grossziehen. Audit fand einzelne englische Antworten, Listenlastigkeit und
wenige schlechte Marker. Empfehlung: vor Training einen Strict-Pass fahren:
English-Answer-Drop, URL-Drop, Bad-Marker-Drop, evtl. nur reviewed/high-rank
Pfade.

### deepset/germanquad

- Status: nicht geladen
- Grund: `datasets` 4.8 blockiert Legacy-Dataset-Scripts; `GermanQuAD.zip` aus
  dem offiziellen HF-Script liefert HTTP 403.
- Lizenz laut Dataset Card: CC-BY-4.0

Bewertung: weiterhin sinnvoll, aber wir brauchen eine erreichbare Datenkopie
oder einen alternativen offiziellen Mirror. Nicht durch XQuAD oder synthetische
QA unter gleichem Namen ersetzen.

### HuggingFaceFW/fineweb-2 deu_Latn

- Pfad: `data/training/pretrain_v6_candidates/fineweb2_deu_latn/fineweb2_deu_latn.jsonl`
- Records: 5,000
- TXT: 12.36 MB
- Duplikate: 0
- Lizenznotiz: ODC-By-1.0 laut Dataset Card/Plan

Bewertung: erreichbar und deutsch, aber der erste Stream ist noch zu noisy fuer
unsere Zwecke. Viele behaltene Texte sind Kataloge, Listen, Shops, App-/Film-
Seiten oder boilerplate-nah. Empfehlung: nicht direkt in v6 mischen. Erst
strengere Filter: URL=0, Commerce/Shop raus, Table/List-Density raus, Boilerplate
raus, min. Satz-/Alpha-Dichte, optional Sprache/Perplexity/quality score.

### nvidia/OpenMathInstruct-2

- Pfad: `data/training/pretrain_v6_candidates/openmathinstruct2_capped/openmathinstruct2_capped.jsonl`
- Records: 5,000
- TXT: 5.52 MB
- Duplikate: 0
- Lizenz: CC-BY-4.0

Bewertung: brauchbar als capped Mathe-/Reasoning-Quelle. Englisch dominiert,
was okay ist, solange der Mix-Anteil begrenzt bleibt. Nicht als Val-Tail nutzen;
separate Holdout-Buckets sind im source-disjoint Manifest angelegt.

### codeparrot/codeparrot-clean

- Pfad: `data/training/pretrain_v6_candidates/codeparrot_clean_python_permissive/codeparrot_clean_python_permissive.jsonl`
- Records: 4,260
- TXT: 11.93 MB
- Duplikate: 0
- Lizenz: per-record; nur Apache-2.0, MIT, BSD-2/3, ISC, CC0, Unlicense erlaubt

Bewertung: als Python-Code-Pretrain-Kandidat brauchbar, aber noch mit typischem
Code-Boilerplate: Lizenzheader, URLs in Kommentaren, Security-/Key-Themen und
Framework-/Locale-Dateien. Empfehlung: vorhandenen besseren CodeParrot-Batch 1
weiterverwenden oder hier noch License-Docstring-Stripping, Repo-Dedup und
disjunkte Code-Gates anwenden.

## Source-disjunkte Manifest-Splits

`source_disjoint_manifest.jsonl` wurde mit 0.5% Holdout-Buckets erzeugt:

- oasst1_de: 1,693 train / 12 holdout
- fineweb2_deu_latn: 4,972 train / 28 holdout
- openmathinstruct2: 4,970 train / 30 holdout
- codeparrot_permissive: 4,248 train / 12 holdout

Das ist noch kein finaler Eval-Split, aber verhindert fuer diese Kandidaten den
alten Fehler, dass Validation nur das Ende einer gemischten Datei ist.

## Empfehlung

1. OASST-DE und OpenMathInstruct-2 koennen nach Strict-Pass in einen kleinen v6
   Canary.
2. CodeParrot nur mit dem bereits strengeren permissive Batch und Code-Gate.
3. FineWeb-2 erst nach deutlich haerterem Filter groesser ziehen.
4. GermanQuAD nicht abschreiben, aber erst eine erreichbare offizielle Kopie
   besorgen.
5. Vor jedem Mix: Kontaminationscheck gegen `eval/disjoint_pretrain_gate_v1.yaml`
   und `eval/disjoint_code_eval_v1.jsonl`.
