# Public Code Data Expansion - 2026-05-25

## Ergebnis

Wir haben die alte Code-Datenlage nicht blind erweitert, sondern drei Quellen getrennt getestet:

1. `codeparrot/codeparrot-clean` als robuste Python-Rohcode-Quelle
2. `bigcode/the-stack-v2` als gated, multilingualer Kandidat
3. `nvidia/OpenCodeReasoning-2` als moeglicher Code-SFT/Reasoning-Kandidat

Der nutzbare erste Produktionskandidat ist aktuell CodeParrot, streng gefiltert.

## Neuer Downloader

Datei:

- `scripts/data/download_public_code_sources.py`

Eigenschaften:

- nutzt HuggingFace-Token nur ueber Environment (`HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN`)
- schreibt keine Tokens in Manifest/Logs
- filtert CodeParrot auf permissive Lizenzen: Apache-2.0, MIT, BSD-2/3, ISC, CC0, Unlicense
- verwirft GPL/AGPL/LGPL/MPL/EPL/Artistic im Default
- verwirft generierte Dateien, Vendor-Pfade, zu lange Dateien/Zeilen, Secrets, Syntaxfehler
- entfernt fuehrende Kommentar-Lizenzbloecke vor dem Schreiben
- erzeugt Manifest und Reject-Samples

## CodeParrot Batch 1

Pfad auf BITBASTION/Container-Workspace:

- Raw: `/mnt/user/Auralis/NEWGPT/v2data/data/training/code_public/codeparrot_clean_python_batch1_permissive_raw.txt`
- Filtered: `/mnt/user/Auralis/NEWGPT/v2data/data/training/code_public/codeparrot_clean_python_batch1_permissive_filtered.txt`
- Manifest: `/mnt/user/Auralis/NEWGPT/v2data/data/training/code_public/codeparrot_clean_python_batch1_permissive_filtered.manifest.json`
- Audit: `/mnt/user/Auralis/NEWGPT/v2data/reports/code_audit/codeparrot_clean_python_batch1_permissive_filtered_audit.md`

Zahlen:

- gesehen: 82,986 Records
- nach Download-Filter behalten: 35,000 Docs / 250.0 MB raw
- nach Auralis-Codefilter behalten: 30,320 Docs / 197.7 MB
- geschaetzte Tokens raw: ca. 71.4M
- Sprachen: Python only
- Python Syntax: 30,320 ok, 0 Fehler im Endkorpus
- High-Risk Flag Events: 0
- exakte Duplikate: 0

Wichtige Rejects:

- `license_not_allowed`: 34,177
- `python_syntax_error`: 15,041 beim Download-Filter, plus 2,126 im zweiten Filter
- `vendor_path`: 6,440
- `generated_or_compiled`: 2,553
- `possible_secret`: 239
- nach zweitem Filter: URL-dense 2,054, HTML 821, TODO-dense 159

Rest-Rauschen:

- `license_boilerplate`: 765 Docs / 2.52%
- `url_dense`: 577 Docs / 1.90%

Bewertung:

- deutlich sauberer als alter `curated_40b/code.txt`-Slice
- geeignet als kleiner Python-Code-Booster-Kandidat
- noch nicht direkt in SFT mischen; erst disjunkte Code-Evals und ggf. docstring-license stripping nachschaerfen

## The Stack v2 Smoke

Mit HuggingFace-Auth ist `bigcode/the-stack-v2` erreichbar.

Smoke:

- Input: Python + JavaScript, 1M Token-Ziel
- Output: `/workspace/v2data/raw/code/the_stack_v2_smoke_auth_pyjs_container.txt`
- Filtered: `/workspace/v2data/data/training/code_public/the_stack_v2_smoke_auth_pyjs_filtered.txt`
- Filtered Docs: 603
- Filtered Bytes: 2.26 MB
- Python: 322
- JavaScript: 281
- Python Syntax: 322 ok
- High-Risk Flag Events: 0

Problem:

- `download_the_stack_v2_s3.py` schreibt die Daten und das Manifest fertig, crasht danach aber beim Python-Shutdown mit `terminate called without an active exception` / `PyGILState_Release`.
- Das passierte auch mit `--workers 1`.
- Fuer Produktion muss der Stack-v2-Downloader robustifiziert werden, bevor wir grosse Batches laufen lassen.
- Wichtig: im Container eine Container-Config verwenden, z.B. `configs/data_paths_v5_boosted_container.yaml`; mit der Windows/UNC-Config schreibt er in einen falschen Container-internen `/BITBASTION/...` Pfad.

Bewertung:

- Datenquelle wirkt brauchbar und multilingual.
- Downloader noch nicht produktionsreif.

## NVIDIA OpenCodeReasoning-2 Smoke

Smoke:

- Split: `python`
- gesehen: 3,000
- behalten: 0

Grund:

- Feld `question` war bei den geprueften Records nur `-`.
- `solution`, `r1_generation`, `qwq_critique`, `pass_rate`, `judgement` sind vorhanden.

Bewertung:

- Nicht direkt SFT-ready als Prompt-Antwort-Datensatz.
- Koennte spaeter als Loesungscode/Reasoning-Rohmaterial genutzt werden, aber nur wenn Prompts sauber rekonstruiert oder aus Originalquellen nachgeladen werden.
- Nicht in Auralis-SFT mischen, solange die Disjunktheit und echte Aufgabenstellung nicht gesichert sind.

## Naechste Schritte

1. CodeParrot Batch 1 nicht direkt ins Training mischen, sondern zuerst gegen `eval/disjoint_code_eval_v1.jsonl` als Gate verwenden.
2. Filter verbessern: fuehrende docstring-Lizenzbloecke entfernen, nicht nur Kommentar-Banner.
3. Danach CodeParrot Batch 2 groesser ziehen, z.B. 1-2 GB filtered Ziel, wenn Batch 1 manuelle Stichproben besteht.
4. The Stack v2 Downloader reparieren:
   - robust gegen HuggingFace/Datasets finalizer abort
   - korrekte Container-Config erzwingen
   - optional Sprache einzeln pro Prozess statt multi-language loop
5. Fuer Code-SFT nur disjunkte, testbare Aufgaben verwenden:
   - keine Trainingsprompts in Eval
   - Unit Tests / Syntaxcheck / erwartete Ausgabe
   - kein Score auf oberflaechliche Muster wie `def`/`return`
