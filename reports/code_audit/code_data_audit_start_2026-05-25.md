# Code Data Audit Start - 2026-05-25

## Kurzfazit

Wir sollten Code-Daten nicht blind weiterverwenden. Der alte `curated_40b/code.txt`
ist in der ersten Python-Scheibe stark verschmutzt. Das aktuelle v5-Pretraining ist
dagegen nicht sichtbar code-lastig; der laufende 500M-Run lernt also eher Deutsch,
QA, Math und Text. Code muss als separater, sauberer Booster/SFT behandelt werden.

## Gepruefte Artefakte

- Audit-Script: `scripts/data/audit_code_corpus.py`
- Filter-Script: `scripts/data/filter_code_corpus.py`
- Disjunkte Code-Eval: `eval/disjoint_code_eval_v1.jsonl`
- Rohcode-Audit: `reports/code_audit/curated_40b_code_audit_50k_v2.md`
- SFT-Code-Audit: `reports/code_audit/sft_clean_de_v1_code_audit_v2.md`
- Gefilterter Kandidat aus 50k Rohdocs: `reports/code_audit/curated_40b_code_clean_candidate_50k_v2.txt`
- Kandidat-Manifest: `reports/code_audit/curated_40b_code_clean_candidate_50k_v2.manifest.json`

## Rohcode-Audit: `curated_40b/code.txt`, erste 50k Docs

- Geprueft: 50,000 Python-Dokumente
- Gescannte Daten: 0.191 GB
- High-risk Events: 36,466 / 72.932%
- Python-Syntax ok: 15,336
- Python-Syntax fehlerhaft: 34,664 / 69.328%
- Exakte Duplikate: 11

Top-Flags:

- `python_syntax_error`: 34,664
- `license_boilerplate`: 4,527
- `generated_or_compiled`: 1,645
- `url_dense`: 1,445
- `too_short_fragment`: 765
- `html_embedded`: 355
- `vendor_or_dependency_path`: 109
- `possible_secret_generic_token`: 33
- `possible_secret_private_key`: 4

Beobachtung: Viele Samples enthalten defekte Python-Dateien, z.B. unquoted
Docstrings/Metadaten, Migrationen, Boilerplate, Downloader-Skripte und URL-lastigen
Code. Das ist fuer Pretraining als kleine Beimischung tolerierbarer als fuer Code-SFT,
aber fuer einen gezielten Code-Booster zu dreckig.

## SFT-Code-Audit: `sft_clean_de_v1`, Code-Kategorien

- Geprueft: 1,579 Code-bezogene SFT-Records
- High-risk Events: 46 / 2.9132%
- Python-Codeblocks ok: 382
- Python-Codeblocks fehlerhaft: 46
- Keine exakten Duplikate

Beobachtung: SFT-Code ist deutlich sauberer, aber viele Records sind Erklaerungen
oder Debug-Hinweise statt ausfuehrbare Code-Aufgaben. Das erklaert, warum ein Code-SFT
Form und Begriffe lernen kann, ohne wirklich robuste Loesungen zu liefern.

## Gefilterter Rohcode-Kandidat aus 50k Docs

Filter-Regeln:

- Mindestlaenge 300 Zeichen
- Keine Syntaxfehler bei Python
- Keine Generated/Compiled/Notebook/HTML/Traceback/vendor/minified Treffer
- Keine offensichtlichen Secrets
- Keine exakten Duplikate
- Import-only-Dateien werden entfernt

Ergebnis:

- Docs gesehen: 50,000
- Docs behalten: 11,715
- Bytes behalten: 31.17 MB
- Kept by language: Python 11,715

Wichtig: Das ist nur ein Startkandidat, kein Produktionsdatensatz. Die erste Datei
enthaelt noch API-/Downloader-Code mit externen URLs. Fuer einen finalen Booster sollten
wir optional URL-lastige Netzwerk-/Scraper-Skripte staerker abwerten.

## Neue disjunkte Code-Eval

`eval/disjoint_code_eval_v1.jsonl` enthaelt 10 kleine Python-Aufgaben mit Unit-Test-
Erwartungen, z.B. `flatten_dict`, `merge_intervals`, `parse_duration`,
`has_cycle`, `safe_get`, `rotate_matrix_clockwise`.

Diese Eval ist absichtlich nicht aus den bisherigen einfachen Code-SFT-Templates
wie Addieren, Sortieren oder for-Schleife erklaeren abgeleitet.

## Naechste Schritte

1. Full oder groesseres Sample-Audit ueber den kompletten Code-Mix laufen lassen,
   aber nur wenn kein wichtiger Training-I/O auf BITBASTION laeuft.
2. Filterregeln fuer Code-Booster nachschaerfen:
   - URL-/Scraper-Code optional reduzieren
   - Migrationen und generated framework files hart entfernen
   - Lizenz- und Kommentarblobs entfernen
   - Secrets-Regex breiter machen
3. Einen kleinen Code-Booster-Kandidaten bauen, nicht direkt SFT:
   - bevorzugt syntaktisch valider Code
   - kein vendor/generated
   - klare Sprache-Marker
   - kontrollierte Sprachbalance
4. SFT erst danach:
   - mit disjunkter Eval
   - Unit-Test-Gate
   - Regression auf Deutsch/Fakten/Refusal, damit keine Code-Kontamination entsteht
