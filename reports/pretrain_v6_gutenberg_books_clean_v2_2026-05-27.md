# Pretrain v6 Gutenberg Books Clean v2 - Testbericht 2026-05-27

## Ziel

Gutenberg/Open-Books sollten nach dem Download nicht roh in den Mix gehen. Ziel dieses Schritts war:

- Gutenberg-/Etext-Boilerplate entfernen
- Lizenz- und Disclaimer-Bloecke entfernen
- Seitenlisten, Inhaltsverzeichnisse, Index/Glossar und reine All-Caps-Fragmente entfernen
- Gate-Kontamination pruefen
- den sauberen Books-Mix niedrig dosiert gegen den bisherigen v6-expanded-Best testen

Quellen:

- common-pile/project_gutenberg: https://huggingface.co/datasets/common-pile/project_gutenberg
- zkeown/gutenberg-corpus: https://huggingface.co/datasets/zkeown/gutenberg-corpus

## Dateien

- Cleaner: `scripts/data/clean_book_sources_v2.py`
- Rohdaten: `data/training/pretrain_v6_book_sources_gutenberg_v1/book_sources.jsonl`
- Clean JSONL: `data/training/pretrain_v6_book_sources_gutenberg_v1/book_sources.clean_v2.jsonl`
- Clean Text: `data/training/pretrain_v6_book_sources_gutenberg_v1/book_sources.clean_v2.txt`
- Reject Samples: `data/training/pretrain_v6_book_sources_gutenberg_v1/book_sources.clean_v2.reject_samples.jsonl`
- Manifest: `data/training/pretrain_v6_book_sources_gutenberg_v1/book_sources.clean_v2.manifest.json`
- Source-disjoint Manifest: `data/training/pretrain_v6_book_sources_gutenberg_v1/source_disjoint_manifest_clean_v2.jsonl`
- Tokenized: `tokenized/pretrain_v6_book_sources_gutenberg_clean_v2/english.bin`
- Test-Config: `configs/training/pretrain_v6_books_clean_v2_lowratio_500m_from_expanded_best_bitbastion.yaml`
- Runner: `scripts/ops/run_pretrain_v6_books_clean_v2_lowratio_500m_bitbastion.sh`
- Checkpoint: `checkpoints/pretrain_v6_books_clean_v2_lowratio_500m_from_expanded_best_bitbastion/best.pt`

## Clean-v2 Ergebnis

Input:

- Dokumente rein: 334,952
- Textumfang roh: ca. 1.51 GB

Output:

- Dokumente geschrieben: 323,794
- Textumfang sauber: 1,460,573,853 Bytes
- Token nach Tokenisierung: 337,649,654
- Gate-Kontamination: 0 Treffer

Sprachverteilung nach Cleaner:

- Englisch: 320,530 Dokumente
- Deutsch: 3,264 Dokumente

Entfernte Artefakte:

| Grund | Anzahl |
|---|---:|
| gutenberg_boilerplate | 6,333 |
| page_list | 3,478 |
| index_glossary | 450 |
| front_back_matter | 250 |
| all_caps | 231 |
| license_short | 143 |
| table_of_contents | 119 |
| license_block | 83 |
| too_few_words | 61 |
| too_short | 9 |
| mixed_gutenberg_bundle | 1 |

Bewertung: Der Cleaner macht den Datensatz deutlich brauchbarer. Besonders Lizenztext, Projekt-Gutenberg-Hinweise, Inhaltsverzeichnisse, Seitenindex-Fragmente und reine Metadaten wurden sichtbar reduziert. Das ist als Buch-Prose-Top-up sauber genug fuer weitere Experimente.

## 100-Step Smoke

Setup:

- Init: `pretrain_v6_expanded_test_500m_from_v5_best_bitbastion/best.pt`
- Mix: 20% Clean-v2 Books, 80% v6 expanded German/code mix
- Steps: 100
- LR: 5e-6 cosine, warmup 10, min_lr_ratio 0.4
- Batch: `batch_size_per_device=1`, `gradient_accumulation=32`, `seq_length=2048`
- GPU: NVIDIA RTX PRO 5000 Blackwell

Training lief stabil:

- keine Alerts
- keine NaNs/OOMs
- Peak: 18,115 tok/s
- VRAM: ca. 8.8/13.9 GB
- Tokens gesehen: 6,553,600
- Best-Val: 2.6413 bei Step 100

Val:

| Step | val_loss | english | german |
|---:|---:|---:|---:|
| 50 | 2.766 | 4.065 | 2.194 |
| 100 | 2.641 | 4.075 | 2.600 |

Interpretation: Gesamt-Val wurde besser, aber Englisch blieb schwer und Deutsch schwankte stark. Der Lauf ist technisch stabil, aber das Val-Signal ist wegen gemischter Quellen nur begrenzt aussagekraeftig.

## Probe-Ergebnis

Vergleich gegen bekannte Zwischenstaende:

| Modell/Testpunkt | V4/V5 Gate | Clean v2 | Disjoint v1 |
|---|---:|---:|---:|
| v6 expanded best, 100 steps | 52.2% | 47.5% | 22.7% |
| books v1 lowratio, 25% books | 49.1% | 43.1% | 22.5% |
| books clean v2 lowratio, 20% books | 42.7% | 36.9% | 22.5% |

Clean-v2 lowratio Details:

- V4/V5 Gate: 42.7% ueber 11 Probes
- Clean v2: 36.9% ueber 16 Probes
- Disjoint v1: 22.5% ueber 12 Probes

Wichtige Fehlerbilder aus den Probe-Antworten:

- Hallucination-Guard bleibt kritisch: Goethe/Mein Kampf wurde falsch bejaht bzw. mit erfundenem Inhalt beantwortet.
- Fakten bleiben instabil: Bonn wurde als heutige Hauptstadt beantwortet; Wasser wurde als chemisches Element beschrieben.
- Wiederholung bleibt sichtbar: einzelne Antworten laufen in Satzschleifen.
- Code bleibt schwach, trotz etwas disjunktem Code-Signal.

## Fazit

Clean-v2 ist als Datenfilter ein Fortschritt, aber der Books-Mix ist nicht automatisch ein Modell-Fortschritt. Die sauberen Buecher liefern lange Prosa und altes Allgemeinwissen, aber sie loesen nicht die eigentlichen 500M-Probleme:

- zu wenig moderne, disjunkte deutsche Fakten-/QA-Daten
- zu wenig robuste Anti-Halluzination-Beispiele
- zu wenig ausfuehrbar getestete Code-Aufgaben
- Buchdaten sind im aktuellen Download extrem englischlastig
- viele Gutenberg-Texte sind alt, stilistisch weit weg von Instruktionsantworten

Aktuelle Empfehlung:

1. Clean-v2 behalten, aber nicht als grossen Hauptmix verwenden.
2. Books im echten v6 nur niedrig dosieren: eher 5-10%, maximal 15-20% nach Warmup.
3. Buecher eher fuer spaetere Knowledge-/Prose-Curriculum-Phase nutzen, nicht als Ersatz fuer QA/SFT-nahe Pretrain-Daten.
4. Jetzt wichtiger: source-disjunkte moderne Deutsch-QA, deutsche Instruktionen, Hallucination-Guard und ausfuehrbare Code-Evals/Daten.
5. Fuer deutsche Geschichte/Wissen gezieltere deutsche Quellen suchen statt nur Gutenberg-Cap: aktuelle/kuratierte Lexikon-, QA-, Schul-/Wissensdaten mit sauberer Lizenz.

Kurz: Der Filter ist gut. Die Buchdaten sind verwendbar. Aber sie sind nicht die Ursache des aktuellen Qualitaetsproblems und sollten den Expanded-Mix nicht dominieren.
