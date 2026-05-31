# Pretrain v6 Gutenberg / Books Test - 2026-05-27

## Ziel

Mehr Wissens- und Geschichtsprosa in den v6-Datenpool bringen. Der Nutzer wollte explizit Project-Gutenberg-/Buchdaten, auch wenn sie alt sind. Ziel war daher: kontrolliert laden, filtern, tokenisieren und testen, statt ungeprueft 70k Buecher in den Continue zu kippen.

## Quellen

| Quelle | Zweck | Ergebnis |
| --- | --- | ---: |
| `common-pile/project_gutenberg` | Project-Gutenberg-Spiegel, public-domain Metadaten | 332,643 Chunks |
| `zkeown/gutenberg-corpus` | Gutenberg-Corpus mit Buch-Metadaten | 2,309 Chunks |

Output:

- `data/training/pretrain_v6_book_sources_gutenberg_v1/book_sources.txt`
- `data/training/pretrain_v6_book_sources_gutenberg_v1/book_sources.jsonl`
- `data/training/pretrain_v6_book_sources_gutenberg_v1/manifest.json`
- `data/training/pretrain_v6_book_sources_gutenberg_v1/source_disjoint_manifest.jsonl`
- `tokenized/pretrain_v6_book_sources_gutenberg_v1/english.bin`

## Datenstand

- Chunks: 334,952
- Rohtext: 1.51 GB
- echte Tokens mit Helix-v2-Tokenizer: 349,622,011
- Holdout: 1,374 Chunks
- Train: 333,578 Chunks
- Gate-Kontamination: 0 Prompt-Hits
- Common-Pile Sprachverteilung im ersten Cap: 4,226 EN / 48 DE Werke
- zkeown Sprachverteilung im ersten Cap: 40 EN / 1 DE Werke

Das ist also ein starker English-/Book-Knowledge-Block, aber kein deutscher Buchblock. Fuer deutsche Geschichte/Prosa brauchen wir gezieltere deutsche Quellen oder German Commons kulturell/historisch.

## Trainings-Smokes

### A: 65% Books / 35% German Expanded, Init von v5-best

- Config: `configs/training/pretrain_v6_books_augmented_500m_from_v5_best_bitbastion.yaml`
- Status: completed
- Steps: 100
- Val @ 50: 3.706
- Val @ 100: 3.553
- English Val: 4.005 -> 4.009
- German Val: 2.666 -> 2.703
- Peak: ca. 16.7k tok/s
- Ergebnis: stabil, aber Probes fallen deutlich.

### B: 25% Books / 75% German Expanded, Init von expanded-test-best

- Config: `configs/training/pretrain_v6_books_lowratio_500m_from_expanded_best_bitbastion.yaml`
- Status: completed
- Steps: 100
- Val @ 50: 2.792, best
- Val @ 100: 2.889
- English Val: 4.097 -> 3.939
- German Val: 2.576 -> 2.386
- Peak: ca. 17.5k tok/s
- Ergebnis: stabiler; Deutsch-Val verbessert, Gesamt-Best bleibt bei Step 50 wegen schwerem English-Book-Anteil.

## Probe-Vergleich

| Checkpoint | V4/V5 Gate | Clean v2 | Disjoint v1 |
| --- | ---: | ---: | ---: |
| v5 best step 14,500, frueher | 28.2% | 27.7% | 11.7% |
| v6 expanded 100-step best | 52.2% | 47.5% | 22.7% |
| Books 65% best | 37.3% | 25.0% | 6.0% |
| Books 25% best | 49.1% | 43.1% | 22.5% |

## Bewertung

Die Gutenberg-Daten sind technisch sauber nutzbar und liefern sehr viele Tokens. Als dominanter Kurz-Continue schaden sie aber den aktuellen Capability-Probes, weil das Modell in Richtung altenglische Buchprosa gezogen wird. Mit 25% Anteil bleibt es deutlich stabiler, aber es verbessert die aktuellen Gates noch nicht gegen den expanded-test-best.

## Empfehlung

- Gutenberg/Open-Books behalten, aber im echten v6-Mix nur niedrig dosieren: 10-20%.
- Nicht direkt ab v5-best mit hohem Buchanteil trainieren; erst den besseren expanded-Mix nutzen, dann Book-Anteil langsam dazumischen.
- Fuer deutsches Geschichts-/Wissensprofil gezielt deutsche Buch-/Commons-Quellen suchen, weil der aktuelle Gutenberg-Spiegel im ersten Cap fast nur Englisch liefert.
- Fuer einen laengeren Continue erst ein Curriculum bauen: moderne Deutsch/QA/Code stabilisieren, dann Books als Knowledge-Prosa-Block einstreuen.
