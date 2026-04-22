# LESSONS — Auralis v2

Append-Only. Beste Erkenntnisse aus v1 (Tag 0) und jeder neuen Erfahrung in v2.

---

## Aus Auralis v1 übernommen (2026-04-22)

### L-001 — Prompt-Format-Konsistenz ist kritisch
Ein einziger Unterschied zwischen Training- und Inference-Prompt (`<|user|>` vs. `User:\n`) hat in v1
wochenlang die Inference-Qualität verschleiert.
**Regel v2:** EIN Prompt-Builder für Training + Inference + Eval + API. Byte-weiser Test Pflicht.

### L-002 — LoRA lernt Patterns, nicht zwingend Fakten
Blutdruck-LoRA v1 erreichte bei 212 Samples Loss 0.0099 (reine Memorization), neue Fragen
scheiterten trotzdem.
**Regel v2:** MoRA für Fakten, DoRA für Patterns. Val-Split mit **disjunkten** Fakten.
Early-Stopping bei Val-Loss 0.2–0.3. Min. 800–1500 Samples pro Topic.

### L-003 — Tokenizer ist nicht nachträglich tauschbar
GPT-2-Tokenizer war für Deutsch ~50 % zu ineffizient — aber jedes trainierte Gewicht hing dran.
**Regel v2:** Eigener 200 k SentencePiece. Einmal richtig, dann nie wieder anfassen.

### L-004 — Daten-Mix vor dem Training prüfen
german-commons Cultural Subset hat v1 Richtung historisches Deutsch verzerrt.
**Regel v2:** Bewusste Mix-Ratios im Config, Stichproben-Reviews vor jedem Run.

### L-005 — Baseline-Tests ab Tag 1
Ohne feste Eval-Fragen keine ehrliche Progress-Messung.
**Regel v2:** 50 Baseline-Fragen in `eval/baseline_questions.yaml` committed, automatisch bei
jedem Checkpoint laufend.

### L-006 — Optimizer-State bewusst behandeln
Drei Modellversionen (v20, v28, v30) in v1 durch vergessene `--reset-optimizer` verloren.
**Regel v2:** `--reset-optimizer` ist **Default** bei SFT / Continued Pretrain, nicht Ausnahme.

---

## Neue Lessons aus Phase 0 (2026-04-22)

### L-007 — SentencePiece `normalization_rule_name: nmt_nfkc` normalisiert Newlines zu Space
Beim ersten SP-Training sind alle `\n` im Chat-Template zu `" "` geworden → byte-exakter Roundtrip gescheitert (exakt der L-001-Bug-Typ aus v1).
**Regel v2:** `normalization_rule_name: identity` + `byte_fallback: true`. Dann wird jeder Byte, inklusive `\n` (`<0x0A>`), erhaltend kodiert und dekodiert. `quality_report.py` prüft den Roundtrip byte-exakt — **Pflichttest vor jedem Tokenizer-Commit**.

### L-008 — StarCoderData enthält NUL-Bytes (und andere C0-Controls)
Rohe Code-Files (Binärdateien, Editor-Artefakte, generated code) haben verstreute `\x00`-Bytes. SentencePiece-Training emittiert für jedes NUL eine Warnung und kann bei genug NUL-Bytes crashen.
**Regel v2:** Bytes unter `0x20` (außer `\t` `\n` `\r`) vor dem Tokenizer-Training aus dem Korpus strippen — Schritt läuft automatisch nach `prepare_corpus.py`.

### L-009 — SentencePiece `num_threads=0` wird nicht als „alle Kerne" interpretiert
SP verlangt `1 ≤ num_threads ≤ 1024` — ein 0 führt zum sofortigen Abort.
**Regel v2:** im Training-Script `args.num_threads or max(1, os.cpu_count())` — CLI-Default 0 bedeutet auto.

### L-010 — 15 GB Korpus × 10 M Sentences × 200 k Vocab sprengt 32 GB RAM
EM-Training blähte beim ersten Versuch die RAM-Belegung auf bis zum OOM-Kill (exit 127).
**Regel v2:** `input_sentence_size = 5_000_000` (bei 32-64 GB RAM). Erhöhen nur wenn explizit auf 128-GB+-Pod trainiert wird.

### L-011 — HuggingFace `datasets` v4+ blockt Script-basierte Loader
SlimPajama (`cerebras/SlimPajama-627B`), Dolma (`allenai/dolma`), Proof-Pile-2 (`EleutherAI/proof-pile-2`) sind alle nicht mehr ladbar: `Dataset scripts are no longer supported, but found *.py`.
**Regel v2:** Vor Aufnahme einer neuen Quelle in `download_*.py` prüfen, ob sie **parquet-only** verfügbar ist (z.B. `open-web-math/open-web-math` statt `proof-pile-2`). Im Zweifel: `datasets.load_dataset(name, streaming=True)` lokal smoke-testen, bevor das Multi-GB-Download gestartet wird.

### L-012 — „Tokens pro 100 Wörter" ist für Code-Pretraining keine gute Metrik
Code-Zeilen bestehen oft aus 2-3 „Wörtern" (`return x;`), aber aus vielen Tokens. Das /100-Words-Ziel trieb nach oben, obwohl die Kompression gut war.
**Regel v2:** Code-Gate auf `tokens_per_kb` umgestellt (Target ≤350 tokens/KB ≈ ≥2.9 bytes/token). EN/DE bleiben auf /100-Words (dort ist die Metrik stabil).
