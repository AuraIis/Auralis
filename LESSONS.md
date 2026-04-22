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
