# HISTORY — Auralis v2

Chronologisch, Milestones. Append-Only.

---

## 2026-04-22 — Projekt-Start Auralis v2
- Brief + 7 Phasen-SPECs + `SPEC_DATASETS.md` in `Doc/` finalisiert
- Projekt-Skelett angelegt (`pyproject.toml`, Verzeichnisbaum, `.gitignore`)
- `STATUS.md`, `LESSONS.md`, `HISTORY.md` initialisiert
- Memory-System für Claude Code befüllt (User / Projekt / Feedback / v1-Lessons / v1-Datasets)
- **Entscheidung: Modellgröße 1B** (statt 2-3B) — schneller, günstiger, v1-Kapazität verdoppelt aber bleibt klein.
- **Entscheidung: Daten-Root `//BITBASTION/Auralis/AuralisV2/`** (NAS, 25 TB frei); v1-SFT-Pool `I:/Auralis/NEWGPT/data/` bleibt lokal.
- Phase-0-Vorarbeit komplett: Baseline (50 Fragen), byte-gleicher Prompt-Builder + Tests, Daten-Config + 3 Phase-1-Download-Scripts + v1-Inventory-Script.
- Leitlinie verankert: synthetische Daten-Generierung ist erwünscht, wenn Open-Source-Quellen dünn sind (DeepSeek V3 / Qwen 3.5 30B lokal — v1-erprobt).

## 2026-04-22 — Phase 0 abgeschlossen (Tokenizer + Phase-1-Daten)
- **v1-DE-Reuse:** 23.7 GB dedupliziertes deutsches Pretraining-Material auf NAS (~4.7 B Tokens, 8.87 M Docs), 9:35 Min.
- **EN-Downloads:** Wikipedia EN (12 GB), FineWeb-Edu sample-10BT (40 GB), OpenMathInstruct-2 (8 GB) — zusammen ~15 B Tokens.
- **Code-Downloads:** StarCoderData 9-Sprachen-Subset (3.5 GB) + open-web-math (0.88 GB) — zusammen ~1.25 B Tokens.
- **Nicht geladen** (Dataset-HTTP-404 oder `datasets` v4+ script-ban): SlimPajama, Dolma, Proof-Pile-2. Gesamt-Deckung Phase 1 = **~21 B / 25 B = 84 %**, Lücke für Phase 2 reserviert.
- **Tokenizer-Korpus:** 15.5 GB Mix (50 EN / 40 DE / 10 Code), NUL-bereinigt.
- **Tokenizer-Training (SentencePiece Unigram, 200 k Vocab, 32 Threads):** 14.6 Min.
- **Quality-Report PASS:** EN 123 tok/100 w (Ziel ≤135), DE 133.8 (≤150, v1 war ~220), Code 313.6 tok/KB (≤350), Unknown 0 %, **Chat-Template-Roundtrip byte-exakt**.
- **Neue Lessons L-007..L-012** in `LESSONS.md`: SP-Normalisierung `identity` zwingend, NUL-Strip-Pflicht, `num_threads ≥ 1`, `input_sentence_size = 5 M` bei 32-GB-RAM, HF v4 `Dataset-scripts` verbot, Code-Metrik auf `tokens/KB` umgestellt.

## 2026-04-23 — Phase 0.5 abgeschlossen (Modell-Architektur)
- `src/auralis/model/` komplett implementiert: config dataclass, RMSNorm, SwiGLU-FFN, Mamba-2 (pure torch), Gated Linear Attention (pure torch), Sparse Attention mit Sliding-Window + Global Tokens, RoPE, Scaled-Normal Init, KV-Cache-Dataclass.
- `helix_model.py`: `HelixBlock` pre-norm style, `HelixModel` mit heterogenem Stack aus Config, `build_model(yaml)` Factory, tied-embedding LM-Head.
- Zwei Configs: **`helix_v2_100m.yaml`** (8 Layers, 134 M Params, Test-Modell für CPU) und **`helix_v2_1b.yaml`** (28 Layers, d=1280, ~954 M Params — trifft 1B-Ziel innerhalb 5 %).
- **50/50 Tests grün** in 2.7 s auf CPU: Config-Loading + Validation, alle Layer (Shapes, Backward, Causal-Masking, Window-Masking, Global-Tokens-Bypass, RoPE-Norm-Preservation), End-to-End Forward/Backward auf 100M-Modell.
- Forward auf frisch-initialisiertem 100M-Modell liefert Loss **12.37** — nahe dem Theorie-Wert von `ln(200000) = 12.20` für uniform-prior über das 200k-Vocab. Keine NaN/Inf.
- Pure-Python-Varianten von Mamba-2 und GLA liefern die Referenz-Semantik für CPU-Tests. Für echtes GPU-Pretraining wird Phase 1 zusätzlich `mamba_ssm` und `flash-linear-attention` einbinden (Config-Flag, Interface bleibt gleich).
