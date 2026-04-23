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

## 2026-04-23 — Phase 1 Pretraining-Pipeline fertig (Launch-ready)
- `scripts/data/tokenize_for_pretraining.py`: batched SentencePiece-Encoding → uint32 .bin + int64 .idx pro Sprache, atomare Writes + Manifest, Resume-safe.
- `src/auralis/training/` komplett: `PretrainDataset` (memmap), `MixedDataLoader` mit largest-remainder-Partitionierung für Mix-Ratios, `build_optimizer` mit decay-Split für Normen/Biases, `build_scheduler` (cosine + constant_with_warmup), `PretrainTrainer` mit Gradient-Accumulation, Grad-Clip, Checkpoint-Rotation, NaN-Abbruch, Val-Loss-Alarm nach 3 Regressions.
- `scripts/pretrain/train_phase1.py`: CLI-Entry mit Preflight + Resume + Device-Override + `torch.compile`-Flag.
- `scripts/pretrain/smoke_test.py`: End-to-End-Validation in 30 s (134M-Modell, 20 Steps, synthetische Tokens, Checkpoint-Roundtrip).
- `configs/training/phase1_pretrain.yaml`: 80 k Steps × 128 effective Batch × 2048 Tokens = ~21 B Tokens (matched zur tatsächlichen Daten-Deckung).
- **64/64 Tests grün in 4 s** (+14 neue Training-Tests: `dataset`, `optimizer`, `trainer`).
- Tokenization der 88 GB Phase-1-Daten startete parallel (Background) — Durchsatz ~6 MB/s auf SMB, geschätzte Dauer ~4 h.
- **Launch-Guide** `docs/PHASE_1_LAUNCH.md` deckt RunPod-Setup, Preflight, Monitoring, Rollback-Prozeduren und Milestone-Erwartungen ab.
- **GPU-Launch selbst wurde NICHT automatisch gestartet** — das kostet $500-800 auf RunPod und braucht Michaels explizite Entscheidung + Account-Setup.

## 2026-04-23 — Blackwell-GPU-Validation auf Unraid
- Auralis-Docker-Container (Ubuntu 22.04, Python 3.11, torch 2.7.0+cu128) auf Unraid-Host mit RTX PRO 5000 Blackwell (47 GB VRAM) benutzt.
- V2-Datenverzeichnis via `mount --bind /mnt/user/Auralis/AuralisV2 /mnt/user/Auralis/NEWGPT/v2data` + `docker restart` in den Container eingehängt (SHFS propagiert laufende Mounts nicht).
- Libraries installiert: `flash-linear-attention`, `mamba-ssm 2.3.1`, `causal-conv1d 1.6.1`, `flash-attn 2.8.3` (alle cu128-kompatibel). Nach Triton 3.6-Upgrade kompilieren beide Triton-basierte Libraries (mamba-ssm + fla).
- **Library-Swap-Hooks eingebaut**: Pro Layer-Typ separat aktivierbar per Env-Var (`AURALIS_USE_CUDA_KERNELS`, `AURALIS_USE_MAMBA_KERNEL`, `AURALIS_USE_GLA_KERNEL`, `AURALIS_USE_FLASH_ATTN`). Interface für `HelixBlock` unverändert; default bleibt native pure-torch. GLA-Backend unterstützt gleiche Parameter-Shapes native/fla, Mamba-Backend nicht (Architektur-Unterschied).
- **Smoke-Test-Resultate (250M, bf16, batch=4):**
  - seq=256 native: 147 tok/s, 13.0 GB VRAM, Loss 12.16 → 11.59 (Δ+0.57, lernt)
  - seq=512 native: 82 tok/s, 24.85 GB VRAM
  - seq=512 gla-kernel: 88 tok/s, **21.27 GB VRAM (-14 %)**, numerisch identisch zu native
  - seq=512 gla+flash: identische Resultate (Sparse macht nur 3/12 Layer)
- **Blackwell-Erkenntnis:** Hauptvorteil der Kernels ist **VRAM-Ersparnis** (chunkwise statt materialisiertem [B,L,H,D,D] State), nicht primär tok/s. tok/s-Speedup wird bei seq=2048 Phase-1-Config deutlicher.
- **Mamba-Kernel auf Blackwell aktuell problematisch** — Triton compile bug in mamba_ssm selbst mit Triton 3.6. Für RunPod H100/H200 ok, für Blackwell Mamba native lassen.
- **(Später am selben Tag) Blackwell-Fix gefunden: `TRITON_OVERRIDE_ARCH=sm89`** — emuliert Ada (Compute-Capability 8.9) auf Blackwell (sm_120). Weder sm90 (WGMMA-Intrinsics fehlen) noch default (sm_120 unbekannt) funktionieren; sm89 ist das erste abwärtskompatible Target das Blackwell akzeptiert. Alle drei Kernels (mamba_ssm + fla + flash-attn) laufen jetzt gleichzeitig, numerisch identisch zur native-Referenz.
- **Final-Messungen Blackwell, 250M bf16, ALLE Kernels aktiv:**
  - seq=256 batch=4: 220 tok/s, 6.68 GB VRAM (-49 % vs native)
  - seq=512 batch=8: 1 928 tok/s, 16.36 GB (23× vs 3090 pure-python)
  - seq=1024 batch=4: 3 628 tok/s, 16.84 GB
  - **seq=2048 batch=2: 2 713 tok/s, 17.74 GB — Phase-1-Config validiert, Loss Δ +1.52 in 15 Steps**
- Docs (`docs/PHASE_1_LAUNCH.md`) um sm89-Workaround und per-Hardware Kernel-Setup erweitert.
