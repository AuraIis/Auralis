# Codex Review Brief — Helix v2 "corpus20b" Data Pipeline Audit

**Purpose:** Independently verify that the freshly-built ~24B-token training corpus
(`corpus20b`) and its data pipeline are **correct and not corrupted**. This corpus is the
fix for a previously discovered *triple data-corruption* in Helix's code pretraining (see
§3). We already ran checks (all passed) but want a second, independent pass — including
scrutiny of the weak spots we list in §8. **Be adversarial. Assume nothing. Decode and read
the real data, don't trust token counts.**

---

## 1. What this is
Helix v2 / Auralis is a from-scratch ~0.9B German-primary hybrid LLM (Mamba-2 / GLA /
Sparse-Attn). We are assembling a clean, code-heavy ~24B-token corpus to re-train the base
(warm-start from the foundation checkpoint). The model writes good German but failed at code;
we traced that to corrupted code *data*, not model size. This corpus is the remedy.

## 2. System topology & how to access the data
Two machines, one SMB share:

| Resource | Local Windows path (read via SMB) | Container path (auralis-blackwell) |
|---|---|---|
| Tokenized bins | `U:\NEWGPT\v2data\tokenized\corpus20b\` | `/workspace/v2data/tokenized/corpus20b/` |
| Old/ref bins | `U:\NEWGPT\v2data\tokenized\curated_40b\` | `/workspace/v2data/tokenized/curated_40b/` |
| Tokenizer | `U:\NEWGPT\v2data\tokenizer\helix_v2_tokenizer.model` | `/workspace/v2data/tokenizer/...` |
| JSONL sources | `U:\NEWGPT\v2data\data\fresh\` | `/workspace/v2data/data/fresh/` |
| Cleaned text | `U:\AuralisV2\cleaned\` | `/workspace/v2data/cleaned/` |
| Scripts | `U:\AuralisV2\scripts\` | `/workspace/v2data/scripts/` |
| Config (mix) | `U:\AuralisV2\configs\training\corpus20b_codeheavy.yaml` | same under `/workspace/v2data/` |
| Dataloader | `U:\AuralisV2\src\auralis\training\dataset.py` | same |
| SFT seed | `U:\AuralisV2\raw\sft\os_security_de_seed.jsonl` (+ `.verified.jsonl`) | same |

- `U:` = `\\192.168.178.5\Auralis` = unraid `/mnt/user/Auralis`.
- **Local Python env with deps:** `C:\auralis_venv\Scripts\python.exe` (torch 2.6+cu124,
  transformers 4.46.3, sentencepiece, numpy). Use this for decode/inspection locally.
- **SSH to the host:** `ssh bitbastion` → then `docker exec auralis-blackwell <cmd>`. The
  container has python+numpy+sentencepiece and the GPU.
- **Tokenizer facts:** SentencePiece, vocab 200k, **eos_id = 3**.
- **Bin format:** `.bin` = flat `uint32` token stream. `.idx` = flat `int64` pairs
  `[offset_tokens, n_tokens]` per document (written by `tokenize_anneal.py`).

> **Symlink note:** in `corpus20b/`, `en.bin` and `de_curated.bin` are **Linux symlinks** to
> `../curated_40b/{english,german}.bin`. Verified: they **do resolve over Windows SMB** here
> (read as 43.9 GB / 12.1 GB). If your environment differs and they don't resolve, read the real
> targets in `curated_40b\`. The other 8 bins are real files.

## 3. The three historical bugs (what to watch for)
These corrupted the OLD `curated_40b/code.bin` and are the reason for this rebuild:
1. **Docstring/quote stripping** — a prose cleaner (built for DE/EN) removed lines containing
   `"""`, leaving unbalanced docstrings → ~74% of code no longer parsed as Python.
2. **Python-2 code mixed in** — py2 syntax (print statements) trained wrong syntax.
3. **Line-per-doc tokenization** — `tokenize_for_pretraining.py` treated every code *line* as a
   separate "document" and appended `</s>` after each → code atomized into ~30-token fragments
   (old `code.bin`: 23.6M docs / 709M tok = ~30 tok/doc). The model never saw whole files.

The fix: code re-derived from RAW starcoder via `filter_code_quality.py` (ast.parse-py3 gate,
language-aware), and everything tokenized with `tokenize_anneal.py` (EOS per *document*,
newlines preserved). **Confirm these bugs are absent in `corpus20b`.**

## 4. Data inventory — `corpus20b/` bins (what to verify)
| Bin | Source | Docs | Tokens | tok/doc | Notes |
|---|---|---|---|---|---|
| `code.bin` | raw starcoder, ast-py3-filtered, multi-lang | 844,170 | 1,147,216,968 | 1359 | from `filter_code_quality.py` |
| `code_algo.bin` | OpenCoder opc-annealing algorithmic_corpus | 5,322,799 | 1,460,443,215 | 274 | |
| `code_snip.bin` | OpenCoder synthetic_code_snippet | 2,818,342 | 1,601,320,846 | 568 | |
| `code_qa.bin` | OpenCoder synthetic_qa | 2,376,842 | 1,586,150,707 | 667 | |
| `math.bin` | openmath (cleaned) | 7,285,333 | 2,161,245,405 | 297 | EN math QA |
| `german_commons.bin` | german_commons.edu (edu-filtered) | 2,127,505 | 1,362,325,634 | 640 | OCR-noisy public-domain books |
| `german_fresh.bin` | fresh FineWeb2-deu_Latn, 2-GPU edu-filtered | 7,412,611 | 4,295,893,465 | 580 | NEW download |
| `stackexchange.bin` | common-pile/stackexchange (OS sites) | 2,819,982 | 2,384,454,424 | 846 | askubuntu/superuser/serverfault/unix + SO(capped) |
| `en.bin` (symlink) | curated_40b/english.bin (FineWeb-edu) | — | 11,792,854,409 | — | reused, used as subset via ratio |
| `de_curated.bin` (symlink) | curated_40b/german.bin | — | 3,236,521,471 | — | reused |

Code total = **5.79B**; new tokens this run = **11.33B**.

## 5. Pipeline / scripts (review the logic, not just outputs)
All under `U:\AuralisV2\scripts\`:
- `data/stream_hf_text.py` — HF streaming download. Two modes: prose (collapse newlines, one
  line/doc) and `--jsonl` (preserve newlines, for code). Used for FineWeb2-DE + OpenCoder.
- `data/filter_code_quality.py` — **multilingual** starcoder filter. Python → hard `ast.parse`
  (py3) gate; other langs → heuristics (min lines, comment ratio, ascii, not-minified,
  not-autogenerated, keyword density). Per-language scoring 0–5, keep ≥ threshold.
- `data/blocks_to_jsonl.py` — converts starcoder `<|code|>…<|endcode|>` blocks → JSONL (one
  record/file, newlines preserved); also a `--mode lines` for already-line-per-doc text.
- `data/stream_stackexchange.py` — targeted streamer for common-pile/stackexchange with
  per-site token caps (OS sites uncapped-ish, stackoverflow capped at 0.5B).
- `data/tokenize_anneal.py` — JSONL → bin/idx, **EOS per document**, `--max-len 8192`. *This is
  the correct tokenizer.* (Contrast: `tokenize_for_pretraining.py` is the line-per-doc one that
  caused bug #3 — do NOT use it for code.)
- `data/score_corpus_edu.py` + `data/edu_embed.py` — German edu-quality filter (frozen
  `intfloat/multilingual-e5-large` embedder + a trained linear head `eval/results/de_edu/edu_clf.pt`,
  threshold 2.4). Kept ~78–80% of fresh FineWeb2-DE.
- `sft/build_os_security_sft.py` — merges/normalizes the 5 German OS/security seed files.
- `sft/verify_seed_code.py` — executor-verifies the Python rows of the SFT seed.
- `sft/verify_seed_nonpy.py` — syntax-checks non-Python seed rows (we flagged this as unreliable).

## 6. The mix manifest — `configs/training/corpus20b_codeheavy.yaml`
- `data.data_dir: /workspace/v2data/tokenized/corpus20b`, `mix_ratios` (10 keys, sum→1.0,
  loader normalizes). Effective proportions: DE ~35% / EN 24% / Code 23% / Math 9% / SE 9%.
- `init_from: foundation/step_50000.pt` (warm-start, continued-pretrain, NOT from scratch).
- Loader = `MixedDataLoader` in `src/auralis/training/dataset.py`: each `mix_ratios` key K loads
  `K.bin`; reserves the **last `val_split_bytes` (2 MB)** of each bin for validation; samples
  random `seq_length` windows from the flat stream (doc boundaries ignored at train time).

## 7. Checks to run (please do all; commands are starting points)
Run locally with `C:\auralis_venv\Scripts\python.exe`, or on the container via SSH.

1. **EOS at doc boundaries (anti-bug-#3):** for each new bin, for many docs, the token at
   `idx[d]` span end must be `eos_id (3)`, and there must be **0** EOS *inside* the span.
   *(We got 0/0 over 30k docs — reproduce independently.)*
2. **Decode & read content:** decode random docs from each bin; confirm code is whole/multi-line
   with intact docstrings/comments, German is readable, math has problem+solution. Look for
   garbling, mojibake, repeated boilerplate, truncation artifacts.
3. **Python validity of code bins:** sample many Python docs across `code*`; `ast.parse` them →
   what fraction parse? (Old bug = mass parse-fail.) Note non-Python langs in `code.bin`.
4. **tok/doc sanity:** confirm hundreds of tok/doc (not ~30). Flag any bin near ~30.
5. **idx ↔ bin consistency:** `sum(idx[:,1]) == len(bin)` (tokens accounted for); offsets
   contiguous and monotonic.
6. **Mix ratios:** load the YAML; confirm keys map to existing bins, ratios sum≈1, and
   `MixedDataLoader` constructs and yields a batch without error.
7. **Symlinks:** confirm `en.bin`/`de_curated.bin` resolve to the curated bins (on container).
8. **SFT seed:** re-run `verify_seed_code.py`; spot-read `os_security_de_seed.jsonl`; sanity-check
   the security/agent-trace content is accurate and defensively framed.

## 8. Weak spots WE did NOT fully verify — please scrutinize these
Be skeptical here; these are the most likely real problems:
- **De-duplication / train-val & cross-source overlap (HIGH PRIORITY):** `german_fresh` (fresh
  FineWeb2-DE) and `de_curated` (older FineWeb2-DE) share the *same upstream dataset* (different
  crawls) → likely **duplicate or near-duplicate documents**. We did NOT dedup across them. This
  risks over-repetition and, worse, **train/val leakage** (the val split is the last 2 MB of each
  bin; if a fresh-DE doc also sits in de_curated's train region, val is contaminated). Quantify
  the overlap (e.g. hash/MinHash a sample of docs across the two DE bins, and across
  `german_fresh` vs the foundation's held-out DE val set if available).
- **Truncation at max-len 8192:** `tokenize_anneal.py` truncates any doc >8192 tokens. How many
  docs were truncated, and how many tokens lost? (Esp. `stackexchange`, long Q&A threads.)
- **EOS-window train/val split correctness:** confirm `val_split_bytes` actually holds out
  complete docs and that windows can't straddle the train/val boundary.
- **`en.bin`/`de_curated.bin` reuse:** these were the foundation's *training* data. If we later
  evaluate on a held-out set derived from them, that's leakage. Confirm the eval/held-out sets
  are disjoint from these bins.
- **Non-Python SFT code unverifiable:** `verify_seed_nonpy.py` is unreliable (no TS parser, prose
  interleaving). Treat its numbers as noise; if you have node/tsc/shellcheck, do better.
- **Math is English-only** (`openmath`); the model is German-primary — is that the intended mix?
- **`stackexchange` is CC BY-SA** (share-alike) — flag if license matters for the artifact.

## 9. How to report
Please return: (a) per-check PASS/FAIL with evidence, (b) any corruption found (with the exact
bin + doc index + decoded snippet), (c) the dedup/overlap quantification from §8, (d) anything we
missed. If something is broken, we re-tokenize that source — so pinpoint the source + script.
