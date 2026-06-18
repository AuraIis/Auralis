# LESSONS — Auralis v2

Append-only. Best insights from v1 (day 0) and every new experience in v2.

---

## Carried over from Auralis v1 (2026-04-22)

### L-001 — Prompt-format consistency is critical
A single difference between training and inference prompt (`<|user|>` vs. `User:\n`) obscured the inference quality in v1
for weeks.
**Rule v2:** ONE prompt builder for training + inference + eval + API. Byte-wise test mandatory.

### L-002 — LoRA learns patterns, not necessarily facts
The blood-pressure LoRA v1 reached loss 0.0099 on 212 samples (pure memorization), yet new questions
still failed.
**Rule v2:** MoRA for facts, DoRA for patterns. Val split with **disjoint** facts.
Early stopping at val loss 0.2–0.3. Min. 800–1500 samples per topic.

### L-003 — Tokenizer is not swappable after the fact
The GPT-2 tokenizer was ~50% too inefficient for German — but every trained weight hung on it.
**Rule v2:** Own 200k SentencePiece. Once correctly, then never touch again.

### L-004 — Check the data mix before training
The german-commons cultural subset skewed v1 toward historical German.
**Rule v2:** Deliberate mix ratios in the config, sample reviews before every run.

### L-005 — Baseline tests from day 1
Without fixed eval questions there is no honest progress measurement.
**Rule v2:** 50 baseline questions committed in `eval/baseline_questions.yaml`, running automatically at
every checkpoint.

### L-006 — Handle optimizer state deliberately
Three model versions (v20, v28, v30) lost in v1 due to a forgotten `--reset-optimizer`.
**Rule v2:** `--reset-optimizer` is the **default** for SFT / continued pretrain, not the exception.

---

## New lessons from Phase 0 (2026-04-22)

### L-007 — SentencePiece `normalization_rule_name: nmt_nfkc` normalizes newlines to space
On the first SP training all `\n` in the chat template became `" "` → byte-exact roundtrip failed (exactly the L-001 bug type from v1).
**Rule v2:** `normalization_rule_name: identity` + `byte_fallback: true`. Then every byte, including `\n` (`<0x0A>`), is encoded and decoded losslessly. `quality_report.py` checks the roundtrip byte-exact — **mandatory test before every tokenizer commit**.

### L-008 — StarCoderData contains NUL bytes (and other C0 controls)
Raw code files (binary files, editor artifacts, generated code) have scattered `\x00` bytes. SentencePiece training emits a warning for every NUL and can crash with enough NUL bytes.
**Rule v2:** Strip bytes below `0x20` (except `\t` `\n` `\r`) from the corpus before tokenizer training — step runs automatically after `prepare_corpus.py`.

### L-009 — SentencePiece `num_threads=0` is not interpreted as "all cores"
SP requires `1 ≤ num_threads ≤ 1024` — a 0 leads to an immediate abort.
**Rule v2:** in the training script `args.num_threads or max(1, os.cpu_count())` — CLI default 0 means auto.

### L-010 — 15 GB corpus × 10 M sentences × 200 k vocab blows past 32 GB RAM
EM training inflated RAM usage on the first attempt up to the OOM kill (exit 127).
**Rule v2:** `input_sentence_size = 5_000_000` (with 32-64 GB RAM). Increase only when explicitly training on a 128-GB+ pod.

### L-011 — HuggingFace `datasets` v4+ blocks script-based loaders
SlimPajama (`cerebras/SlimPajama-627B`), Dolma (`allenai/dolma`), Proof-Pile-2 (`EleutherAI/proof-pile-2`) are all no longer loadable: `Dataset scripts are no longer supported, but found *.py`.
**Rule v2:** Before adding a new source to `download_*.py`, check whether it is available **parquet-only** (e.g. `open-web-math/open-web-math` instead of `proof-pile-2`). When in doubt: smoke-test `datasets.load_dataset(name, streaming=True)` locally before starting the multi-GB download.

### L-012 — "Tokens per 100 words" is not a good metric for code pretraining
Code lines often consist of 2-3 "words" (`return x;`), but of many tokens. The /100-words target drifted upward even though compression was good.
**Rule v2:** Code gate switched to `tokens_per_kb` (target ≤350 tokens/KB ≈ ≥2.9 bytes/token). EN/DE stay on /100-words (the metric is stable there).

### L-013 - shuf on large training files kills the trainer's disk IO
During the v1 lessons audit on 2026-04-26, \`shuf -n 5 english.txt\` (56 GB) was run during an active training.
\`shuf\` must read the entire file for the reservoir-sampling guarantee - several parallel shuf processes
blocked the disk channel, trainer tok/s crashed from 33k to 2.4k, data_wait rose to 93%.

Plus: parent-bash loops respawned shuf calls after SIGKILL. Only after killing the parent bash was it over.

**Rule v2:**
- NEVER \`shuf\` on large training files while training is running
- For sample review: \`head -n 5\` (deterministic, sub-second), or \`sed -n 12345,12349p\` for mid-file
- For real random sampling: do it once offline beforehand with \`shuf > sample.txt\`, then read only that small sample
- In general: every bash command on files > 1 GB should be checked for disk load before running (\`iostat -xz 1 2\`)

### L-014 — Checkpoint rotation must tolerate non-step suffixes
Health guards can write emergency checkpoints `step_<n>_emergency.pt`. The rotation in `trainer.py` matched `step_*.pt` and parsed the name position-based (`int(name.split("_")[1])`) → `ValueError: invalid literal for int() with base 10: '10_emergency'`. Crash exactly in the auto-stop path right after the emergency ckpt was written — the run to be rescued died during cleanup.
**Rule v2:** Always extract the step from checkpoint filenames regex-based (`r"step_(\d+)"`), never split position-based. The auto-stop path must be idempotent and must not die from secondary errors out of its own cleanup.

### L-015 — Cross-cutting modules must enumerate ALL trigger layers
The shared `RotaryEmbedding` was only instantiated in `helix_model.py` when at least one `sparse_attention` layer was in the stack. Architectures purely from `plain_attention` with `use_rope=true` got `rope=None` passed through — position encoding silently missing, since a PlainAttentionLayer with `rope=None` simply computed without rotation. The existing tests only checked shape and causality, the bug would only have surfaced at eval (long-context regression).
**Rule v2:** When a shared module can be needed by ≥2 layer types, the build condition must check ALL relevant layer configs (`any(needs_rope(l) for l in layers)`), not just one trigger type. Plus: for every sub-architecture a smoke test with a numerically sensitive assertion (e.g. that permuting the position IDs changes the result) — not just shape/causality.

### L-016 — `pgrep -f` in a wait wrapper matches the wrapper itself
A chained training wrapper `bash -c 'while pgrep -f "train_phase.*runde3"; do sleep 30; done; python sweep.py ...'` survived the trainer but never started the sweep. Reason: the python argument in the wrapper bash contained both strings, so `pgrep -f` matched the wrapper bash itself → wait loop never terminable.
**Rule v2:** In wait wrappers either
- make the pattern unmatchable for its own command line — trick: hide the first letters in a char class, e.g. `pgrep -f "[t]rain_phase.*runde3"` (matches the original process, not the pgrep argument itself because `[t]` as a regex class ≠ literal `[t]`).
- Or wait PID-based: capture the trainer PID before the `wait` loop and use `kill -0 $PID` as a liveness check — no string matching.
- In general: syntax-check wrapper scripts with `bash -n` before a detached start AND dry-run with a short dummy trainer.

### L-017 — Helpful-elaboration trap in honest_refusal SFT generation
During Phase-3 SFT data generation with DeepSeek-V4-Flash via OpenRouter: a generic "You are honest, do not hallucinate" system prompt still reached a ~3% hallucination rate on historical false-premise prompts. Concrete example: for "Who designed the office chair in Goethe's study?" the model produced confident fabrications in 2 of 9 samples — once "Johann Friedrich Funk (1706-1775)", once "Friedrich Justin Bertuch had it made in 1794...". Both sounded plausible, both were invented.

Root cause: the model tries to provide context out of politeness and in doing so confabulates specific details (names, dates, years). A pure prohibition is not enough — the model does not know *which* details it must not give.

**Rule v2 — anti-hallucination system prompt for SFT generation:**
- ❌ NOT sufficient: *"Never hallucinate. Say openly when you don't know."*
- ✅ SUFFICIENT with three components:
  1. **Explicitly forbidden speculation markers** ("presumably", "probably", "supposedly", "allegedly", "likely", "may have been")
  2. **Few-shot examples for GOOD vs BAD refusals** (show the model concretely what you want)
  3. **Allowed: verifiable context debunk** (e.g. "Goethe did not attend a classical Gymnasium..." — verifiable, helps frame the question) vs. forbidden: alternative specific details ("the office chair was presumably designed by X in year Y...")
- A/B test: 0% hallucination rate on 310 test records (vs ~3% baseline), avg out-tokens 143 instead of 241 (more concise due to forbidden filler waffle).
- Additionally: the refusal auto-detection regex must be BROAD — "Ich weiß **es** nicht" does not match "weiß nicht" (non-contiguous words), regex on individual keywords ("weiß", "unbekannt", "nicht überliefert") is more robust.

---

## New lessons from the edu filter + Multi-GPU (2026-05-31)

### L-018 — Thinking models: `max_tokens` covers reasoning AND answer (cost trap)
gemini-3.5-flash as 0-5 edu judge: with `max_tokens=200` only ~6 visible tokens arrived — the ~190 thinking tokens eat the same budget, the `Bewertung:` line is cut off (25/25 unparsed). Worse: thinking tokens are billed as **expensive output** → a supposedly "cheap Flash" cost **€24** for ~12k annotations.
**Rule v2:** With thinking models set `max_tokens` generously (≥512) **and** throttle `reasoning_effort`. For pure classification/rating tasks (0-5) choose a **non-thinking** model — faster, predictable, ~10-50× cheaper. Check the token budget BEFORE the full run on a 25-doc smoke (count succeeded vs parsed separately).

### L-019 — Judge choice for data filtering: cheap ≠ worse, strict ≠ wrong
Switch gemini-3.5-flash → `qwen3-235b-a22b-2507` (non-thinking, OpenRouter): ~40× cheaper AND the **better** judge. Qwen correctly rated web spam/reviews/EuroParl fragments 0-1 where Gemini generously gave 3 — exactly this leniency produced the over-keep of the Gemini-trained classifier on german_commons (64% instead of 45%). FineWeb-Edu itself used Llama-3-70B (non-thinking), no frontier thinking model.
**Rule v2:** For edu/quality rating, a solid dense/MoE instruct model (Gemma-3-27B, Qwen3-235B-2507, Llama-3.x-70B) instead of expensive frontier thinking. Keep ONE judge consistent (no judge mix in the training set). Validate the judge on concrete raw rationales, not just on the score distribution.

### L-020 — german-commons is OCR-historical-dominated (reinforces L-004)
On the attempt to re-pull german-commons as a scaling source: the HF stream is **front-loaded with digitized historical books** (sources `BLBooks`, `DiBiLit`, `DiBiPhil`, `GermanPD`; perplexity 500-1000+; `subset` field useless = `'0'`). In 8000 streamed docs **not a single** modern one (<200 ppl). The "72B News / 54B Cultural" of the dataset card are largely OCR archives (wrong register, Fraktur errors). Our old `max_perplexity=500` + `cultural_keep_ratio=0.05` filter (correctly) kept this out — we ended up at the clean but educationally thin parliamentary layer.
**Rule v2:** german-commons is **not** a modern-German scaling gain. For more high-quality modern German: RedPajama-V2-de (3T, with quality signals) + more fineweb2_de, both edu-filtered. For every new streaming source, first sample the subset/ppl distribution of the first N docs before streaming on a token budget.

### L-021 — Ridge regressor shrinks toward the mean → calibrate the decision threshold
The edu classifier (Ridge on e5 embeddings) predicts scores shrunk toward the mean. A hard threshold at 3.0 yielded precision 0.99 / **recall 0.66** (threw away ~1/3 of real ≥3 docs). The threshold calibrated for max-F1 on the train split (~2.4) brought F1 0.79→0.89 and hit the real keep rate.
**Rule v2:** With regression filters never take the nominal label mark as the decision threshold — calibrate the threshold on train (max-F1 or target keep rate) and store it in the artifact. Per-source keep rates from a cheap 400-doc distribution sweep provide the calibration anchors.

### L-022 — Build DDP additively + gated, checkpoints DDP-agnostic
Built multi-GPU (DistributedDataParallel) into the single-process trainer: strictly `WORLD_SIZE>1`-gated, so the single-GPU path stays bit-identical (the running foundation run must not break). Two traps: (a) `DDP(model).state_dict()` prepends `module.` to every key → a multi-GPU checkpoint no longer loads single-GPU; solution: always save/load `model.module` (the unwrapped core model). (b) Eval/stop/logging must be rank-coordinated, otherwise a DDP collective hangs: rank-0 eval (forward-only, no collective) + barrier, global stop via `all_reduce(MAX)`.
**Rule v2:** Build distributed code additively and gated; verify the single-GPU path via py_compile + dry-run. Always write checkpoints without a `module.` prefix. Multi-GPU correctness needs real 2-GPU validation (RunPod) — a single-GPU box does not prove it.
