# Tokenizer fertility — Helix-200k vs. standard tokenizers

Helix v2 uses a 200k-piece SentencePiece tokenizer (tied embeddings), trained
German-primary (de55/en45). This document records a **measured** comparison of
how densely it encodes text versus widely-used tokenizers — i.e. how many tokens
each needs for the *same* text.

Fewer tokens for the same text = cheaper/faster inference per sentence and more
text per context window — **conditional on equal model quality** (this is a
tokenizer property, not a claim about model intelligence).

## Method

- Metric: **bytes per token** on real corpus samples (higher = more efficient).
  Bytes/token is tokenizer-independent, so it is a fair cross-tokenizer measure.
- Samples: 9 MB German (german-commons + FineWeb2-DE + HPLT-DE), 3 MB English
  (FineWeb-EN), Starcoder Python. Code is reported **with the pipeline's
  `--tab-indent` normalization** (4 leading spaces → tab), exactly as training
  tokenizes it — see `scripts/data/code_format.py`.
- Compared tokenizers: GPT-4o `o200k_base` (same 200k vocab budget — the
  fairest control), GPT-4 `cl100k_base` (100k), Llama-3 (128k).
- Reproduce: `python scripts/eval/tokenizer_fertility.py` (writes JSON report).

## Results — bytes per token (higher = better)

| Sample            | Helix-200k | GPT-4o o200k | Llama-3 128k | GPT-4 cl100k |
|-------------------|-----------:|-------------:|-------------:|-------------:|
| German            | **5.22**   | 4.25         | 3.56         | 3.55         |
| English           | **5.03**   | 4.86         | 4.81         | 4.81         |
| Code (raw)        | 2.43       | 3.94         | 3.98         | 3.97         |
| Code (tab-indent) | 2.77       | 3.55         | 3.58         | 3.58         |

### Helix token count vs. each tokenizer

| Sample            | vs o200k (same budget) | vs Llama-3 | vs cl100k |
|-------------------|------------------------|------------|-----------|
| German            | **−18.7%** (fewer)     | −32.0%     | −32.0%    |
| English           | −3.5% (fewer)          | −4.3%      | −4.4%     |
| Code (tab-indent) | +28.2% (more)          | +29.4%     | +29.1%    |

## What this means (honestly)

**Win — German.** At an *identical* 200k vocab budget, Helix encodes German in
**18.7% fewer tokens than GPT-4o** (32% fewer than Llama-3). The 200k budget was
spent better on German. This is the strongest, fairest claim because it controls
for vocabulary size.

**No English penalty.** German optimization did not cost English — Helix is
slightly ahead of all three there.

**Honest trade-off — code.** Helix is ~29% *less* dense on code (with tab-indent;
63% worse on raw, space-indented code). This is a deliberate design choice:
vocabulary went to European-language coverage, not code patterns. The
`--tab-indent` normalization recovers ~21% of code tokens (255k → 202k on the
sample) but does not close the gap. This is why `bpb_code` is the hardest track
for the model.

## Caveats

- Bytes/token measures **encoding efficiency, not model quality**. Fewer tokens
  does not mean a smarter model.
- At 0.9B the 200k embedding matrix is a relatively large share of parameters
  (the "embedding tax"); this share shrinks as the model scales up.
- Code numbers depend on language/indent style; Python with 4-space indent is the
  best case for `--tab-indent`.
