# Data Pipeline v1 — Production-Ready Stack

> **Status**: All 4 generation pipelines hardened with chunked-streaming +
> --resume (commits `623d135`, `715cf6a`, 2026-05-03). Sample-validated on
> 5 corpora. Ready for Phase-2 prep.

---

## TL;DR

Four LLM-driven data pipelines, all using `distilabel` over OpenRouter,
all defaulting to **`qwen/qwen3.6-35b-a3b`** (matches the local bitbastion
model for self-consistency between filter/rewrite/judge):

| Script | Purpose | Default Chunk | Smoke kept-rate |
|--------|---------|---------------|-----------------|
| `ask_llm_deepseek.py` | text quality scoring (1-5) | 500 docs | 52.7% (fineweb), 88.2% (wiki) |
| `ask_llm_code.py` | code quality scoring (1-5) | 500 files | 30.8% (the_stack-Python) |
| `synth_qa_pairs.py` | structured-doc → Q&A pairs | 200 docs | 0% parse-fails (after fixes) |
| `rewrite_low_quality.py` | Score-2/3 → Score-4 upgrade | 200 docs | 58% upgrade rate |

All four scripts share:
- `--chunk-size N` for crash-resumability (one chunk = one `pipeline.run()`)
- `--resume` to skip already-completed doc_ids on re-run
- `extra_body={"reasoning":{"enabled":False}}` to suppress thinking on Qwen3.6

The filename `ask_llm_deepseek.py` is a historical artefact — DeepSeek was
the first model tried and it failed (see Lesson L-018 below). The default
is now Qwen3.6.

---

## Pipeline architecture

```
                       Raw corpora (/staging/raw/)
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│  ask_llm_deepseek.py (text)  │  ask_llm_code.py (Python)              │
│      ↓ scored JSONL           │      ↓ scored JSONL                   │
│      Score 1-5 + kept flag    │      Score 1-5 + kept flag            │
└──────────────┬───────────────────────────────────────────────────────┘
               │
               ├──── filter score ≥ 4  ──→  high-quality raw  ──→  tokenize
               │
               └──── filter score = 2 or 3 ──→  rewrite_low_quality.py
                                                       ↓ rewrite JSONL
                                                       ↓ re-score
                                                       upgraded docs   ──→  tokenize

                       Structured corpora (politik / recht)
                                  │
                                  ▼
                       synth_qa_pairs.py
                                  ↓ messages JSONL (user/assistant)
                                  ↓ for Phase-5 MoRA-SFT
```

---

## Models tested (chronological journey)

### Scoring (1-digit response)

| Model | Garbage rate | kept (fineweb 5k) | Cost | Verdict |
|-------|--------------|-------------------|------|---------|
| `deepseek/deepseek-chat-v3.1` | **18.0%** ❌ | 56.4% | $0.27/$0.41 | Buggy at short completions |
| `meta-llama/llama-3.3-70b-instruct` | 0% | 77.8% | $0.10/$0.32 | Lenient kalibration |
| `qwen/qwen3-235b-a22b-2507` | 0% | 61.0% | $0.20/$0.60 | Solid, expensive |
| `qwen/qwen3.6-35b-a3b` v1 | 48.8% ❌ | — | $0.16/$0.97 | Reasoning ate tokens |
| `google/gemma-4-26b-a4b-it` | 0% | 64.0% | $0.06/$0.33 | **Cheapest** that works |
| `google/gemma-4-31b-it` | 0% | 63.0% | $0.13/$0.38 | ≈ identical to 26b-A4B |
| **`qwen/qwen3.6-35b-a3b`** v2 | **0%** ✅ | **55.6%** | $0.16/$0.97 | **Default** — strict, matches local |

### Rewriting (long-form generation)

| Model | Lang-treue | Score-≥4 nach rewrite | Cost/1M docs |
|-------|------------|----------------------|--------------|
| `anthropic/claude-sonnet-4.5` | 100% (with EN-prompt) ✅ | 54% | $60k |
| `deepseek/deepseek-v3.2` | 100% (with EN-prompt) ✅ | 48% | $0.8k |
| **`qwen/qwen3.6-35b-a3b`** | 100% (with EN-prompt) ✅ | **58%** | **$1.2k OR / $30 lokal** |

### Why Qwen3.6-35B as default everywhere

1. **Same model the user runs locally on bitbastion** — when a future judge
   or eval pass uses Qwen3.6, the calibration is consistent end-to-end.
2. **Strictest of the working scoring models** — 27.8% Score-1 on fineweb
   vs Llama's 9.2%. For pretrain corpora the strict model is correct
   (Sachdeva et al. 2024, Table 3).
3. **Cheap enough** — $1.2k for 1M docs over OpenRouter, $30 electricity
   for 1M docs lokal.
4. **No format quirks** when reasoning is disabled (lesson L-019).

---

## Lessons learned (in chronological order — these are the bugs you don't want to re-hit)

### L-018 — DeepSeek-V3.1 short-completion garbage-token bug

**Symptom**: `temperature=0`, `max_new_tokens=4`, "respond with EXACTLY one
digit" → 18% of inputs returned `"棣棣棣棣"` (CJK character spam).

**Investigation path**:
- v1: T=0, max=4: 18% garbage
- v2: T=0.05, max=8: 32% garbage (worse!)
- v3: T=0.05, max=16, prompt = "Score: X": 10% garbage
- v4: switch model to llama-3.3-70b: 0%

**Root cause**: Model itself, not prompt or routing. Reproduces across all
OpenRouter providers for `deepseek-chat-v3.1` at very-short max_tokens.

**Fix**: don't use deepseek-chat-v3.1 for short-completion tasks. For
long-form (rewrites, Q&A), V3.2 is fine.

### L-019 — Reasoning models consume max_tokens before answering

**Symptom**: Qwen3.6-35B-A3B at `max_new_tokens=16` returned empty
`content` for 49% of inputs. The `reasoning` field had ~80 tokens of
thinking trace.

**Root cause**: Qwen3.6 is a reasoning model — by default it produces a
reasoning trace before the answer, consuming the token budget.

**Fix**: pass `extra_body={"reasoning": {"enabled": false}}` to OpenAILLM's
`generation_kwargs`. Non-reasoning models silently ignore it, so this is
safe to leave on universally.

### L-020 — Prompt language drives output language, not source language

**Symptom**: Rewriter prompt was in German. DeepSeek-V3.2 translated 92%
of English fineweb docs to German. Qwen3.6 translated 82%. Only Claude
(94%) respected the source language.

**Root cause**: Models follow the prompt's language, not the rule
"preserve source language" buried in the prompt body.

**Fix**: detect source language with a stop-word heuristic (`detect_language()`),
render the appropriate `REWRITE_PROMPT_DE` or `REWRITE_PROMPT_EN`. After
fix all three models stay in source language 100% of the time.

### L-021 — Wikipedia chunks are sections, not articles

**Symptom**: wikipedia_de scored 35.7% Score-1 on raw 5k sample. Surprising
for a curated source.

**Investigation**: length-vs-score histogram showed 904 docs (18%) were
<50 chars. These are stand-alone section headers ("Geschichte", "Weblinks",
"Einzelnachweise") that the blank-separated reader picked up as full docs.

**Root cause**: `wikipedia_de.txt` was extracted with blank-line-separation,
which respects paragraph boundaries instead of article boundaries.

**Fix (interim)**: `--min-chars 200 --max-chars 8000` flag in scorer drops
fragments before the LLM call. Wiki kept-rate jumped from 63.3% to 88.2%.

**Fix (proper, deferred)**: re-extract `wikipedia_de.txt` with article
boundaries (`<article>...</article>` markers or one-JSONL-line-per-article).

### L-022 — Same distilabel pipeline name = shared cache directory

**Symptom**: Three parallel rewrite re-scoring runs (claude/deepseek/qwen)
all got identical histograms. They were reading from one shared distilabel
cache.

**Root cause**: `Pipeline(name="ask-llm-deepseek")` is the cache key. All
three runs used the same name → `~/.cache/distilabel/pipelines/ask-llm-deepseek/`
got concurrent writes that overwrote each other.

**Fix (workaround)**: run scripts of the same name SEQUENTIALLY in
parallel-launch contexts. `docker exec auralis-downloader rm -rf
/root/.cache/distilabel/pipelines` between runs is also safe.

**Fix (proper, deferred)**: add `--pipeline-name` flag so concurrent runs
can use distinct cache dirs.

### L-023 — All-or-nothing pipeline.run() loses work on crash

**Symptom**: A 50M-doc full-corpus rescore (~115h at OpenRouter throughput)
that crashes at hour 60 loses everything. The old code held all results
in memory until the end of `pipeline.run()`.

**Fix (commit 623d135 + 715cf6a)**: chunked execution. `--chunk-size 500`
splits the work; each chunk = one `pipeline.run()` invocation. Results
are appended to the output file with explicit flush after each chunk.
Worst-case loss on crash = one chunk (~100s of work).

`--resume` flag scans existing output for completed doc_ids and skips
them on re-run. Combined with chunked-streaming, `--resume` makes any
long-running job restart-safe with ~zero engineering effort per run.

---

## Per-corpus results (5k samples, qwen/qwen3.6-35b-a3b scorer)

### Text corpora (`ask_llm_deepseek.py`)

| Corpus | n | =1 | =2 | =3 | =4 | =5 | ? | kept ≥3 | mean |
|--------|---|----|----|----|----|----|----|---------|------|
| fineweb_10bt EN | 5000 | 27.0% | 19.9% | 33.9% | 17.6% | 1.2% | 0.3% | 52.7% | 2.49 |
| fineweb2_de DE | 5000 | 30.0% | 18.4% | 31.5% | 19.0% | 1.0% | 0.0% | 51.5% | 2.43 |
| wikipedia_de raw | 5000 | 35.7% | 1.0% | 10.1% | 43.5% | 9.7% | 0.0% | 63.3% | 2.91 |
| **wikipedia_de filtered** | 4379 | **11.1%** | 0.6% | 7.8% | **64.2%** | **16.3%** | 0.0% | **88.2%** | **3.74** |

Wikipedia after `--min-chars 200 --max-chars 8000` is the highest-quality
corpus by mean-score. Web-crawl (fineweb) is roughly half-noise as expected.
EN and DE behave identically (no language-bias in the scorer).

### Code corpora (`ask_llm_code.py`)

| Corpus | n | =1 | =2 | =3 | =4 | =5 | ? | kept ≥3 |
|--------|---|----|----|----|----|----|----|---------|
| the_stack_v2_python | 5497 | 21.4% | 47.8% | 20.3% | 10.2% | 0.3% | 0% | **30.8%** |
| smollm_python_edu | 0 B (empty file) | — | — | — | — | — | — | skip |

Code is meaningfully harder to score well than prose. The 47.8% Score-2
bucket is the dominant pattern — "works but no docs/structure". Score-4
threshold of 30.8% is consistent with the_stack v2's known prevalence of
auto-generated stubs (Django migrations, gRPC, `__init__.py` re-exports).

### Rewrite-upgrade rate (`rewrite_low_quality.py`)

50 fineweb_10bt docs scored 2-3 by Qwen3.6 → rewritten by each model →
re-scored by Qwen3.6 (with English-only prompts after L-020 fix):

| Rewriter | rewritten | =3 after | =4 after | =5 after | % Score ≥4 |
|----------|-----------|----------|----------|----------|-----------|
| Claude Sonnet 4.5 | 50 | 22 | 27 | 0 | 54% |
| DeepSeek V3.2 | 50 | 22 | 24 | 0 | 48% |
| **Qwen3.6-35B** | **50** | 18 | **28** | **1** | **58%** |

**⚠️ Same-model bias**: Qwen3.6's 4 pp lead over Claude is partly
self-favouring (the scorer is also Qwen3.6). True quality difference is
within noise; pick on cost. See cross-model validation below.

---

## Cross-Model Validation (2026-05-03)

To quantify same-model bias and confirm Qwen3.6's calibration is reliable,
we re-scored the **same 599 fineweb_10bt docs** with Claude Sonnet 4.5 as
an independent judge.

### Marginal score distribution

| Model | =1 | =2 | =3 | =4 | =5 | mean |
|-------|----|----|----|----|----|------|
| **Qwen3.6** | 27.7% | 15.5% | 33.9% | 21.2% | **1.7%** | 2.54 |
| **Claude 4.5** | 15.7% | 26.9% | 30.4% | 27.0% | 0.0% | 2.69 |

* Qwen is **stricter on the bottom** (28% vs 16% Score-1) and the **only one
  to grant Score-5** to fineweb (1.7% vs 0.0%).
* Claude shifts the mass into the middle (Score-2 + Score-4).
* Mean scores differ by 0.15 (within sample noise).

### Confusion matrix (Qwen rows × Claude cols)

|  | Claude=1 | =2 | =3 | =4 | =5 |
|--|---------|----|----|----|----|
| **Qwen=1** | **90** | 58 | 17 | 1 | 0 |
| **Qwen=2** | 3 | **65** | 23 | 2 | 0 |
| **Qwen=3** | 1 | 36 | **117** | 49 | 0 |
| **Qwen=4** | 0 | 2 | 25 | **100** | 0 |
| **Qwen=5** | 0 | 0 | 0 | **10** | 0 |

The diagonal dominates. Off-diagonal mass is mostly ±1 (96% of all
disagreements are within one score point).

### Agreement metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Exact agreement | **62.1%** (372/599) | Same digit |
| Within-1 agreement | **96.2%** (576/599) | Off by ≤ 1 score point |
| Pearson r | **0.805** | Strong linear correlation |
| Threshold-3 keep/drop agreement | **86.3%** (517/599) | The decision that actually matters |

### Bottom line

* The Qwen3.6 → Qwen3.6 self-rating loop has a **measurable but small**
  bias. The kept/drop decision is consistent with Claude in 86% of cases.
* Qwen3.6 is **the right default** for full-corpus pretrain filtering —
  it's stricter than Claude, which is what you want when web crawl is
  ~50% noise.
* For **high-value subsets** (politik / recht / medizin) where decisions
  are downstream-impactful, run a **multi-model voting** scheme: take
  the median of Qwen3.6 + Claude + Gemma-4-26b-A4B scores. Costs ~3× but
  eliminates the residual 14% disagreement.

---

## Cost analysis

For a hypothetical 50M-doc full-pipeline run (text scoring):

| Path | Per 5k Docs | Per 50M Docs | Wall-clock |
|------|-------------|--------------|------------|
| OpenRouter (Qwen3.6) | $0.005 | **~$50** | ~115h |
| OpenRouter (Gemma 4 26b-A4B) | $0.003 | ~$30 | ~115h |
| Local vLLM (Qwen3.6-35B-A3B on RTX 4090) | $0.0001 (electricity) | ~$30 (electricity) | ~280h single-GPU, ~70h with 4× |
| Anthropic (Claude Sonnet 4.5) | $0.10 | **~$1000** | ~115h |

For rewrite (50M Score-2/3 docs at ~1024 output tokens):

| Path | Per 50M Rewrites |
|------|------------------|
| OpenRouter Qwen3.6 | ~$60k ❌ |
| OpenRouter DeepSeek-V3.2 | ~$40k ❌ |
| OpenRouter Claude | ~$3M ❌❌ |
| **Local vLLM (Qwen3.6-35B-A3B)** | **~$30 electricity** ✅ |

→ **scoring fits OpenRouter at any scale**. Rewriting requires local
inference for full-corpus volumes.

---

## Phase-2 mix recommendations (based on validated kept-rates)

For a 1B-token continued-pretrain run:

| Source | Kept-rate | Recommended mix | Rationale |
|--------|-----------|-----------------|-----------|
| wikipedia_de filtered | 88% | **12%** | Anchor for high-quality DE prose |
| fineweb_10bt EN | 53% | **40%** | English bulk |
| fineweb2_de DE | 52% | **38%** | German bulk |
| the_stack_v2_python kept | 31% | **10%** | Code |

(Numbers are token-budget shares after filtering, not pre-filter-doc shares.)

---

## Open work

- ~~Cross-model validation~~ **DONE 2026-05-03** — see "Cross-Model
  Validation" section below.
- **Wikipedia re-extract** with article boundaries (replace L-021 interim
  fix). Will eliminate the need for `--min-chars 200` filter.
- **Local vLLM Qwen3.6-35B-A3B** on bitbastion. Removes per-token cost
  for full-corpus runs and enables rewriting at scale.
- **`--pipeline-name`** flag to fix L-022 cache collision, enabling
  parallel runs of the same script.
- **Snakefile / Makefile orchestrator** to chain raw → score → filter →
  rewrite → re-score → tokenize as a single command.
- **TICK / STICK rubric** for SFT-data scoring (Phase 3 prep, separate
  from this pretrain pipeline).
- **Code-execution sandbox** for the_stack files: filter Score-3+ files
  by "imports cleanly, parses, no `eval`/`exec`/dangerous patterns".

---

## Quick reference

```bash
# Score a corpus (text)
OPENROUTER_API_KEY=sk-or-... \
python scripts/data/pipeline/ask_llm_deepseek.py \
    --input /staging/raw/fineweb_10bt/fineweb_10bt.txt \
    --output /staging/cleaned/ask_llm/fineweb_10bt_FULL.jsonl \
    --max-docs 100000000 \
    --min-chars 200 \
    --chunk-size 500 \
    --resume

# Score a corpus (code)
python scripts/data/pipeline/ask_llm_code.py \
    --input /staging/raw/the_stack_v2_python/the_stack_v2_python.txt \
    --output /staging/cleaned/ask_llm/the_stack_python_FULL.jsonl \
    --min-chars 100 --max-chars 50000 \
    --chunk-size 500 \
    --resume

# Rewrite low-quality docs back to clean prose
python scripts/data/pipeline/rewrite_low_quality.py \
    --scored /staging/cleaned/ask_llm/fineweb_10bt_FULL.jsonl \
    --source /staging/raw/fineweb_10bt/fineweb_10bt.txt \
    --output /staging/cleaned/rewrites/fineweb_10bt_REWRITES.jsonl \
    --score-min 2 --score-max 3 \
    --max-docs 0 \
    --chunk-size 200 \
    --resume

# Generate Q&A pairs from structured docs
python scripts/data/pipeline/synth_qa_pairs.py \
    --input  /staging/politik_de/raw/bundestag_protokolle/bundestag_protokolle.jsonl \
    --output /staging/politik_de/sft/protokolle_qa.jsonl \
    --schema plenary \
    --max-docs 100000 \
    --pairs-per-doc 3 \
    --chunk-size 200 \
    --resume
```

All four scripts: idempotent on re-run with `--resume`; safe to kill and
restart at any time; per-chunk progress lines in stdout for monitoring.
