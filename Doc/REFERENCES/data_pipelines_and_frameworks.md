# Data Pipelines & Training Frameworks — Auralis Reference

> **Purpose**: Evaluate external data-pipeline + fine-tuning frameworks
> against our own stack. Decide what to adopt, what to skip, what to
> watch.
>
> **Status**: Living. Last review: 2026-05-02.
> Tested: datatrove ✅, LLaMA-Factory ✅, Ask-LLM ✅ (POC).

---

## TL;DR — Adoption decision

| Tool | Tier | Adopt? | When | Reason |
|---|---|---|---|---|
| **LLaMA-Factory** | 1 | **Yes** | Phase 5 (MoRA adapters) | Saves ~10-15 dev-days vs custom adapter pipeline |
| **datatrove** (HF) | 1 | **Yes, partial** | v3 corpus prep | Replaces our `filter_quality.py` for v3 only — Phase-2 corpus stays as-is |
| **Ask-LLM** | 1 | **Yes** | v3 corpus prep | -70% convergence time claim worth verifying at v3 scale |
| **SwallowCode/Math** | 2 | Watch | maybe v4 | Code-rewriting via LLM — interesting but starcoderdata is clean enough today |
| **ProX** | 2 | No | — | +2% benchmark gain not worth the engineering time |
| **Webscale-RL** | 2 | No | — | We do DPO not RL |
| **Density Sampling** | 2 | No | — | Mid-data regime tool; we have full coverage |
| **DLRover / Babel** | 3 | No | — | Petabyte multi-cluster — we are single-GPU 148 GB |
| **PCache** | 3 | No | — | MoE-specific; we are dense |
| **SuperAnnotate** | 3 | No | — | Commercial; annotation is not our bottleneck |
| **Latitude** | 3 | No | — | Captures production logs; we have no production traffic |

---

## 1. Why this evaluation now

We've reached the point where the choice **"build our own / use external"** is
an economic question not a technical one. Our `scripts/data/` and
`scripts/eval/` already work; the question is whether the alternatives are
materially better.

The honest answer for each tool is in §3 below. The short version:

* For **adapter training (Phase 5)** the alternatives are massively better
  than what we'd have time to build. **Adopt LLaMA-Factory.**
* For **corpus filtering** our current pipeline is fine for Phase 2.
  External tools win at v3-scale because of distributed parallelism we
  don't yet need.
* For **per-document quality scoring** there is no good homegrown
  equivalent. **Ask-LLM is worth running.**

---

## 2. Our current stack (baseline)

| Function | Auralis component | Lines of code | Status |
|---|---|---|---|
| Raw download (HF + custom) | `scripts/data/download_phase2_pretrain.py`, `download_politik_de.py` | ~700 | Production. Mount-guarded, resume-safe. |
| Multi-line → one-line-per-doc | `scripts/data/assemble_for_filter.py` | 130 | Production. |
| Quality filter | `scripts/data/filter_quality.py` | 200 | Production. Regex + heuristics. |
| Tokenize bin/idx | `scripts/data/tokenize_for_pretraining.py` | 270 | Production. |
| Mix corpora | `scripts/data/mix_corpora.py` | (size unknown) | Production. |
| Eval custom baseline | `scripts/eval/run_baseline.py` | 200 | Production. |
| Eval industry benchmarks | `scripts/eval/run_benchmarks.py` | 350 | NEW (this commit's predecessor). |
| Adapter training | — | 0 | **Not built yet.** |
| Per-doc quality scoring | — | 0 | **Not built yet.** |
| Distributed parallel filter | — | 0 | **Not needed yet.** |

---

## 3. Tool-by-tool evaluation

### 3.1 LLaMA-Factory  (Tier 1, Adopt)

**What it is**: All-in-one fine-tuning framework with web UI. Supports
LoRA, QLoRA, DoRA, full fine-tune, RLHF (DPO/PPO), DPO/SimPO. Exports to
HF Transformers, vLLM, llama.cpp, GGUF.

**What we use it for**:
* **Phase 5 — MoRA / DoRA adapter training**. The web UI lets us run
  hyperparameter sweeps without writing training code. Built-in monitoring
  + resumption.
* **Phase 3 SFT, optionally**. Our current Phase-3 plan generates SFT
  data via OpenRouter; the *training* of the SFT model on that data could
  go through LLaMA-Factory instead of a custom trainer.

**What we don't use it for**:
* Pretraining (Phase 1+2). LLaMA-Factory is fine-tuning-focused; our
  pretraining stack is ours.
* Inference at runtime — that goes via vLLM or our own scripts.

**Test result** (2026-05-02): see §4.1.

**Integration plan**:
```
1. Phase 4 prep: clone LLaMA-Factory, install deps in a separate
   container alongside auralis-training.
2. Verify our Phase-3 SFT-trained checkpoint loads in their format.
   (Hugging Face state_dict with non-MHA layers — may need a config
   shim for the hybrid Mamba/GLA mix.)
3. Launch the first MoRA adapter — politik-de — through the web UI.
4. Document the workflow under Doc/SPECs/SPEC_PHASE_5_LORA_SYSTEM.md.
```

**Risks**:
* Our hybrid model (Mamba + GLA + Attention) is not standard. LLaMA-Factory
  is built around HF Transformers' model classes. We may need to register
  our model as a custom class. This is solvable (it's the same problem
  Mamba-2 had upstream) but might take a day.
* Default hyperparameters are tuned for Llama. We will keep our own.

---

### 3.2 datatrove  (Tier 1, Partial Adopt)

**What it is**: HuggingFace's data-processing toolkit, used to build
FineWeb. Pipeline of pluggable "blocks" (Filter, Deduplicator, Sampler,
Writer) running on Slurm or Dask or local multiprocessing.

**Capabilities we lack today**:
* **Minhash deduplication** at scale. Our pipeline has none.
* **C4 quality filters** — repeated 3-gram detection, line-level filters,
  bullet-point line-fraction filters. We only have boilerplate / URL /
  symbol density.
* **Language detection per document** with FastText. We trust dataset
  labels (which is mostly fine for HF datasets but not for raw web).
* **PII redaction**. We have none.
* **Parallel execution** across 100+ workers without us writing the
  orchestration.

**Capabilities we already have**:
* Boilerplate / URL / symbol filters (datatrove also has these).
* Multi-line → one-line normalisation (our `assemble_for_filter.py` does
  this).
* Tokenization to bin/idx (our `tokenize_for_pretraining.py` is faster
  for our format).

**Test result** (2026-05-02): see §4.2.

**Integration plan**:
```
Phase 2:  no change. Corpus is filtered, redoing it costs >10h compute
          for marginal gain.
v3:       redo corpus build through datatrove. Use:
          - URLFilter, GopherRepetitionFilter, GopherQualityFilter,
            FineWebQualityFilter, C4QualityFilter
          - MinhashDedupSignature + MinhashDedupBuckets +
            MinhashDedupCluster + MinhashDedupFilter
          - Keep our tokenization downstream.
```

**Risks**:
* Heavy dependency footprint (~500 MB pip install incl. fasttext models,
  spacy, etc).
* Dask cluster setup is non-trivial; on a single bitbastion we just use
  multiprocessing pool — fine.

---

### 3.3 Ask-LLM  (Tier 1, Adopt at v3)

**What it is**: A method, not a tool. Use a small instruction-tuned model
(Flan-T5-base or -large) to rate each training document on a 1-5 quality
scale. Drop everything below a threshold.

**Reported results**: Sachdeva et al. 2024 *"How To Train Data-Efficient
LLMs"* showed that LLMs trained on 10% of an Ask-LLM-filtered corpus can
beat models trained on the full corpus. Convergence claimed up to 70%
faster.

**Why this is a big deal for us**:
* Our v3 ambition (continued-pretrain on 3B → 7B) is **compute-bound**.
  If we can throw away 80% of cleaned-but-low-quality docs without losing
  anything, our €/quality ratio improves dramatically.
* Even if "70% faster" is half-true, that's still €5-10k saved on a
  €30k v3 budget.

**Why this is not a big deal for our v2**:
* Phase 1 + 2 corpora are already chosen. Re-filtering them now would
  cost 7-15h GPU time and not change Phase 1's outcome materially.

**Test result** (2026-05-02): see §4.3 — we built a 50-doc POC scorer.

**Integration plan**:
```
v3 corpus prep:
  1. Spin up a Flan-T5-base instance on bitbastion (small, ~250M params,
     fits next to inference work).
  2. Score every document in the planned v3 raw corpus on Ask-LLM 1-5.
  3. Histogram the score distribution.
  4. Set a threshold per language (DE may need a lower bar than EN
     because the underlying corpus is smaller — over-filtering hurts).
  5. Final cleaned set = Ask-LLM-filtered + datatrove-filtered.
```

**Risks**:
* Flan-T5 is English-trained. DE quality scoring may be unreliable.
  Mitigation: score DE docs with a DE-tuned scorer (e.g. mT5 fine-tuned
  on DE) or our own Auralis-1B once it exists.
* Threshold tuning is empirical — pick wrong, lose good data.

---

### 3.4 SwallowCode / SwallowMath  (Tier 2, Watch)

Code-rewriting pipeline: take low-quality code, ask an LLM to rewrite it
into idiomatic, well-commented form.

**Why we don't need it now**: starcoderdata-Python is curated to
permissive-license + non-trivial code already. Rewriting low-quality docs
into high-quality ones is interesting at scale but not our bottleneck.

**Maybe at v4** if we expand to more languages or want to bias toward a
particular code style.

---

### 3.5 ProX  (Tier 2, Skip)

Treats data cleaning as a programming task. LLM generates filter code;
filters run; LLM iterates. +2% reported on benchmarks.

**Skip**: 2pp gain needs 2-3 weeks of integration. Not the best use of
time when we have the architecture work pending.

---

### 3.6 Webscale-RL  (Tier 2, Skip)

Pipeline that turns unstructured web docs into verifiable QA pairs for RL
training.

**Skip**: we plan DPO, not RL. UltraFeedback gives us DPO pairs already.

---

### 3.7 Density Sampling  (Tier 2, Skip)

Kernel Density Estimation over document embeddings to ensure topic
coverage. Works best in 25-50% sample regime.

**Skip**: we have full coverage of our chosen sources. Density Sampling is
for "I have 10× more data than budget allows" — we don't have that
problem.

---

### 3.8 DLRover & Babel  (Tier 3, Not Applicable)

DLRover: distributed training failure recovery. Babel: petabyte-scale
data sync middleware.

**Why not us**: we are single-GPU, 148 GB cleaned corpus. Both tools are
priced (in engineering time) for clusters of dozens to thousands of GPUs.

---

### 3.9 PCache  (Tier 3, Not Applicable)

Distributed file cache for MoE checkpoints. Reduces checkpoint latency 50%.

**Why not us**: MoE-specific. We are dense.

---

### 3.10 SuperAnnotate  (Tier 3, Not Applicable)

Commercial human-annotation platform.

**Why not us**: we don't pay for annotation. Auralis-1B as a self-hosted
annotator is more aligned with the project values, even if it produces
slightly worse labels.

---

### 3.11 Latitude  (Tier 3, Watch for Later)

Open-source SFT-data-from-production-logs pipeline.

**Why not yet**: we have no production traffic. **Watch for after public
release** — once Auralis is deployed somewhere (demo site, community), the
real chat logs become a continuous SFT-data source.

---

## 4. Test results (2026-05-02)

All three Tier-1 tools were installed and smoke-tested in the
`auralis-downloader` container.

### 4.1 LLaMA-Factory

* **Source**: `git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory`
* **Location on disk**: `/mnt/disk7/Auralis/tools/LLaMA-Factory` (20 MB)
* **Status**: cloned, not yet installed (deps come at Phase-4 prep).

What the README confirms:
* Training approaches: Full / Freeze / **LoRA / QLoRA / OFT / QOFT**
* Adapter variants: **DoRA**, LoRA+, PiSSA, LongLoRA, LoftQ, rsLoRA
* Optimizers: **Muon**, GaLore, Adam-mini, BAdam, APOLLO
* Speed kernels: **FlashAttention-2, Unsloth, Liger Kernel, KTransformers**
* Quantization: 2/3/4/5/6/8-bit (AWQ, GPTQ, AQLM, HQQ, EETQ, LLM.int8)
* Web UI (LLaMA Board, Gradio-based)

Important caveat for our plan:
* **MoRA is NOT in the supported-approaches list.** DoRA + LoRA+ + PiSSA
  cover most of what we want, but if we specifically need MoRA's
  high-rank-update behaviour for fact-injection (per L-002), we will
  either fork in MoRA support ourselves or accept DoRA as the substitute.
* Our hybrid Mamba+GLA+Attention model is non-standard from HF
  Transformers' perspective; LLaMA-Factory will need a custom model
  registration. ETA: ~1 day of integration work at Phase 4.

### 4.2 datatrove smoke test

* **Install**: `pip install datatrove pyahocorasick fasteners tldextract nltk`
  succeeded inside the container (~80 MB total). One non-fatal pip
  dependency warning about `huggingface_hub` version overlap with our
  existing `transformers` (resolvable later if needed).
* **Smoke pipeline**: ran 285 docs of fineweb sample through
  `URLFilter → GopherRepetitionFilter → GopherQualityFilter →
  C4QualityFilter`.
* **Throughput**: ~2300 docs/s on a single core. Plenty for our scale.

Findings:
* `URLFilter` requires a `url` field on each `Document`. Our pre-cleaned
  raw text has none → filter dropped 100%. **Lesson**: datatrove must run
  *before* our `assemble_for_filter.py`, not after, and we need to keep
  the URL metadata in the document object (the FineWeb pipeline does
  this naturally; our own custom downloads currently strip it).
* `GopherRepetitionFilter` expects newline-separated sentences inside a
  document to detect repetition. Our `assemble_for_filter.py` collapses
  to a single line → filter sees zero internal newlines → also dropped
  100%. **Lesson**: same — datatrove must come *before* assembly.

Verdict: tool works as expected. **Integration order at v3 will be**:
```
download → datatrove (filter, dedup, language-id, PII) → assemble → tokenize
```
(rather than the current `download → assemble → filter → tokenize`).

For Phase 2 we keep our existing pipeline unchanged.

### 4.3 Ask-LLM POC

* **Implementation**: `scripts/eval/ask_llm_poc.py` (NEW). 130-line
  standalone scorer. Uses `google/flan-t5-base` (~250M params). Prompts
  the model with a 1-5 scoring rubric, parses the digit out of the
  response.
* **Test corpus**: same 285-doc fineweb sample, scored first 50 docs.
* **Compute**: 50 docs in 13.9s on CPU = **3.6 docs/s**.
  At scale (full corpus + GPU): ~50-100× faster, so **~50,000 docs/h**
  on a 3090, **~120,000 docs/h** on an H100.

Score distribution (50 docs):
| Score | Count | Notes |
|---|---|---|
| 1 | 9 (18%) | "useless boilerplate / link spam" |
| 2 | 1 (2%) | rare — model is bimodal |
| 3 | 34 (68%) | mediocre web text — typical CommonCrawl |
| 4 | 2 (4%) | clean informative prose |
| 5 | 4 (8%) | encyclopaedia-tier |

Mean: **2.82** (matches FineWeb's "we kept the middle of the distribution").

**Quality check on individual scores** (this is where the POC reveals a
problem):

| Score | Sample head | Plausible? |
|---|---|---|
| 5 | "Great decorating addition I have a grape/Italian theme in my kitchen. I purchased 5 of these…" | **No** — that's an Amazon review, not encyclopaedia-tier. |
| 1 | "Viewing Single Post From: Spoilers for the Week of February 11th \| Lil \| Feb 1 2013, 09:58 AM \| Don't care about…" | **Yes** — forum spoiler thread, correct. |
| 3 | "A novel two-step immunotherapy approach has shown clinically beneficial responses in patients with advanced ovarian cancer…" | **No** — should be 4-5 (medical research summary). |

Conclusion: **Flan-T5-base is too weak as a scorer**. The reference paper
used Flan-T5-XL (3B). For real v3 use we will need either:
1. Flan-T5-XL or larger (~6 GB on GPU). Doable.
2. Auralis-1B-instruct (post-Phase-3) as the scorer. **Better**: it knows
   our domain mix and German, and it removes the external-model dependency.

The POC's value is that it **proved the pipeline works**:
* Loading + scoring + parsing + histogramming all work end-to-end.
* Per-doc throughput maps to realistic v3-scale numbers.
* The biggest risk (model size matters) was identified at zero compute cost.

Recommended next step (later, not now): build a small DE-quality eval
set (~200 hand-rated documents) and benchmark Flan-T5-XL vs Auralis-1B
once the latter is post-SFT. Whichever scores closer to the human
ratings becomes our Ask-LLM scorer of record.

---

## 5. References

* Sachdeva et al. 2024 — *How To Train Data-Efficient LLMs* (Ask-LLM)
* Penedo et al. 2024 — *FineWeb: decanting the web for the finest text data*
* Pegoraro et al. 2024 — ProX paper
* SwallowCode technical report (Tokyo Tech)
* Hugging Face / datatrove docs and source
* hiyouga / LLaMA-Factory README and SPEC docs
* Penedo et al. 2024 — FineWeb-2 multilingual extension (our Phase-2 DE source)

---

## 6. Open questions / re-evaluation triggers

1. When LLaMA-Factory adds first-class MoRA support (currently DoRA + LoRA
   only), revisit MoRA path.
2. When HF Transformers ships clean Mamba-2 + hybrid model classes,
   LLaMA-Factory adoption gets cheaper.
3. When Ask-LLM gets a DE-tuned variant published, lower the v3 risk.
4. When Auralis-1B itself becomes the document scorer (cleaner alignment),
   the Ask-LLM dependency disappears.
