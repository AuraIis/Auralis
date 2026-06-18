# Auralis v2 / Helix v2

<p align="center">
  <img src="docs/auralis_logo_512.png" width="200" alt="Auralis logo"><br>
  <em>A from-scratch, German-primary ~0.9B hybrid LLM (Mamba-2 / GLA / sparse attention).</em>
</p>

Auralis is the assistance system. Helix v2 is the in-house LLM underneath it.

The current working state is in [STATUS.md](STATUS.md). The overarching
project idea and model philosophy are in
[Doc/AURALIS_V2_PROJECT_BRIEF.md](Doc/AURALIS_V2_PROJECT_BRIEF.md). The
technical architecture spec is in
[Doc/SPECs/SPEC_PHASE_0.5_MODEL_ARCHITECTURE.md](Doc/SPECs/SPEC_PHASE_0.5_MODEL_ARCHITECTURE.md).

## Quick Links

- [Current Status](STATUS.md)
- [Roadmap & Status (Blueprint)](docs/auralis_roadmap_blueprint_en.svg)
- [Blueprint: Tool-Use & Verifier](docs/BLUEPRINT_TOOL_USE_VERIFIER.md)
- [Blueprint: DoRA Domain Adapters](docs/BLUEPRINT_DOMAIN_ADAPTERS_DORA.md)
- [Future Backlog](docs/ZUKUNFT_BACKLOG.md)
- [Docs Index](docs/DOCS_INDEX.md)
- [Project Brief / Core Idea](Doc/AURALIS_V2_PROJECT_BRIEF.md)
- [Model Architecture](Doc/SPECs/SPEC_PHASE_0.5_MODEL_ARCHITECTURE.md)
- [Data Cleaning Pipeline V3](docs/data_cleaning_pipeline_v3.md)
- [Dataset Market App](docs/dataset_market_app.md)
- [Evaluation](eval/README.md)
- [Lessons](LESSONS.md)
- [History](HISTORY.md)

## Current Focus (as of 2026-06-06)

Pipeline, checkpointing, tokenizer and training run stably. A lot has happened
since the German edu-data filter:

1. **Foundation run completed** (1B, de55/en45, warm-start v3 up to step 50k):
   healthy training, language + factual grounding demonstrated.
2. **SFT (instruction tuning):** the base, which could barely answer, was turned
   into an answering assistant. v1 (~32k diverse DE+EN, gpt-4o-verified,
   decontaminated) + v2 (+ German reasoning slice, gpt-4o math-checked).
   **SFT teaches FORM, not KNOWLEDGE** — confirmed by benchmarks.
3. **Benchmarks** (own MC log-likelihood runner, n=300): Helix-SFT beats
   SmolLM2-360M + TinyLlama-1.1B on `mmlu_de`; Qwen's MMLU lead shrinks
   from ~22 (EN) to ~7 (DE). The language strategy (200k vocab, de55/en45) pays
   off measurably. Absolute values low = under-training / size signal.
4. **Next direction** (triple-aligned, order gated): tool-use
   first (small model learns to VERIFY instead of guess) -> annealing including code
   -> DoRA domain adapters. Specs:
   [Tool-Use](docs/BLUEPRINT_TOOL_USE_VERIFIER.md),
   [DoRA](docs/BLUEPRINT_DOMAIN_ADAPTERS_DORA.md),
   [Backlog](docs/ZUKUNFT_BACKLOG.md).

Roadmap at a glance:

![Roadmap & Status](docs/auralis_roadmap_blueprint_en.svg)

Details: the "Update 2026-06-06" block in [STATUS.md](STATUS.md), the timeline
in [HISTORY.md](HISTORY.md), the lessons (incl. L-018..L-022) in
[LESSONS.md](LESSONS.md).

## What Helix can do today — and what it cannot (honestly measured)

The 0.9B model runs live in the Auralis Hub (PyTorch-Ollama shim) with auto-router,
tool execution, local de-Wikipedia RAG + web search, input normalizer and
single-turn context. State of the measured capabilities:

| Can do | Behavior |
|---|---|
| German facts | capitals, authors, general knowledge — fast, correct on common facts |
| **Honest abstain** (signature) | says "I don't know" for invented/unknown terms instead of hallucinating |
| Math via tool | never computes in its head — tool call, execution, verified result |
| RAG / grounded | local de-Wikipedia (2.84 million articles) + live web; reads the context, answers with evidence or abstains |
| Code | simple, runnable functions; clean stop |
| Auto-router | automatically chooses math / code / RAG / web / chat |
| Robust against "dirty" input | normalizer cleans up typos/slang/umlauts *before* the model |

| Cannot (yet) do — measured, model-size-bound | |
|---|---|
| Reliable world knowledge | confabulates untrained facts; RAG mitigates, the real fix is a larger model |
| Deep/open explanations | form yes, content not always correct |
| Code logic / generalization | fails beyond simple functions |
| Semantic paraphrases | "the drink with the bull" does not reliably find "Red Bull" |
| Multi-turn conversations | weak (hence single-turn in operation) |

**German vs. English:** Helix *understands* English (bilingual pretrain), but was
only instruction-trained in German — English **answers** are noticeably
weaker (more confabulation, partly language mixing). This is **by design**: a
German-primary assistant. For best results, ask in German.

**Methodology:** every capability has a test gate; decisions are made via gates,
not via val loss. Negative results (e.g. embedding retrieval, dirty-data SFT,
an open "explain" archetype) are documented and parked instead of being shipped
prettied up. The next big lever is the same measured ceiling everywhere:
**model size** (upcycle ~2B / from-scratch 3B) — the entire serving stack
(tokenizer, router, tools, RAG, normalizer, gates) carries over directly.

## Project Structure

```text
configs/          YAML configs for model, training, data and experiments
data/             local data, audits and intermediate artifacts
Doc/              original master specs and phase specifications
docs/             current working docs and experiments
eval/             probes, benchmarks and eval documentation
scripts/          download, cleaning, tokenize, training, eval, experiments
src/auralis/      Python package: tokenizer, model, training, inference
tests/            pytest suites
tokenizer/        Helix-v2 tokenizer and quality report
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

On the training server, many jobs run in the Docker container
`auralis-training`. Container paths there typically start with
`/workspace/v2data`.

## Ground Rules

1. The current status is in `STATUS.md`, not in old phase specs.
2. Specs in `Doc/SPECs/` are design history plus reference, but not always
   today's run plan.
3. No large run without audit, tokenize manifest and capability probes.
4. No tokenizer change without a deliberate tokenizer-v2 experiment.
5. New boosters like Knowledge-DNA stay experimental until an ablation
   is unambiguously positive.
