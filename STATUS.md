# STATUS - Auralis v2

As of: 2026-06-07

This is the current short-form truth for the repo. If old phase plans,
April/May status states, or specs contradict it, this file takes precedence first,
then the reports from 2026-05-29, then the respective working docs.

## Update 2026-06-07b — Modular Adapters (LoRA) proven: Skills WITHOUT collateral damage

LATEST STATE. Second architecture milestone today: Helix can receive modular skills via
adapters WITHOUT permanently damaging the base — the core of the modular vision.

Problem (measured): Full-FT calibration forgets tool-use/facts after ~50 steps
(catastrophic forgetting). Two full-FT rounds (calib v1+v2) delivered NO checkpoint
with honesty AND retention at the same time (oscillates / `12+15` breaks / Einstein over-refuses).

Solution: LoRA adapter on a FROZEN base (`checkpoints/tool_sft_v12/step_600`).
- `src/auralis/adapters/lora.py`: LoRA/DoRA layers, inject into 188 GLA/Attn/FFN modules
  (Mamba `mamba_ssm` kernel excluded — reads `.weight` directly, un-wrappable), 1.2% trainable,
  alpha control (`set_adapter_scale`), save/load (~MB instead of 10 GB).
- Trainer integration (`smoke_sft_de --adapter-r`), adapter gate (`calib_gate --base/--alpha-sweep`).
- Two PEFT pitfalls fixed + documented in the code: (1) DoRA reconstructs full weights
  -> memory-hungry -> LoRA; (2) grad-ckpt OOMs with a frozen base -> `enable_input_require_grads`
  (PEFT trick: make embedding output grad-requiring) -> 47GB -> 14.7GB.

ALPHA SWEEP (honesty adapter `honesty_adapter_v1` on step_600, held-out bank, n=60 invented):
```
alpha   inv-abstain (honesty)   people-answer   math-tool
0.00    3%   (= exactly Base)    5/5             5/5      <- Control: adapter off = base
0.25    18%                      5/5             5/5
0.50    95%                      5/5             5/5      <- SWEET SPOT (recommended alpha)
0.75    100%                     5/5             5/5
1.00    100%                     5/5             5/5
```
> **Adapter OFF = exactly base. Adapter ON = controllable additional behavior. alpha=0.5 delivers
> 95% abstention WITHOUT tool or fact loss.** What full-FT could NOT do TWICE (honesty
> OR retention), the dosable adapter does in ONE run, on a guaranteed-intact base.

This validates the modular roadmap: honesty-LoRA now, code-LoRA after annealing,
knowledge-MoRA later. Open (next session): inference path base+adapter@0.5 (deployable).

## Update 2026-06-07 — Tool-Use Math END-TO-END (verified computation instead of guessing)

LATEST STATE. Helix now solves arithmetic via a verified external tool,
instead of guessing in its head. Structurally solved: "12 + 15 = 12" (guessing) becomes
`<tool:python>print(12+15)</tool>` -> executor 27 -> "12 + 15 equals 27."

Built (all KEY-FREE / self-generating; the calculator is the ground truth):
- `scripts/sft/tool_harness.py` — AST-whitelist calculator (no RCE, selftest 14/14) +
  generation loop with `</tool>` stop sequence + result injection + resume.
- `scripts/sft/gen_tool_traces.py` — tool-SFT traces (modes call_only/full, --simple-rebump).
- `scripts/sft/tool_gate.py` — DUAL gate (math->tool, facts->no tool) + end-to-end
  (`--mode full`: result_usage_rate, answer_numeric_match), type breakdown, best-by-GATE.
- `smoke_sft_de.py` — `<result>` block MASKED from the loss (token-exact verified) ->
  model does NOT learn to fake results.

Phases (each gated, best-by-gate instead of val_loss — val_loss was demonstrably misleading here):
- Phase 1 (call_only): tool call + stop. step_400: tool 100% · false_tool 0% · parse 97% · correct 68%.
- Phase 1.1 (enriched translation traces, language->formula). step_500: correct 93%.
- Phase 2 (full, result injection -> final answer).

PROMOTED: **`checkpoints/tool_sft_v12/sft_smoke_step_600.pt`**
  correct **94%** · parse **100%** · fake_result **0%** · false_tool **0%** · answer_match **85%**
  Buckets: percent 24/24 · word 21/21 · speed 10/10 · english 7/7 · time_unit 16/17 · simple 16/21 (76%)

Honest limits: in-distribution (trained task types, new numbers; freely phrased
questions untested); `simple` bucket weak due to sqrt/`hoch 2` (= operator mapping, not +-*/);
answer_match measured conservatively (German decimal comma "59,5" vs executor "59.5" counts as
mismatch). Tool-use adds NO knowledge — knowledge gaps remain (annealing/scaling).

Next session (NOT pulled forward): (1) normalize gate number comparison (comma/dot/trailing-0),
(2) split simple bucket into basic/advanced + extract failure cases, (3) targeted sqrt/power traces,
(4) then calibration/R-tuning (key-free, self-labeling against gold/MC/executor).

## Update 2026-06-06 — 1B foundation ran + SFT (behavior) + reasoning slice

LATEST STATE. Takes precedence over ALL sections below (incl. 2026-05-31). The
1B policy/preflight gates further down are satisfied and thus historical — the
foundation run HAS run.

Where we stand:

- 1B foundation warmstart v3 RAN through step 50000
  (`checkpoints/pretrain_1b_bilingual_de55_en45_foundation_warmstart_v3/step_50000.pt`).
  Healthy training, language + fact grounding demonstrated (knowledge profile n=57:
  history/geography strong, science/translation weaker).

- SFT v1 (~32k diverse DE+EN, gpt-4o-verified [269 hallucinations caught],
  decontaminated) RAN. From the base that could barely answer, an
  ANSWERING assistant emerged (Vienna/Madrid correct, clean stopping via
  eos-loss-weight 2.0). SFT teaches FORM, not KNOWLEDGE — confirmed by benchmarks.

- Benchmarks (own MC log-likelihood runner, n=300): Helix-SFT beats
  SmolLM2-360M + TinyLlama-1.1B on mmlu_de; Qwen's MMLU lead shrinks from ~22
  (EN) to ~7 (DE). Language strategy (200k vocab, de55/en45) pays off measurably.
  Absolute values low (under-training/size signal). Details:
  `docs/PROJEKT_STAND.md`.

- Reasoning slice built + verified: 2500 DE (natively generated) + 2500 EN
  (GSM8K converted). gpt-4o-verify 100% on math -> ~9.4% wrong math
  caught/corrected. Clean in the Helix format.

- SFT v2 RUNNING (36.6k = SFT v1 + reasoning slice, ~13.5% reasoning, 1 epoch,
  bucket+grad-ckpt). val low point at step 2100 (val 2.580 — better than v1 ~2.81),
  then overfit uptick. Keeper: `checkpoints/sft_v2/sft_smoke_step_2100.pt`.
  Quicktest + re-benchmark next.

Direction decided afterward (triple-triangulated Michael+GPT+Claude):

1. Tool-use FIRST (math tool harness): small model learns to VERIFY instead of guess.
   Spec: `docs/BLUEPRINT_TOOL_USE_VERIFIER.md`.
2. Annealing (FineWeb-2-DE/Cosmopedia/Python-Edu already loaded) including code.
3. DoRA math/logic/code on annealed base. Spec:
   `docs/BLUEPRINT_DOMAIN_ADAPTERS_DORA.md`.

Order gated (`ZUKUNFT_BACKLOG.md`). Core principle: adapter amplifies
the latent, installs nothing -> code-DoRA locked until code annealing.

Infra note: `data/` and `checkpoints/` live ONLY on BITBASTION
(`/workspace/v2data`, 36T), not on the Windows box (gitignored, too large).
Only code syncs (U:\ <-> container). This is intentional, not data loss.

## Update 2026-05-31 — Edu data filter (German) + Multi-GPU

This block is the latest state and takes precedence over the older 1B-canary/500M
sections below.

Context: The bilingual 1B ramp (de55/en45) ran through step ~3400 (best.pt),
the learning behavior was disappointing. Clean diagnosis (not from the gut):

- NOT the eval (Qwen-2.5 on the same probes = sensible, 37/50).
- NOT the architecture (all-plain-attention control ~ on par with Helix
  up to step 300).
- Rather: under-training (~3.4B tokens ~ 16% Chinchilla) AND a
  quality-inverted German mix (the weakest source got the
  most budget).

Data quality (FineWeb-Edu methodology for German, rebuilt):

- LLM annotation 0-5 on educational value. Judge: `qwen3-235b-a22b-2507` via
  OpenRouter (non-thinking, ~40x cheaper than gemini-3.5-flash, stricter and
  more accurate on web text). 12k labels, ~1 EUR.
- Cheap classifier: frozen multilingual-e5-large + Ridge head + calibrated
  threshold. Val Pearson 0.866, Keep-F1 0.872.
- Corpus filter @ threshold 2.0: fineweb2_de ~38% kept, wikipedia_de in full,
  german_commons DROPPED (~2-5% keep, EuroParl/OCR fragments).
- German-v2 = edu-filtered fineweb2_de + wikipedia_de ~ 2.0B high-quality
  tokens (covers the ~1.8B DE need of the foundation run without repetition).
- Config: `configs/data_paths.curated_v2_german.yaml` (re-tokenizes only DE).

Multi-GPU / DDP (new, PR #1, branch `feat/multigpu-ddp`):

- DistributedDataParallel in the trainer, strictly gated on `WORLD_SIZE>1` ->
  single-GPU path bit-identical (verified: py_compile + dry-run).
- DDP-agnostic checkpoints (no `module.` prefix -> single-GPU loadable),
  no_sync during grad-accum, rank-0 eval+barrier, global stop via all_reduce.
- torchrun launcher: `scripts/ops/run_pretrain_multigpu.sh`.
- Measured throughput: 12.9k tok/s/GPU (1B, Blackwell). Full 1B (~20B tok):
  ~18 days 1 GPU, ~5 days 4 GPU. Not yet validated on real multi-GPU
  (test box has 1 GPU) -> short 2-GPU run on RunPod before the long haul.

Infra decision: training stays on BITBASTION (1 GPU, free) for the
foundation run; for fast/large runs RunPod multi-GPU (spot, thanks to
resume), NOT Colab (compute units + session limits unsuitable).

Scaling sources (if more German needed): RedPajama-V2-de (3T modern,
with quality signals) + more fineweb2_de, edu-filtered. german-commons
discarded (OCR-historical, see L-020). multitask_german_32k secured for the
later SFT phase.

Open / next:

1. fineweb2_de full scoring running (~38% keep) -> then tokenize German-v2
   (back up old `german.bin`, only DE fresh via `curated_v2_german.yaml`).
2. Then foundation warmstart from ramp `best.pt` on the better data.

## Short Decision

1B is not started yet.

The safety framework for a 1B canary is now built, but the preflight
is not yet green. The next real step is not another 500M-SFT
patch, but the final, audited 1B clean/tokenized mix.

Current 1B preflight:

- Report: `reports/auralis_1b_readiness_preflight_v2_2026-05-29.md`
- Result: `ready_to_launch: False`
- Eval prompts: 70
- Training units scanned: 382,763
- Hash collisions: 0
- Substring hits: 0

Interpretation:

- The leak/disjointness side is currently clean.
- The start is blocked because the final 1B data mix is not yet reliably
  entered as a clean/tokenized mix and released via preflight.
- `configs/data_paths_1b_samples_container.yaml` must point to real,
  existing 1B clean and tokenized paths before start. `.bin` token files need
  the matching `.idx`.

## Binding 1B Policy

The 1B run may only start when data and gates are green beforehand.

Binding files:

- 1B Readiness Gate: `eval/auralis_1b_readiness_gate_v1.yaml`
- Frozen Target/Retention Gate: `eval/sft_response_frozen_target_retention_v2.yaml`
- 1B Preflight Config: `configs/eval/auralis_1b_readiness_preflight.yaml`
- 1B Preflight Script: `scripts/eval/one_b_readiness_preflight.py`
- Guarded Canary Config: `configs/training/pretrain_1b_canary_readiness.yaml`
- Guarded Canary Runner: `scripts/ops/run_pretrain_1b_canary_readiness.sh`

Promotion rule:

- Target must pass.
- Retention must have 0 regressions.
- A single retention regression means: not promotable.
- Eval probes are not loosened to make a run green.
- New probes only added append-only.

Important target/retention axes:

- Photosynthesis as a real concept, not just a keyword hit.
- Faust/Goethe as a confident known fact.
- Bonn formerly vs. Berlin today.
- Answer known facts, refuse invented entities.
- Do not confuse Goethe with `Mein Kampf`.
- Do not refuse Faust I due to overdominant honesty training.

## 500M State

No tested 500M checkpoint is promotable.

Frozen-Gate-v2 results:

| Checkpoint | Target | Retention | Promotable |
|---|---:|---:|---:|
| `v8_safe` | 8/25 | 18/25 | no |
| `hybrid_v1_40` | 9/25 | 17/25 | no |
| `hybrid_v12_bridge_60` | 10/25 | 17/25 | no |
| `hybrid_v12_repair_v2_80` | 9/25 | 17/25 | no |

Current conclusion:

- `v8_safe` only remains the most stable relatively speaking, because retention
  breaks the least.
- Hybrid/v12 moves photosynthesis/Faust partly, but loses retention.
- Further 500M mini-patches are diagnostic work, not promotion work.
- 500M must not be treated as solved or production-ready.

## Diagnosis

The current errors are not simple prompt, score, or loss problems.
The model shows interference:

- Photosynthesis/Faust can be improved locally.
- In doing so, Bonn/Berlin, known-fact retention, or safe counter-facts tip over.
- Honesty/refusal is too dominant on some known facts.
- Invented entities are sometimes nevertheless ascribed details.

This argues against further small repair SFTs on 500M and in favor of a cleanly
weighted 1B pretrain/SFT mix.

## Adaptive / Live Gates

The adaptive training layer can run the v2 frozen gate live alongside:

- Adaptive Frozen-Gate Bridge: `src/auralis/adaptive/frozen_gate.py`
- Adaptive Trainer CLI: `scripts/train/adaptive_curriculum.py`
- Live trace: `<output-dir>/frozen_gate_trace.jsonl`

New live metrics:

- `frozen_target_pass`
- `frozen_retention_pass`
- `frozen_target_failures`
- `frozen_retention_failures`
- `frozen_promotable`

Test status per handoff:

- local adaptive tests: green
- container smoke for frozen-gate live bridge: green
- mini smoke with 500M-v8 only checked the wiring, not content scores.

## Next Steps

1. Build final 1B clean mix.
   - Fill `cleaned.german`, `cleaned.english`, `cleaned.code` with real paths.
   - Enter tokenized paths.
   - Ensure existence of `.bin` and matching `.idx`.

2. Audit the data mix before tokenizing.
   - modern German QA/facts stronger than in the 500M.
   - cap refusal/honesty, do not let it dominate.
   - explicitly plan for confident-correct facts.
   - dose books low, do not use as the main knowledge carrier.

3. Run preflight again:

```bash
python scripts/eval/one_b_readiness_preflight.py \
  --config configs/eval/auralis_1b_readiness_preflight.yaml \
  --output-json reports/auralis_1b_readiness_preflight_v2_$(date -u +%F).json \
  --output-md reports/auralis_1b_readiness_preflight_v2_$(date -u +%F).md
```

4. Only when `ready_to_launch: true`, start the canary:

```bash
bash scripts/ops/run_pretrain_1b_canary_readiness.sh
```

## Stop Criteria in the 1B Canary

Stop immediately or do not promote when:

- Retention gets even a single error.
- Bonn historically drops out.
- Berlin today tips over.
- Goethe/Faust is refused or becomes wrong.
- Photosynthesis only sounds good but connects sugar/oxygen/plants/light
  wrongly.
- invented entities are embellished.
- Target does not improve even though loss drops.

## Current References

- `reports/auralis_1b_readiness_plan_2026-05-29.md`
- `reports/codex_handoff_1b_readiness_v2_2026-05-29.md`
- `reports/auralis_1b_readiness_preflight_v2_2026-05-29.md`
- `reports/learning_neuro_hybrid_v12_2026-05-29.md`
- `docs/DOCS_INDEX.md`
- `eval/README.md`

## What No Longer Counts as the Current State

- `STATUS.md` as of 2026-05-17 as the run plan.
- The old `pretrain_mix_v4_boosted_500m` as the current main path.
- The April plan `curated_40b` as the active main mix.
- Old paths like `tokenized/phase1` or `checkpoints/phase1_pretrain` as the
  default for new runs.
- SFT as a repair for a weak/noisy base model.
