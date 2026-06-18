# Training Waves — from 250M to 1B

Binding procedure for Phase 1. Based on Michael's approach
(2026-04-23): **250M = decision simulator, 1B = production run.**
Don't tip everything over at once.

Lock down beforehand:
- Data mix: [`configs/data/phase1_mix.yaml`](../configs/data/phase1_mix.yaml)
- Canary model: [`configs/model/helix_v2_mid_500m.yaml`](../configs/model/helix_v2_mid_500m.yaml) (preferred) or [`helix_v2_250m.yaml`](../configs/model/helix_v2_250m.yaml) (when token budget is tight)
- Production model: [`configs/model/helix_v2_1b.yaml`](../configs/model/helix_v2_1b.yaml)

## Why waves at all?

A 250M run doesn't make the 1B training faster. It makes it
**de-risked**. The time advantage comes from fewer false starts, not from
fewer FLOPs. That's why the 250M is a **scaled twin** — same
tokenizer, same layer-order logic, same pipeline, just narrower.

## Parameter sizes at a glance

| Config | Role | Params | Rationale |
|---|---|--:|---|
| `helix_v2_100m.yaml` | CPU/test | 135 M | fast unit/smoke tests |
| `helix_v2_250m.yaml` | Canary (when budget is tight) | 261 M | d=768, 12 layers |
| **`helix_v2_mid_500m.yaml`** | **Canary (preferred)** | **517 M** | d=1024, 20 layers — true scaled twin |
| `helix_v2_1b.yaml` | Main run | 954 M | d=1280, 28 layers |

*Note on the 250M naming:* The explicitly desired architecture
(`d_model=1024, 20 layers, n_heads=16, d_head=64, d_ffn=2816, tied, 200k
vocab`) works out arithmetically to ~517 M, because the 200k vocab alone eats 205 M
embedding params. We therefore name it cleanly as "mid_500m" and
keep `helix_v2_250m.yaml` as a fallback for tighter budgets.

## The four rounds

### Round 1 — Infrastructure (short canary)
**Goal:** find bugs, not benchmark.

- **Model:** `helix_v2_250m.yaml` or `helix_v2_mid_500m.yaml`
- **Tokens:** 50 M – 200 M
- **Duration:** 1-3 h on H100, ~$3-10
- **Entry gate:** `inference_compat.py` on a fresh checkpoint PASS
- **Exit gate all YES:**
  - Forward/backward stable, no NaN
  - `grad_norm` healthy (no explosion / no collapse)
  - BF16 autocast stable
  - Checkpoint save + reload OK
  - Val loss falls visibly over 20-50 eval points
  - No health alerts with `level=STOP`

### Round 2 — Data-mix ablation
**Goal:** pick the better data mix — not guess.

- **Model:** `helix_v2_mid_500m.yaml` (or 250m if budget is tight)
- **Config:** [`configs/ablation/mix_variants.yaml`](../configs/ablation/mix_variants.yaml)
- **Three candidates (no more):**
  1. `baseline_75_20_5` — reference point
  2. `de_heavy_70_25_5` — more modern German
  3. `code_heavy_72_20_8` — more structure
- **Tokens:** 0.75-1.0 B per variant (≈ 2.5-3 B total)
- **Duration:** 6-12 h on H100, ~$30-60
- **Exit gate:**
  - one winner by `decision_gates` in `mix_variants.yaml`
  - no per-language regression > 0.05 vs. baseline
  - tiebreak: DE val_loss in a tie

Starter: `python scripts/pretrain/mix_ablation.py`

### Round 3 — Winner validation
**Goal:** final go/no-go for 1B.

- **Model:** same canary
- **Mix:** round-2 winner
- **Tokens:** 1.5 – 2.0 B
- **Duration:** 12-18 h on H100, ~$60-90
- **Exit gate:**
  - val_loss trend shows further improvement (not just noise)
  - per-language val_loss falling evenly
  - cumulative health alerts < 5 (WARN), 0 STOP
  - baseline score (50 questions) shows a measurable learning effect

### Round 4 — 1B main run
**Goal:** productive Phase 1, no open questions left.

- **Model:** `helix_v2_1b.yaml`
- **Mix:** round-3-validated mix (probably `baseline_75_20_5`)
- **Tokens:** 21 B (actually available) or 25 B (brief target)
- **Duration:** 3-4 weeks H100 / ~16 days H100 with good kernels
- **Cost:** $500-800

## What is **not** allowed in which round

| Round | Taboo |
|---|---|
| 1 | New hyperparameter ideas, LR sweeps, data experiments |
| 2 | Seq-length sweeps, architecture changes, additional sources |
| 3 | Mix changes, LR changes |
| 4 | Anything except exactly the validated setup |

## What gets documented between rounds

After each round:
1. `MANIFEST.yaml` in the checkpoint folder (automatically by `run_report.py`)
2. Run `scripts/eval/regression_dashboard.py --ckpt-dir …`
3. Record the decision in `HISTORY.md` with date + reference to the dashboard

## Downloads the pod additionally has to pull

Not on the NAS:
- `HuggingFaceFW/fineweb-2` `eng_Latn` (for EN top-up)
- `bigcode/the-stack-v2` (replaces StarCoderData)

Scripts for that are prepared in the download modules
(`scripts/data/download_english.py --sources fineweb2_en`,
`scripts/data/download_code.py --sources the_stack_v2`). None of it
starts locally — only on the pod with a gigabit HF connection.
