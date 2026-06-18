# Post-mortem: "Measure first, then suspect the data"

**Core lesson in one sentence:** Almost everything that looked like *"the model isn't
learning / the data is bad"* with Helix v2 was in truth **broken or misleading
measurement, a learning rate set too high, a guard bug, or raw decoding** — and we
were repeatedly close to wrongly attributing it to the training data. Only *clean
measurement* revealed the true state.

Context: Helix v2, ~954M hybrid (6× Mamba-2 + 16× GLA + 6× Sparse-Attn), bilingual
DE/EN, 200k vocab, warm-start continued pretraining. This document is the honest
chronicle of the misdiagnoses, so they don't repeat.

---

## The cases (each: symptom → first (wrong) guess → true cause)

### 1. Apparent val-loss regression on warm-start
- **Symptom:** After the LR peak, val_loss rose.
- **First guess:** The new German data is to blame.
- **Test:** Reproduced on the old **and** new pool → **not the data**.
- **True cause:** LR too high for a warm-start (fresh AdamW + re-ramp to the
  from-scratch peak destabilizes an already-converged checkpoint).
- **Fix:** lower continuation LR (3e-5), short warmup.

### 2. The "1.172 → 1.222" regression = invalid comparison
- **Symptom:** bpb_de seemingly worsened from 1.172 to 1.222.
- **First guess:** The model got worse.
- **True cause:** The old "best" 1.172 was measured on a **different val set**
  than the new 1.222. Apples to oranges.
- **Fix:** Step-0 eval (load checkpoint, **without** training, identical set) → true
  baseline → no real regression.

### 3. The measurement itself was the main culprit (broken multiple times)
- `tokens_per_byte` was guessed (0.2338) instead of measured (**0.176**) → bpb_de ~33%
  inflated.
- The eval was **stochastic** (stateful RNG kept running → each eval drew *different*
  tokens) → the "curve" was sampling noise, not a clean signal.
- The German val was only the **Wikipedia tail** (not representative); the
  English tail was accidentally trivial → the bpb **gap looked like 3.2** (mirage),
  really ~**1.04**.
- **Consequence:** "regression" and "huge language gap" were **measurement artifacts**.
- **Fixes:** deterministic eval (`reset_rngs`), measured tokens/byte,
  representative val sampling, step-0 diagnosis (kernels on/off).

### 4. The emergency brake stopped the run wrongly (step 4250)
- **Symptom:** Auto-stop "val_regression".
- **First guess:** The model is regressing.
- **True cause:** The guard counted "no new best value" as a regression (instead of
  *real* consecutive increases, with no tolerance). The model was **healthy**.
- **Fix:** real consecutive-increase logic + `min_delta`; `error_if_nonfinite`; hard tokenizer check.
  After that, resume → clean up to ~35k.

### 5. "Model learns no facts" (Munich flip) = decoding artifact
- **Symptom:** Greedy generation: "capital = Munich→Berlin→Munich" across
  checkpoints.
- **First guess (also shared by two external reviewers):** Knowledge is not anchored
  → data/scaling problem.
- **True cause:** A **rigorous margin measurement** (`NLL(wrong) − NLL(right)`,
  multiple distractors, 5 categories) yielded: **history 100%, geography 86%, overall
  72%** (2-way even 87.5%). The knowledge **is there**. Greedy only measured the
  **answering behavior** (drifts during free generation), not the internal knowledge.
- **Correction:** *not* "knowledge missing", but "**knowledge present, decoding/answering still
  raw**".

---

## Separate the terms cleanly (so we don't conflate them again)
| Measurement | measures |
|---|---|
| **Recall margin** `NLL(wrong)−NLL(right)` | **knowledge in the model** |
| **Top-k after fact prompt** | **retrieval proximity** (does the right candidate come out on top?) |
| **Greedy generation** | **answering behavior** during free generation |
| **SFT / instruction** | **format & controllability** (≠ factual knowledge) |

## Corrected milestone view (as of ~35k/50k)
- **A — Stable training:** ✅ confirmed (val↓, grad stable, 0 alerts)
- **B — Language learning (DE/EN fluent, separated):** ✅ confirmed
- **C — Instruction following:** open (SFT phase)
- **D — Factual anchoring:** ✅ surprisingly strong (history/geography), only
  **science weak (29%)**
- **E — Knowledge-DNA:** unproven, *optional* boost — not needed to
  talk about factual anchoring

## What actually concerns the DATA
Only **science facts** are genuinely weak (Au/Ag, Jupiter/Mars, boiling points). There
**"more science-dense data"** is the correct, *specific* lever — not "more
data" in general. Everything else was measurement/tooling/decoding.

## Concrete safeguards (already implemented)
- Deterministic eval · measured tokens/byte · representative val · step-0 diagnosis
- Corrected regression guard (`min_delta`, real consecutive-increase logic) · non-finite protection
- Rigorous fact-recall battery (margin + top-k, multiple distractors, 5 categories)
  → mandatory metric from step 50k, not eyeballing.

**Mnemonic for the team:** Before a bad number "is the data" — check whether the
number actually measures what you think it does.
