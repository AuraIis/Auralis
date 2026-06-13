# Experiment (future / v3+): diffusion-draft → AR-refine hybrid

**Status: SPECULATIVE research note, not a roadmap item.** Do not start before the
current pretraining + SFT + tool-use levers are exhausted. This is a "would it be
worth a test someday?" design, written down so the idea isn't lost — with explicit
kill criteria so it can't quietly eat months.

## Hypothesis (with the roles corrected)

Naive framing: "a diffusion model thinks, the AR model writes." That role split is
likely **wrong** — autoregressive (AR) models are currently the *better reasoner*
(chain-of-thought maps onto sequential generation). Diffusion's real edge is
**global structure + revision + parallel refinement**, not reasoning.

Corrected hypothesis:
> A small **discrete text-diffusion model drafts a coarse global structure/skeleton**
> (it sees the whole at once and can revise); **Helix (AR) refines that draft into
> fluent, correct surface text**. Each model does what it is actually good at.

The bar this must clear is NOT "beats naive AR" — it is **"beats a well-prompted
AR-only baseline (plan-then-write / CoT) at equal or lower total compute."** If it
can't beat that, it is not worth the second model.

## Where it might genuinely help — and where it won't

Likely helps: long-form with global coherence, documents/reports with structure,
code with an up-front architecture, constrained/infilling tasks (diffusion's
bidirectional context is a real advantage here).

Likely does NOT help: short open chat, pure step-by-step reasoning, anything where
a single AR pass with CoT already nails it. Be honest about this up front.

## Architecture spectrum (the interface is the whole problem)

**Option A — text handoff (cheap, start here).** Diffusion emits a *text* draft
(outline / skeleton / noisy first pass). Helix conditions on it:
`<|draft|>…<|end|>\n<|user|>…\n<|assistant|>…`. No joint training; both models stay
independent. Risk: if the draft is plain text, why not just let Helix produce it via
CoT? The win must come from diffusion contributing structure AR wouldn't generate
the same way (global layout, revision passes). Cheap to falsify — do this first.

**Option B — latent handoff (powerful, only if A shows promise).** Diffusion emits
latent plan vectors; Helix cross-attends to them. More elegant, potentially the real
payoff — but needs the two models to **share a representation**, i.e. joint training.
This is exactly where most such systems fail. Do NOT start here.

## Staged plan (each stage gates the next; cheap → expensive)

- **Stage 0 — baseline first (no new model).** Implement and measure the AR-only
  `plan-then-write` and CoT baselines on a fixed eval set. *This is the number the
  hybrid must beat.* If AR-only-with-planning is already strong, the bar is high —
  good to know before spending anything.
- **Stage 1 — small diffusion drafter.** Train a small (~100–250M) masked-diffusion
  LM (LLaDA-style) on the **same 200k Helix tokenizer + a slice of the same corpus**
  (tokenizer compatibility is non-negotiable — avoids a re-encode bridge). Goal: it
  produces usable coarse drafts, nothing more.
- **Stage 2 — teach Helix to refine.** Build draft→refine SFT data by taking good
  target outputs and *corrupting/coarsening* them into "drafts" (mask spans, drop
  function bodies, shuffle/outline). Helix learns `(draft, instruction) → polished
  output`. This stage is useful **even standalone** (Helix gains a refine/edit skill).
- **Stage 3 — connect (Option A) + evaluate** end-to-end vs the Stage-0 baseline on
  the structured tasks above. Measure quality AND total compute (drafter steps + AR).
- **Stage 4 — latent handoff (Option B), joint training.** ONLY if Stage 3 beats the
  baseline with headroom. Expensive; treat as a separate project.

## Evaluation

- Fixed held-out set weighted toward the "likely helps" task types.
- Metrics: task-appropriate (exec-pass for code, structure/coherence + human/LLM
  judge for long-form), plus **total inference compute** (diffusion is many denoising
  steps — a quality win at 5× compute is not a win).
- **Mandatory baselines:** AR-direct, AR + CoT, AR + plan-then-write. The hybrid is
  only interesting if it beats the *best* of these at ≤ comparable compute.

## Kill criteria (be ruthless)

- Stage 0 baseline (AR + planning) already saturates the eval → **stop**, no room.
- Stage 1 drafter can't produce coherent skeletons at small scale → **stop**.
- Stage 3 hybrid ≤ best AR baseline at equal compute → **stop** (Stage 2's refine
  skill may still be worth keeping standalone).
- Total compute to match AR > ~2× → **stop** unless a specific high-value niche.

## Cost & sequencing

Two models, two inference passes, and (for Option B) joint training. For a
0.9B-budget, German-primary project this is a **research side-quest**, explicitly
**after** the proven accuracy levers (data, SFT, tool-use) are done. Realistic
earliest: a Helix v3 exploration, and even then Stage 0–2 first.

## Why write it down anyway

Stage 2 (Helix learns to refine/edit a draft) is **independently valuable** and
low-risk — a good edit/refine skill helps tool-use, self-correction, and the Hub,
regardless of whether the diffusion drafter ever ships. That alone may justify a
small slice of effort someday.
