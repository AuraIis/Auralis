# Blueprint — Domain adapters (DoRA: Math / Logic / Code) (Helix v2)

> **Status:** Design / decided-with-reservation (Michael + GPT + Claude, June 2026). Not implemented.
> **Phase:** **AFTER** tool use **and** after code annealing. See `BLUEPRINT_TOOL_USE_VERIFIER.md`
> (stage 5) and `ZUKUNFT_BACKLOG.md` phase 3–5.
> **Core principle (non-negotiable):**
> *An adapter amplifies latent ability — it installs no new one.*

---

## 1) Idea
**Freeze** the main model, train one small adapter per domain that amplifies the *ability*
(not the knowledge):
- `dora:math` — calculation-path structure, step by step
- `dora:logic` — reasoning, case distinction
- `dora:code` — code structure, tests, repair patterns

Adapters are trained separately and loaded/swapped per task (or routed).

## 2) Why DoRA — and not LoRA or MoRA
From the v1 lessons (**L-002**): *LoRA learned patterns, not facts.* Hence the clean separation:
| Technique | for what | Helix use |
|---|---|---|
| **DoRA** (weight-decomposed) | **patterns / skills** | math calculation path, code structure, logic → **right here** |
| **MoRA** (high-rank) | **facts / knowledge** | knowledge injection (separate path, not this document) |

Math/logic/code ability is a **skill pattern** → DoRA fits. Facts would need MoRA.

## 3) The catch — order is mandatory, not style
An adapter turns weights that are **already there**. If the ability is missing from the base *latently*,
the adapter has nothing to amplify.

| Adapter | latent base in the current base? | verdict |
|---|---|---|
| `dora:code` | **0% real code in pretraining** (only code-as-prose) | **locked** until code annealing — otherwise built on sand |
| `dora:math` | thinly present (was in the mix) | moderately amplifiable, better worth it after annealing |
| `dora:logic` | thinly present | moderately amplifiable |

→ **Code annealing (Python-Edu is ready) is the prerequisite for `dora:code`.**
Order reversed = burned time.

## 4) Relationship to tool use (important — not competing)
DoRA **does not replace the verifier.** Tool use makes answers *verifiable* (correctness from
the outside); DoRA makes the model *more fluent/structured* in the domain. Order:
1. **Tool use first** (highest lever at 0.9B, correctness from the outside).
2. **Annealing** (latent ability into the base).
3. **DoRA afterward** (amplifies the latent, including "calls the tool cleanly").

A DoRA math adapter should ideally **learn the tool-use behavior too**, not try to
replace mental arithmetic.

## 5) Targeting on the hybrid architecture (implementation core)
Helix is not a pure transformer stack. DoRA hangs off **linear projections** — those exist
in all three layer types, but have to be wired up in the trainer:
| Layer type (count) | adaptable linears |
|---|---|
| Mamba-2 (6×) | `in_proj`, `out_proj` (possibly `x_proj`/`dt_proj`) |
| GLA (16×) | `q_proj`, `k_proj`, `v_proj`, `g_proj`, `out_proj` |
| Sparse-Attn (6×) | `q/k/v/o_proj` |
| MLP (SwiGLU, all) | `gate/up/down_proj` |

**Open design decision:** target all projections (max capacity, more params) vs.
only attention/GLA projections (common, leaner). **Ablation first, then default.** Embedding
(tied 200k) stays frozen.

## 6) Training recipe (starting values, to be calibrated)
```
base:        frozen (no grad)
adapter:     DoRA, rank r ∈ {8,16,32} (ablate), alpha = 2r
target:      see §5 (default decision via ablation)
lr:          ~1e-4 (higher than full-SFT, since only the adapter learns)
data/adapter: domain-pure (math-only / logic-only / code-only), 5–20k each
              — math/logic derivable from our reasoning slice
eval:        disjoint val splits per domain (L-002: avoid the memorization trap)
             + negative guard: general benchmarks must NOT drop (adapter off = baseline)
```

## 7) Multi-adapter at runtime
- **Variant A (simple, first):** explicit mode — user/caller selects the adapter
  (`--adapter math`). No router risk.
- **Variant B (later):** small router/classifier selects the adapter from the question.
  Its own error source (wrong adapter) → only once A is in place and it's worth it.
- Adapters are small → several in RAM, fast swapping; do **not** merge into the base
  (otherwise modularity is gone).

## 8) Honest ceiling at 0.9B
- DoRA mainly raises **form/fluency** in the domain — no jump to "strong reasoning".
- Effect **gated on base quality**: small before annealing, larger after.
- No substitute for scaling. DoRA is a polish, not a foundation.
- Realistic expectation: measurable, *moderate* lift on domain benchmarks with held
  general level — no more, no less. "Measure first, then decide."

## 9) Success criteria
- Domain benchmark (e.g. mmlu_de math slice / GSM8K-de / logic probe) **with adapter > without**.
- General benchmarks (de/en MMLU/ARC/HellaSwag) with adapter **≥** without (no general loss).
- Adapter size ≪ base; load/swap < 1 s.

## 10) Prerequisites (gates) — check off before start
- [ ] Tool use stage 1–2 green (math tool proves the harness)
- [ ] Code annealing done (for `dora:code`) — `math`/`logic` possibly earlier
- [ ] DoRA targeting wired into the v2 trainer (§5) + ablation rank/target
- [ ] disjoint domain val splits built (L-002)
- [ ] negative-guard eval (general benchmarks) set up as a mandatory gate

---
*Order gated. `dora:code` without code annealing = §1 core principle violated = building on sand.
Reminder of the 500M dead end: layer before foundation = garbage.*
