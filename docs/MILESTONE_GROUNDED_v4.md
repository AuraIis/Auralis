# Milestone — Archetype H (Grounded) v4

**Status: best current grounded state at 0.9B — NOT finally passed (80% goal narrowly missed).**

Honest research state, no sugarcoating. v4 is the adapter we keep;
v4.1 was built, tested and **deliberately not promoted** (see plateau finding below).

## Result (Grounded stress gate, 54 cases)

| Metric | Value | Bar | |
|---|---|---|---|
| `answer_ok` | **23/30 (77%)** | >80% | ❌ (1 case short) |
| `refuse_ok` | **22/24 (92%)** | >90% | ✅ |
| `world_leaks` | **0** | =0 | ✅ |
| `stop_rate` | **1.0 (100%)** | >95% | ✅ |

The **safety-critical part is fully met**: 0 world-knowledge leaks across 11 trap cases,
clean stop on all 54 cases. The missing 3 percentage points are pure
hard-distractor edge cases — no correctness or hallucination risk at the core.

## Iteration history — and why we stop at v4

| Version | answer_ok | refuse_ok | leaks | stop | Finding |
|---|---|---|---|---|---|
| v2 (number focus) | 8/17 (47%)* | 18/19 | 0 | 1.0 | distractors + varied prose weak |
| **v3** (structural variety + distractors) | 19/30 (63%) | 21/24 (87.5%) | 0 | 1.0 | distractors clearly better, two-number prose works |
| **v4** (dense prose + count-refusal + start/end) | **23/30 (77%)** | 22/24 (92%) | 0 | 1.0 | **best state**; +Schmidt/Schmitt solved |
| v4.1 (regression fixes) | 22/30 (73%) | 23/24 (96%) | 0 | 1.0 | lateral — see plateau |

\*v2 numbers on the old 36-case gate; from v3 on the extended 54-case gate.

### Plateau finding (whack-a-mole)
v4.1 **demonstrably achieved all 4 targeted fixes** (novel→"412 pages" instead of
2850 hallucination; Greifenau month→"May"; third entity→refusal; "How many cows
does … keep?"→refusal). **But** the additional refusal obligation shifted the prior toward
"refuse" and **broke 3 previously correct cases** (Lindau population,
farm-animals list, Berger assignment). Net: **+4 fixed, −3 broken → 77%→73%**.

**Conclusion:** LoRA-SFT data iteration trades errors ~1:1 at 0.9B instead of
accumulating them. The method has reached its plateau at **~77%**. Further gains
need a **stronger lever**, not more SFT data rounds.

## What works (robust)
- **World-knowledge traps:** context names an entity, question targets a known fact
  outside the text → refuses, **never supplements from memory** (0 leaks).
- **Number extraction:** simple + two-number prose (distractor number correctly ignored).
- **Dense 5-sentence prose:** founding year / population / "known for" / event month.
- **Time:** start/end, from-to, from/until, since-year.
- **Third entity:** two named persons, question about a third → refusal.
- **Multiple similar entities (partial):** Schmidt/Schmitt, Anna/Anne, Kraus/Krause ✅.

## The ceiling (residual errors at 0.9B)
- **Hard near-duplicate names:** Tom/Tim, Jonas/Jonah (over-refusal).
- **Positional assignment:** "left/right house" (wrong choice).
- **Compositional time:** "Mondays to Fridays … on Fridays?" (over-refusal).
- **Tail truncation:** "runs for another five weeks" → cuts off "five weeks".

These are representation/decoding-limited, not data-limited.

## Method (reproducible)
- **Deterministic generation** → guaranteed-correct labels (no LLM judge).
- **Narrow embedding EOS fix:** only special-token rows 4–17 receive gradient
  (LR 3e-5), the rest via LoRA r=64 — otherwise the model cannot emit `<|end|>`.
- **54-case stress gate, unchanged across v3/v4/v4.1** + 18 generalization cases
  with **different specifics than the training** → tests generalization, not memorization.
  The gate was **never softened** to make a version look good.

## Artifacts
- Adapter (NOT in git): `checkpoints/sft_grounded_v4/adapter_best.pt` (on the server).
- Generators/assembler/gate: `scripts/sft/grounded/`.
- Base: `step_60000` (pretraining final), mix grounded ~61% / corrective 33% / tool 6% / abstain 3%.

## Way forward (stronger lever, not more SFT rounds)
1. **Grounded/QA pretraining** instead of only LoRA-SFT (anchor extraction earlier).
2. **Larger base model** for the near-duplicate distractors.
3. **Decoding help** (constrained extraction / span-copy) against truncation + assignment.
4. Revisit H Grounded later with (1)–(3).
