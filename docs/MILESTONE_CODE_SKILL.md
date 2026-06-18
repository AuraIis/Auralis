# Milestone — Code skill (Helix 0.9B)

**Status: Door opened, but code cleanly proven as the 0.9B limit. v3 is the code adapter.
Comes back on the table only at 3B/7B.** Honest research state including two negative findings.

## Core result
After the code SFT, Helix 0.9B writes **syntactically perfect, runnable code that stops cleanly
and recalls known patterns** — but it does **not generalize new logic** and **cannot
correct itself from test feedback**. That is a representation-capacity limit, parallel to the
grounded ceiling ([[grounded-archetype-h-ceiling]]).

## The arc (3 stages, each verified with an executor gate)

### 1. v3 — door open ✅
- Base: `step_60000` (final pretrain) + **narrow-embedding-EOS fix** (rows 4–17).
- Data: `code_curated_v1` (1189, executor-verified) + `code_verified_v1` (173) + corrective + abstain.
- Gate (executor, 18 tasks): **syntax 1.0, eos 1.0, pass 11/18, unseen 2/9 (22%)**.
- **Root of the old "0/5":** the earlier `code_lora_v1/v2` trained on the wrong base
  (`sft_smoke_step_2000`) **without** the embedding fix → couldn't emit `<|end|>` → gibberish.
  With the correct base + fix: coherent, runnable code.

### 2. v4 — more pattern data ✗ (negative finding)
- +53 deterministic pattern-class functions (map/filter/reduce/digits/sort/dedup/string),
  ×3 weighted, gate functions deliberately excluded (transfer test).
- Gate (24 tasks): **unseen 3/14 (21%) — flat**; on the *same* 18 tasks **9/18 vs v3 11/18 ↓**.
- **Interference** instead of generalization: `nur_gerade → filters odd`, `dritte_potenz → ×3`,
  `doppelt → +2`; it even broke `remove_duplicates` + `zaehle_vokale`, which v3 could do.
- **Lesson:** dense, similar code SFT data produces interference at 0.9B, not generalization.

### 3. Repair loop — use feedback ✗ (negative finding)
- Inference loop on v3: write → hidden tests → **short standardized** error feedback
  (`TEST FAILED / function / input / expected / got / Fix the function only.`) → 1 repair.
- **pass@1 12/24, repair@1 12/24, repair_gain +0** (unseen likewise +0).
- **Lesson:** Given a shown error, the model produces the same wrong logic again —
  0.9B cannot use test feedback for self-correction.

## Conclusion
| Ability | 0.9B |
|---|---|
| Syntax / compiles | ✅ 100% |
| Clean stop (`<\|end\|>`) | ✅ 100% |
| Recall known patterns | ✅ |
| **Generalize new logic** | ❌ (~22%, more data = worse) |
| **Repair from feedback** | ❌ (gain +0) |

Code is **proven as a limit** for 0.9B, not as "almost there". The next sensible lever
is **not** more SFT/repair, but a **larger base model (3B/7B)** or significantly more
code *pretraining*. Until then: **keep v3 as the code adapter** (coherent, safe, stops cleanly).

## Method / reproduction
- Deterministically generated, **guaranteed-correct** verified tasks (reference code self-checked).
- **Executor gate** (`scripts/sft/code/executor_gate.py`): generates code → `compile` → runs the asserts;
  metrics syntax_rate / pass_rate / eos_rate / **unseen_pass** (main metric = transfer, not memorization).
- **Repair gate** (`scripts/sft/code/repair_gate.py`): pass@1 / repair@1 / repair_gain, short feedback.
- Trainer = the same narrow-embedding-fix trainer as Grounded ([[sft-eos-embedding-fix]]).

## Artifacts
- Adapter (NOT in git): `checkpoints/sft_code_v3/adapter_best.pt` (server). v4 discarded.
- Data: `code_curated_v1` (1189 verified), `code_verified_v1` (173), `code_v4` (patterns, negative finding).
- Scripts: `scripts/sft/code/`.
