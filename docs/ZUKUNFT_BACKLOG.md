# Future Backlog — parked ideas & resources (by phase)

> **Purpose:** Michael's idea machine produces many good but often *far-ahead*
> directions. They are parked here — with an honest phase tag, so that nothing gets
> lost and nothing gets pulled forward before the base/the size can carry it.
>
> **Ground rule:** Every stage is gated on the previous one + on model size.
> Do not skip the order. (Reminder of the 500M dead end: layer before foundation = garbage.)
>
> **Guiding principle (tool-use philosophy):** *A small model should not know or be able to do everything.
> It should learn when it has to verify.* — not guess, but verify.

---

## Where we stand
- ✅ **Pretraining** (1B, de55/en45) — completed, knowledge + language demonstrated.
- ⚗ **Phase 1 — base SFT** (currently active): turn the base into an *answering* assistant.
  Smoke passed ("Berlin"), pilot (282 examples, DeepSeek + premium verify) running.

## Phase 2 — calibration / honesty SFT  *(as a share of the SFT mix)*
- Behavior: answer first, stable facts directly; **markers only when there is risk** (🔧 current source,
  ⚠️ caution, ❗ definition-dependent); correct a false premise; no fake live-check.
- Status: **generator + verify pass built & validated.** Comes as a slice into the main SFT.

### Helix-R-Tuning v1 (self-labeling, KEY-FREE) — source: R-Tuning (arXiv 2311.09677)
> The paper's finding = exactly our problem: classic instruction tuning **forces** an
> answer → the model guesses on unknowns. Solution: split questions into "in model knowledge / not sure"
> and deliberately train uncertainty. Refusal is a **generalizing meta-skill**.
>
> **IMPORTANT — self-labeling does NOT mean "Helix judges itself".** It means: Helix answers
> questions with a KNOWN solution, a **script** compares against gold/MC/regex/executor:
> ```
> answered correctly   -> confident-answer (KEEP, retention anchor)
> answered wrong       -> train uncertainty/refusal ("I cannot answer this reliably")
> calculation          -> tool-needed (call tool, do not guess)
> ```
> **Key-free feasible** for verifiable answers: MC (MMLU-DE/ARC-DE), factual questions with
> an unambiguous answer, math (executor), translation with a clear target. **Not** key-free for
> open answers ("Explain a volcano") → later teacher/rule-checks/local judge.
>
> **Recipe Helix-R-Tuning v1:** (1) gold-label question bank (MMLU-DE/ARC-DE + own fact battery
> + contrastive cases Bonn/Berlin, Pluto). (2) let Helix answer. (3) auto-label (see above).
> (4) SFT: known→answer correctly, unknown→do not invent, calculation→tool.
>
> **R-Tuning-R (replacement)** = on uncertain items, do NOT show the correct answer, but train a genuine
> "I don't know" → highest refusal rate, but **over-refusal danger**.
> → **MANDATORY retention gate** (Vienna/Madrid/Berlin/Pluto must stay answered; invented
> entities must NOT be embellished; `12+15`→tool). One retention regression = not
> promotable. Dose abstention moderately (RLVR humility finding: too much = refusal bot).

### MEASURED — Calib v1 (June 2026): capability proven, recipe too coarse
- Probe (key-free): step_600 hallucinates **100 %** of the invented entities (60/60), knows
  capitals ~74 %, works only 6 %. → clear calibration need.
- Calib-SFT v1 (714 ex: 155 abstention / 38 confident / 600 anchors, ~20 % abstention).
- Dual gate (held-out): Honesty **0→93 % abstention** on NEW invented ones, capitals
  held (89 %). **BUT the demo revealed what the aggregate gate masked:** over-refusal
  leaks onto known facts (Einstein→"I don't know") AND breaks math dispatch (`12+15`→
  abstention instead of tool; `15 %` still worked). → **step_50 NOT promotable.**
- **Lesson:** the gate only measured capitals → over-refusal hid in the unmeasured space.
  Greedy demo caught it, sampled aggregate did not. ("Check whether the number measures what you think.")
- **Fix Calib v2:** (a) **mix in tool-SFT traces** (don't forget dispatch), (b) confident
  anchors BROADER (people/facts, not only capitals) + many short known-facts (against
  "short→abstain"), (c) abstention share < 20 %. **Gate v2:** measure retention more broadly
  (capitals + people + math dispatch), not only capitals.

## Guiding principle — better development loop, NOT recursive self-improvement
> Source: Anthropic, "Recursive Self-Improvement". Helix gets better **not through "more model"**,
> but through a **better loop around the model**: model + tools + tests + human.
- **Human = direction-setter** (what to test? what is success/fail? which direction?). Model/tools
  execute (generate data, run tests, find bugs, reports) — but **no auto-promote**.
- **New bottleneck = review/verification**, not generation. Hence hard gates everywhere:
  ```
  no gate       -> no promote
  no verify     -> no dataset
  no benchmark  -> no claim
  no sandbox    -> no tool-use
  ```
- **No** "Helix trains itself and we let it run" (at 0.9B technically nonsense + dangerous).
- Later optional: auto-experiment loop (small run → gate → report → promote/reject/retry),
  still with a human on the goals.

## Phase 3–4 — tool use / verifier (AFTER solid SFT)  ⭐ DECIDED (Michael + GPT + Claude, June 2026)
> **Triple-triangulated.** Tool-use is the next *big* step after the SFT — DoRA comes
> only later. Rationale: a 0.9B reliably computes `347×892` wrong, but can learn
> *when* it should call a calculator. Verification **outside** the model instead of guessing
> in the head. This is the consistent continuation of the anti-hallucination USP: **don't guess, verify.**

**Rock-hard order (do not skip):**
1. Finish reasoning-SFT + check whether answers become more structured.
2. **Math tool harness FIRST** (alone, simplest case) — proves the harness.
3. Tool-use SFT for simple calculations / units / small numeric logic.
4. **Only then** code annealing (Python-Edu is ready).
5. Then code + own tests + self-repair.
6. Then *possibly* code-DoRA.

**The harness is the actual work (80 %), the SFT traces are the easy 20 %:**
- **Stop sequence `</tool>`** → generation stops, model hands over control.
- **Sandbox executes** (Docker, **no network**, timeout, RAM/CPU limit, only tmpdir, no system paths).
- **Loop injects `<result>…</result>`** → model continues generating until `<|end|>`.
- Without this loop the model just *hallucinates* the number in disguise = exactly the problem to be solved.
- **Hidden tests = data/eval tool, NOT inference** (= our gpt-4o-verify pattern for code):
  while building data an *external* instance checks independently; with the real user there are no
  hidden tests (the user *is* the spec), so all that remains is "write own tests + run them".
- **Honest ceiling at 0.9B:** you reliably get the *behavior* (test-before-accept,
  repair on error), **not guaranteed** good code quality/autonomous debugging.
  Repair traces teach the *pattern*, not guaranteed the correct fix. Still a win.
- **Building blocks already present:** verify pattern (= hidden tests) · Python-Edu (= code annealing) · Docker (= sandbox).

## Phase 5+ — advanced (needs code ability + size 3B–7B+)
- **Model-*built* tools** (writes + tests tools in sandbox). Risky, only with code
  pretraining (`code.bin` ~677M tokens, not yet in the mix) + code-SFT.
- **Reasoning-RL / RLVR:**
  - 📌 **MiniMaxAI/SynLogic** — https://huggingface.co/datasets/MiniMaxAI/SynLogic
    - ~49k examples, **35 logic tasks** (Sudoku, Cipher, Game-of-24, Cryptarithm…).
    - Format: **RL-with-verifiable-rewards** (`<think>/<answer>` + verifier) — *not* SFT.
    - Language: **EN + ZH** (no German). License: **MIT** ✅.
    - Target size per the authors: **7B / 32B**. Unusable for 0.9B.
    - **Verdict:** very good, but wrong phase. Needs: larger model **+** a complete RL pipeline
      (GRPO/verifier). Only in the reasoning/scaling phase.

## Scaling context (why the order matters)
- Vocab stays fixed at 200k → share drops with size (27 %@1B → ~13 %@3B → ~10 %@7B+).
- Reasoning-RL, autonomous tools, hard logic: only worthwhile from 3B–7B+, on a solid SFT base.

---

## Learned from comparable projects (Zamba, SmolLM2, TinyLlama, Karpathy)
> Source: reports read (arXiv 2405.16712, 2502.02737, 2401.02385, llm.c). Finding:
> Helix independently hit the same problem classes → real model engineering.
> These points are **NEW** or **refinements**, to be applied AFTER the current SFT run.

### ⭐ v-next step 1 — annealing phase (Zamba's biggest lever: MMLU 50.8→57.7)
- Short continued-pretrain phase (~5% tokens) with **only top data** (cleanest DE +
  math + **code from `code.bin`**) + **LR almost to zero**. Polishes the base before SFT goes on top.
- Picture: the last week of study before the exam, only from the best textbook.
- Effort: medium (anneal mix + short run). At 0.9B possibly smaller than +7. Gated on SFT result.

### 💸 Cheap improvements (immediate/cheap, after the run)
- **Substring decontamination** (SmolLM2): don't only filter exact eval probes, also
  *reformulated* ones (substring/fuzzy). ~20 lines → watertight eval honesty.
- **EOS audit** (TinyLlama lost 2.3 trillion tokens to an EOS bug): verify once that every
  SFT example ends with exactly one `<|end|>` in the right place + the loss mask catches it.
- **Iterate the data mix by eval** (SmolLM2): per-category eval → adjust the mix in a targeted way
  instead of guessing. = our knowledge-profile strategy, just more systematic.

### 📊 Benchmarks — NEXT concrete step (right after the SFT run)
> Expert consensus: without standard numbers Helix is not *assessable*. Benchmarks are a
> measurement, not a maturity level — the base is technically already benchmarkable.
- **Blocker = `lm-eval-harness` wrapper:** Helix is its own architecture, needs a small
  wrapper (log-likelihood interface). That is the actual "readiness" work, nothing else.
- **Measure:** `Base (50k)` vs `Base+SFT` vs baselines (**Qwen2.5-0.5B, SmolLM2-360M/1.7B, TinyLlama**)
  on the same harness → fair comparison + shows the SFT lift.
- **Benchmarks:** MMLU/HellaSwag/ARC (multiple-choice, likelihood — no generation needed)
  + **German variants** (home-turf advantage, fairer) + IFEval/MT-Bench (instruction, needs SFT).
- **Honest expectation:** first numbers modest (0.9B, undertrained → MMLU possibly ~chance).
  Low ≠ failure — it is the first *honest* baseline. DE benchmarks first.
- *Publicity-effective* only after **annealing** (that's when Helix is strongest).

### 🏗️ Architecture note (v3 / scaling)
- Zamba: **ONE shared attention** instead of several separate ones → param-efficient. For Helix v-next
  check whether attention layers can be shared → more budget for the Mamba.

### ✅ Confirmed (we already do this right)
- Grad clipping 1.0 · deterministic eval · quality>quantity · code needs GitHub not web.

---

### Parked "great-but-far-ahead" ideas (chronicle)
1. Autonomous tool-building agent (decompose→build→test→run) → **Phase 5+**.
2. Verifier agent (claim decomposition, confidence, verification need) → **Phase 2 visible / 3–4 with tools**.
3. SynLogic / reasoning-RL for 7B–32B → **Phase 5+**.

*All three correct. All three need the finished small assistant + more size first.*
