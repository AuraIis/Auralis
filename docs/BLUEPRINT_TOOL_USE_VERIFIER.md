# Blueprint — tool use & self-verification (Helix v2)

> **Status:** design / decided (Michael + GPT + Claude, June 2026). Not yet implemented.
> **Phase:** comes AFTER the current reasoning-SFT (sft_v2). See `ZUKUNFT_BACKLOG.md` → Phase 3–4.
> **Guiding principle:** *A small model should not know or be able to do everything. It should learn when it has to verify.*
> Don't guess — verify. Consistent continuation of the anti-hallucination USP.

---

## 1) Why (evidence, not hope)
Helix is ~0.9B and undertrained. Benchmarks (June 2026) show: the model does **not**
reliably compute multi-step math in its head — at this size that is physically normal,
not a bug. "Beating" more parameters into the brain scales poorly.

**The lever is not the ability to compute, but learning to verify.** A small model can
learn very well to recognize a pattern ("this must be checked") and call a tool.
The correctness then comes from **outside** (calculator / code runner), not from the model.

```
347 × 892 in the head  → 0.9B guesses, mostly wrong
print(347*892) call    → model only has to KNOW that it should call → learnable
```

## 2) Honest ceiling (what we do NOT promise)
- You reliably get the **behavior**: test-before-accept, repair on error,
  call a tool on uncertainty.
- You do **not get guaranteed** good code quality or autonomous debugging. The
  self-repair loop presupposes that the model reads a traceback and produces a
  *correct* fix — exactly the ability that wobbles at 0.9B.
- Repair traces teach the **pattern** ("error = feedback, try again"), not guaranteed
  the solution. Still a win and fully on the USP line.

## 3) The 80/20 truth
> **The SFT traces are the easy 20 %. The inference harness is the real 80 %.**

Producing training data with a *precomputed* `<result>` is trivial. The hard
part is the runtime loop that **actually executes** the tool and feeds the result in live.
Without this loop the model just *hallucinates* the number in disguise = exactly the problem
we want to solve.

---

## 4) Rock-hard order (do not skip)
| Stage | Content | Prerequisite |
|---|---|---|
| 0 | Reasoning-SFT done (sft_v2) — check whether answers become more structured | running |
| **1** | **Math tool harness ALONE** (simplest case — proves the harness) | stage 0 |
| 2 | Tool-use SFT: calculations / units / small numeric logic | stage 1 green |
| 3 | Code annealing (Python-Edu is in `anneal_candidates/`) → code latent in the base | stage 2 |
| 4 | Code + own tests + self-repair + hidden-test data gate | stage 3 |
| 5 | *possibly* code-DoRA on an annealed base | stage 4 |

**Rationale stage 1 first:** the math tool only checks 5 things, all independently testable:
1. Does Helix recognize "I need a tool"? 2. Does it write the call correctly? 3. Does
generation stop at the call? 4. Does the harness execute Python? 5. Does Helix incorporate the result correctly?
Code brings 7+ problems at once (quality, tests, tracebacks, repair, spec,
hidden tests, sandbox) — too much at once.

**Rationale code-DoRA last:** an adapter **amplifies latent ability — it does not install
a new one.** Helix has seen 0 % real code in pretraining (only code *as prose*). Code-DoRA
on this base = building on sand. First code annealing (latent ability), then adapter.

---

## 5) Harness architecture (the actual build)

### 5.1 Inference loop (state machine)
```
1. Model generates text
2. Stop sequence </tool> reached?  → halt generation, hand over control
3. Parse tool call (language + body between <tool:python> … </tool>)
4. Sandbox executes → stdout/stderr/exit
5. Inject <result> … </result> into the context
6. Model continues generating (back to 1) until <|end|>
7. Guard: max. N tool calls per answer (infinite-loop protection)
```

### 5.2 Format (one convention, byte-exact, train == inference)
```
<|user|>
Was ist 47 mal 83?
<|end|>
<|assistant|>
<tool:python>
print(47 * 83)
</tool>
<result>
3901
</result>
47 mal 83 ergibt 3901.
<|end|>
```
- `</tool>` is the **stop sequence** of generation (central design decision).
- `<result>…</result>` is written by **the harness**, never the model (precomputed in training,
  live at runtime).
- Learn tags as plain text (no mandatory special tokens), but **consistently** — otherwise
  the byte-exact train/inference match breaks (cf. L-001: prompt mismatch).

### 5.3 Sandbox (non-negotiable)
**Never** run model-generated code unprotected.
```
Docker container · NO network · timeout (e.g. 5 s) · RAM limit · CPU limit
only tmpdir · no system paths · read-only root · non-root user
```
We sit in Docker anyway → substrate present. Minimal: an ephemeral sub-container/`nsjail`
per call.

## 6) Training data

### 6.1 Math-tool MVP (stage 2)
Three task types, small and measurable:
- **Calculation** (`print(347*892)`)
- **Units** (`print(3*60+25)` → hours→minutes)
- **Small numeric logic** (average, percentage, ratio)

Generation: teacher produces question + correct tool call; the **harness** computes `<result>`
deterministically (not the teacher → no teacher math errors). Format byte-exact as in 5.2.

### 6.2 Code phase (stage 4) — data-type mix
```
40 %  simple Python tasks with tests
20 %  error → traceback → repair → test again
15 %  edge cases
15 %  explain code
10 %  "spec unclear, I need details"
```
Keep tasks small (sort a list, duplicates, prime, palindrome, CSV/JSON, small
classes) — **no large projects**.

### 6.3 Hidden tests = data/eval tool, NOT inference
> This is exactly our gpt-4o-verify pattern, just for code (= external, incorruptible check).
- **Building data:** model writes a solution **+ own tests** → sandbox executes → *independent
  hidden tests* check against it → only traces that pass **both** go into the SFT set.
  (Prevents "passes own tests but is junk" like `def addiere(a,b): return 5`.)
- **Real user:** there are **no** hidden tests (the user *is* the spec). All that remains is
  "write own tests + run them". Do not confuse the two.

## 6b) STATE: math harness MVP built (June 2026)
`scripts/sft/tool_harness.py` — built + tested:
- **Safe calculator** (AST whitelist, no RCE): selftest 14/14, rejects `import`,
  `__import__/system`, `open`, compute bomb `9**9**9`, assignment, /0.
- **Loop proven** (`--selftest-only`, scripted fake model): Stop@`</tool>`
  → executor → `<result>` injection → resume. Transcript correct, result from
  executor (not model).
- **Open:** model does not yet *emit* tool calls → needs tool-SFT.
- **Self-generating data:** math-tool traces need NO teacher/key — the
  calculator IS the ground truth (problem → canonical call → executor result →
  answer template). This makes tool-SFT data buildable even without an OpenRouter key.

### Hard tool gate (before promoting a tool-SFT checkpoint)
```
✗ model token stream itself contains "<result>"  → stop sequence failed → FAIL
✗ no parseable <tool:python>…</tool>             → FAIL
✗ final answer number ≠ executor number          → result not taken over → FAIL
✓ stops @</tool> · executor computes · answer == executor number
```
(In the harness the model cannot write `<result>` at all — generation stops
at `</tool>`. "Fake-result" = the stop sequence failed. That is exactly what the gate checks.)

**State:** Phase-1 (call_only) passed — best step_400: tool_rate 100%, false_tool 0%,
fake_result 0%, parse 97%, correct **68%**. Phase 1.1 (enriched translation traces) aims
for correct ≥80% before Phase 2. Trainer `<result>` masking built + token-exact verified.

### Phase-2 end-to-end gate (additional — measures result USAGE)
Phase 2 can fail in 3 ways → 3 metrics:
```
result_usage_rate    : does the final answer use the executor number at all?       (Fail 1: ignores <result>)
answer_numeric_match : is the executor number EXACTLY in the answer?                (Fail 2: number copied wrong)
fake_result_rate     : did the model write <result> itself anyway?                 (Fail 3: hallucinates block) -> MUST be 0
```
Plus still: false_tool 0 · parse >95% · correct (capped by Phase 1.1). Promotion only if all green.

## 7) Success criteria (measurable, not "looks good")
- **Stage 1/2:** on a math probe set (n≥200): tool-call hit rate ↑, end-answer
  correctness clearly > base-without-tool (goal: correctness follows the calculator, not the guessing).
- Stop sequence catches in ~100 % (no "model writes `<result>` itself").
- **Stage 4:** share of solutions that pass independent hidden tests, ↑ vs. code-SFT-without-loop.
- Negative guard: tool-use must **not** worsen the general SFT quality (benchmarks de/en)
  (measured separately as a counter-check).

## 8) Building blocks already present
- **Verify pattern** (gpt-4o pass) = hidden-test mechanism, already built & validated.
- **Python-Edu** (`data/raw/anneal_candidates/`) = code-annealing data, already loaded.
- **Docker** = sandbox substrate, present.
- **Byte-exact prompt builder** (L-001 lesson) = basis for train==inference format.

## 9) Open implementation questions (clarify before stage 1)
- Wire the stop sequence cleanly into the inference path (sampler must halt at `</tool>`).
- Sandbox call latency per call (container spawn vs. persistent worker pool).
- Tokenizer: are tags encoded efficiently? (don't let `<tool:python>` etc. fall apart into 10 tokens.)
- DoRA targeting on the hybrid arch (Mamba `in_proj/out_proj`, GLA `q/k/v/g` = linear → adaptable,
  but must be wired in the trainer) — only relevant from stage 5.

---
*This document records a triple-triangulated decided direction. The order is
gated. No pulling stage 4/5 forward before stages 1–3 are green (reminder of the 500M dead end:
layer before foundation = garbage).*
