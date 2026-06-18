# AURALIS — project status & overall history (v1 → Helix v2 → now)

> **Purpose:** single-source-of-truth overview for Michael and new reviewers/AIs.
> Synthesis — the *detailed* append-only logs are `HISTORY.md` (chronicle) and
> `LESSONS.md` (numbered lessons L-001…L-022). State: v3 foundation ~step 35k/50k.

---

## 0) Prehistory: Auralis **v1** (the predecessor)
An earlier project that made v2 possible in the first place — above all through its **mistakes**:
- **L-001:** training- vs. inference-prompt mismatch (`<|user|>` vs `User:\n`) obscured the quality for *weeks* → v2: ONE prompt builder, byte-exact test.
- **L-002:** LoRA learned *patterns, not facts* (loss 0.0099 = memorization, new questions failed) → v2: MoRA for facts, DoRA for patterns, disjoint val splits.
- **L-003:** GPT-2 tokenizer ~50% too inefficient for DE, but baked in hard → v2: **own 200k SentencePiece, done right once**.
- **L-004:** german-commons skewed the direction toward historical German → v2: deliberate mix ratios + sample reviews.
- **L-005/006:** no baseline tests; 3 model versions lost to a forgotten `--reset-optimizer`.

v1 delivered ~23.7 GB of deduplicated German pretraining material (~4.7B tokens), reused in v2.

## 1) v2 Phase 0–1 (Apr 2026): foundation
- **Phase 0 (tokenizer):** own **200k SentencePiece** (50 EN/40 DE/10 code), quality PASS (DE 134 tok/100w vs v1 ~220), byte-exact roundtrip. Lessons L-007..L-012 (SP normalization `identity`, NUL strip, threads, RAM, HF-`datasets`-v4 ban, code metric `tokens/KB`).
- **Phase 0.5 (architecture):** **Helix = 28L hybrid (6× Mamba-2 + 16× GLA + 6× sparse-attn)**, RMSNorm/SwiGLU/RoPE, tied 200k, d_model 1280, ~954M. Pure-torch reference + GPU kernels. Init loss 12.37 ≈ ln(200k) ✓, 50/50 tests green.
- **Blackwell bring-up:** mamba_ssm/fla/flash-attn on RTX PRO 5000 — the **`TRITON_OVERRIDE_ARCH=sm89` trick** (sm_120 otherwise rejected). Kernels save mainly **VRAM**.
- **Phase 1:** trainer/dataloader/tokenization/smoke. Bugs L-013 (`shuf` kills trainer disk IO), L-014 (checkpoint rotation crashes on `_emergency`), L-015 (RoPE only built for sparse layer), L-016 (`pgrep` wrapper matches itself). NVMe staging **42× faster**. **1B main run** (80k steps, mix 70/25/5).

## 2) Phase 2/3 + the 500M dead end (Apr–May 2026)
- **SFT data pipeline** (DeepSeek-V4 via OpenRouter), **anti-hallucination prompt** (L-017: a generic "don't lie" is not enough → forbidden speculation markers + few-shot → 0% instead of 3% hallucination). WSL2/3090 inference.
- **500M experiments (v5/v6):** forensics, v6 data plans (Gutenberg-Books, contamination checks), **SFT repair sweep v3→v9** against *interference* (Bonn/Berlin, photosynthesis, Faust/Goethe). **Frozen/live-response gates → no 500M checkpoint promotable**.
- **Diagnosis:** 500M not production-ready; cause = **interference** → way out = *cleanly weighted 1B mix* instead of mini-patches. → Pivot to 1B.

## 3) Bilingual 1B ramp + German edu filter (May 30/31)
- 1B de55/en45 up to step ~3400 disappointing. **Diagnosis:** not eval, not architecture, but **under-training (~16% Chinchilla) + quality-inverted DE mix**.
- **Edu filter (FineWeb-Edu methodology):** judge **qwen3-235b** (non-thinking) instead of gemini (L-018 thinking cost trap €24; L-019 the cheaper judge is *better*); cheaper **e5+Ridge classifier** (Pearson 0.87), threshold calibrated (L-021). **german_commons dropped** (L-020: OCR-historical). **DDP/multi-GPU** additive+gated (L-022).

## 4) Current session: warm-start v2/v3 + measurement post-mortem + knowledge profile
- Warm-start continued-pretraining; **5 misdiagnoses resolved** (all "looked like data, was measurement/LR/decoding"): warm-start LR too high · invalid baseline comparison · broken eval (wrong tokens/byte, stochastic, wiki-only tail → gap mirage 3.2 vs real 1.04) · guard false-stop (step 4250) · "no facts" = greedy-decoding artifact. (Details: `POSTMORTEM_messung_vs_daten.md`.)
- **Built:** step-0 diagnosis, deterministic eval, fixed guard, **rigorous fact-margin battery** (difficulty levels, 5 categories), honest dashboard, parallel CPU data pipeline (RedPajama+HPLT → 74 GB clean), 3 blueprints + post-mortem + data-strategy docs.
- **Knowledge profile (n=57, step 35k):** 95%-easy floor, gradient 95→89→77; **history 92 / geography 83 / tech concepts 80 / science 67 / language 64** (strict). Code = concepts only (**0% code trained**).
- **State:** v3 ~35k/50k healthy (val 2.34, 0 alarms); 50k check armed (gen + profile); SFT after (format ≠ knowledge); scaling 1B→3B→7B+ + targeted data (science/cross-lingual) as the plan.

## Recurring failure patterns (the real maturity)
1. **Suspect the measurement first, then the data** (eval bias, tokens/byte, decoding ≠ knowledge).
2. **Infra/disk coordination** (L-013 `shuf`, NVMe staging, cross-chat collisions).
3. **Judge/tooling choice** (L-018/019: cheap non-thinking judge > expensive thinking).
4. **Interference, not "dumb"** (500M dead end → larger, cleaner mix).

## Terms (do not conflate)
**Margin = knowledge · Top-k = recall proximity · Greedy = answer behavior · SFT = format/steerability.**

## Maturity level (as of now)
- ✅ Stable training · ✅ language learning (DE/EN fluent, separated) · ✅ factual grounding (surprisingly strong, history/geography)
- ⚠️ science + translation weaker · ⚠️ free decoding still raw
- ▷ open: instruction-following (SFT) · scaling 3B+ · knowledge_dna/kernel (unproven, optional boost)

## SFT milestone + benchmarks (June 2026)
**SFT successful:** first real SFT run (~32k diverse DE+EN, fact-checked via gpt-4o-verify
[269 hallucinations caught], decontaminated; ~1 epoch optimal, early-stopped at val plateau).
Helix went from the base, which could not even say "Berlin", to an **answering assistant**:
Vienna ✅, Madrid ✅ (EN!), clean stopping (looping gone, `eos-loss-weight 2.0`). But: on
specific facts **confident hallucination** (light bulb→Goethe), math unreliable.
> **Log sentence:** SFT successfully transformed the *behavior*. The largest remaining weakness
> no longer lies in the answer format, but in the **knowledge quality of the base model**
> (spotty, ~5–6/10; confident hallucination → calibration + better base, order annealing→3B).

**Benchmarks (own MC log-likelihood runner, n=300, acc/acc_norm):**
```
ENGLISH          MMLU  ARC-C HellaSw     GERMAN(*translated)  mmlu_de arc_de hellaswag_de
Helix-SFT        26,3  21,3  29,3        Helix-SFT            27,7    22,7   29,0
Qwen2.5-0.5B     48,3  32,3  50,0        Qwen2.5-0.5B         34,3    26,0   38,0
SmolLM2-360M     26,3  39,3  52,0        SmolLM2-360M         24,3    25,3   27,7
TinyLlama-1.1B   28,3  34,3  61,7        TinyLlama-1.1B       25,3    27,0   38,0
```
**The real takeaway (not "27.7 %"):** on **German** benchmarks Helix shows a clearly
smaller gap than on English ones and **beats several English-centric small models** —
specifically SmolLM2-360M + TinyLlama on **mmlu_de**, SmolLM2 on **hellaswag_de** (on **arc_de**
Helix is last). Qwen's MMLU lead shrinks from ~22 (EN) to ~7 (DE). → **The language strategy
(200k vocab, de55/en45, own tokenizer) pays off measurably.** Absolute values all low
(~chance–38 %) = under-training / size signal. EN 26–29 % refutes "can only do German".

**State in brief:** language model ✅ · German competence ✅ · assistant format ✅ ·
knowledge ⚠️ · commonsense ⚠️ · reasoning ⚠️ · scaling 🚧.
**Next levers (gated, measured):** annealing (FineWeb-2-DE/Cosmopedia/Code — already loaded) →
calibration SFT (honesty) → 3B. Measure first, then decide.

## Tool-use milestone — verified computation (June 2026)
**The transition from "guess" to "verify" for math succeeded.** Helix recognizes a
computation task, calls a safe external calculator (AST whitelist, no RCE), takes over
the result and answers factual questions without a tool. System progress (model + harness +
calculator), not "one prompt worked".

**Before → after (same questions, one day apart):**
```
12 + 15      Quicktest: "12"          → step_600: print(12+15)→27 → "12 + 15 ergibt 27."
15% von 240  Quicktest: garbage       → step_600: print(240*15/100)→36 → "Das sind 36."
80€ −20%     —                        → step_600: print(80-80*20/100)→64 → "Er kostet dann 64 Euro."
Wien/Faust   confident hallucination  → step_600: directly correct, NO tool
```

**Methodology (3 gated phases, best-by-GATE — val_loss was demonstrably misleading):**
Phase 1 (call_only) → 1.1 (language→formula) → 2 (result injection, `<result>` loss-masked).
Promoted: `checkpoints/tool_sft_v12/sft_smoke_step_600.pt`.

**Hard numbers (own dual end-to-end gate, n=100 math + 51 facts):**
```
correct 94% · parse 100% · fake_result 0% · false_tool 0% · answer_match 85%
Buckets: percent 24/24 · word 21/21 · speed 10/10 · english 7/7 · time_unit 16/17 · simple 16/21 (76%)
```
Core proofs: `false_tool 0%` (no misuse) · `fake_result 0%` (no faked results,
thanks to `<result>` masking) · `parse 100%` (harness executes reliably). The field confirms it:
Toolformer (when to call a tool), Qwen2.5-Math TIR (1.5B+Python→MATH 80).

**Limits (honest):** in-distribution (trained types, new numbers); `simple` weak due to
sqrt/`squared`; answer_match conservative (comma/period). Tool-use adds **no knowledge**.

## Modular-adapter milestone — skills without collateral damage (June 2026)
**Helix can get modular behavior skills via LoRA adapter without damaging the base.**
That is the core of the modular vision — and the counter-proof to full finetuning.

**The problem (measured):** full-FT calibration forgets tool-use/facts after ~50 steps
(catastrophic forgetting). Two rounds delivered NO checkpoint with honesty AND retention.

**The solution:** LoRA adapter (1.2 % params) on a **frozen** base. `src/auralis/adapters/lora.py`
(injects 188 GLA/attn/FFN modules; Mamba kernel excluded, since `.weight`-direct). Plus α-control
(`set_adapter_scale`) — the skill strength is dosable at **inference**, without retraining.

**α-sweep (honesty adapter on step_600, held-out, n=60 invented):**
```
α=0.00  inv-abstain 3%   people 5/5  math 5/5   ← = exactly base (control passed)
α=0.50  inv-abstain 95%  people 5/5  math 5/5   ← SWEET SPOT
α=1.00  inv-abstain 100% people 5/5  math 5/5
```
> **Adapter off = exactly base. Adapter on = steerable extra behavior. α=0.5 delivers 95 %
> abstention WITHOUT loss of tool or facts.** What full-FT could not do twice (honesty OR
> retention), the dosable adapter does in one run, on a guaranteed-intact base.

**Roadmap validated:** base (language+tool-use) frozen · honesty-LoRA @ α=0.5 switchable ·
code-LoRA after annealing · knowledge-MoRA later. Two PEFT fixes in the code (DoRA memory; grad-ckpt
+ frozen base → `enable_input_require_grads`).

## Guiding principle
> Data collection happens on the basis of **knowledge profiles**, not the overall val loss.
> And: before a bad number "is the data" — check whether the number even measures what you think.
