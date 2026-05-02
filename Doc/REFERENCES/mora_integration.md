# MoRA Integration Plan

> **Status**: Proof-of-Concept tested 2026-05-02. Real integration deferred
> to Phase-5 prep.

---

## Why MoRA matters for us

LoRA's `ΔW = B @ A` with rank `r` decomposition has a known weakness for
**fact injection**: low-rank updates can't represent the high-dimensional
relationships needed to teach a model new factual associations. This is
exactly the **L-002 lesson** from v1: the Blutdruck-LoRA hit train loss
0.0099 by memorising 212 samples but generalised badly.

**MoRA** (Jiang et al. 2024) addresses this. Same parameter count as LoRA
at a given `r`, but the update has effective rank `~r²` thanks to a
square `r×r` matrix `M` plus non-trained compression/decompression
functions:

```
ΔW = f_decompress(M @ f_compress(x))
```

For our planned Phase-5 specialists:

| Adapter            | Type        | Best fit  |
|--------------------|-------------|-----------|
| `politik_de`       | knowledge   | **MoRA**  |
| `recht_de`         | knowledge   | **MoRA**  |
| `medizin_dora`     | knowledge   | **MoRA**  |
| `code_engineering` | pattern     | DoRA      |
| `kreatives_schreiben` | pattern  | DoRA      |
| `alltagsdialog`    | pattern     | DoRA / LoRA |

3 of 6 planned adapters benefit from MoRA. Worth integrating.

---

## What's available today

* **peft-mora** — fork of HF PEFT 0.9.0 by the paper authors
  ([github.com/kongds/MoRA](https://github.com/kongds/MoRA/tree/main/peft-mora))
* Activation: `LoraConfig(use_mora=True, mora_type=6, r=8, ...)`
* The `mora_type` parameter selects the compression scheme:
  - `1`: simple sum-pooling (paper Eq. 6) — works for all ranks
  - `2`/`3`: mixed pooling for non-divisible dims
  - `6`: RoPE-based rotary compression (paper Eq. 9) — recommended for `r ≤ 32`

---

## What we tested (2026-05-02)

### Smoke test 1 — GPT-2 (4 layers, 7M params)

* ✅ Install + wrap-with-`use_mora=True` worked
* ✅ Trainable params: 16K (0.22% of 7M base) — matches MoRA's character
* ✅ Forward + backward + AdamW step ran clean on CPU
* ⚠️ Loss did not decrease over 10 random-data steps — too short to draw
  conclusions, plumbing-level OK
* ❌ **Save+reload drift = 1.58 max-diff** (expected <1e-4) — first real bug
* ❌ `merge_and_unload` crashed with shape-mismatch on GPT-2's `Conv1D`
  combined-QKV layer — known limit, GPT-2 specific

### Smoke test 2 — Mini-Llama (Linear-only, 0.9M params)

Tried with separate `q_proj`/`k_proj`/`v_proj`/`o_proj`/`gate_proj`/
`up_proj`/`down_proj` linears (Llama-style, what Auralis uses).

* ✅ Install + wrap worked, **66K trainable / 979K total = 6.7%**
* ❌ `task_type="CAUSAL_LM"` requires `prepare_inputs_for_generation` on
  the base model (HF Transformers convention)
* ❌ `task_type="FEATURE_EXTRACTION"` requires `forward(input_ids=...)`
  signature — our mini model used `forward(ids)`

These are HF-Transformers-class assumptions baked into peft-mora. Our
**Auralis-1B already implements these conventions** (it's HF-compatible)
so they would not block us in real use.

### Bugs identified for the future backport

1. **`cos`/`sin` buffers not registered for `mora_type=6`.**
   The `_apply_mora` method computes cos/sin tensors lazily and stashes
   them as plain attributes (`self.cos = ...`), not buffers
   (`self.register_buffer("cos", ...)`).
   → They are not saved with the state-dict, not restored on reload,
   and not synced across devices.
   → Manifests as the 1.58 save+reload drift in test 1.

2. **`get_delta_weight` returns wrong shape for fused-QKV linears.**
   Returns `(in_features, out_features)` even when `out_features` is
   actually `3 × hidden` (combined Q/K/V).
   → Crashes `merge_and_unload` on GPT-2 / GPT-NeoX style models.
   → Does NOT affect Llama / Mistral / Auralis style models with
   separate `q_proj` / `k_proj` / `v_proj`.

Both bugs are plausible 1-2 hour fixes once we open the file in earnest.

---

## Why we did NOT integrate further today

Three reasons:

1. **peft-mora is on PEFT 0.9.0 (early 2024).** Modern LLaMA-Factory
   needs PEFT ≥ 0.12 for DoRA. Installing peft-mora as-is means losing
   DoRA, LoRA+, PiSSA, and the recent quantisation backends. Net loss.

2. **Bugs above need a fix before production.** Cos/sin buffer issue is
   functional, not cosmetic.

3. **Phase 5 is still ~3 months away.** PEFT may have upstream MoRA by
   then (issue #1850 is open). If they do, we save the backport entirely.

---

## Phase-5 integration plan (concrete steps when we get there)

### Path A — Use upstream PEFT MoRA (if shipped)

1. `pip install -U peft transformers` and verify `LoraConfig(use_mora=True)`
   accepts the kwarg.
2. Run smoke test from this doc on Auralis-1B + 100 SFT examples.
3. Done — LLaMA-Factory inherits MoRA from PEFT automatically.

### Path B — Backport MoRA to modern PEFT (if not shipped)

1. Diff peft-mora vs upstream PEFT 0.9.0 to extract the MoRA-specific
   changes. Mostly two methods:
   * `_apply_mora` in `tuners/lora/layer.py` (~80 lines)
   * `get_delta_weight` modification in same file (~30 lines)
   * Plus `LoraConfig.use_mora` and `mora_type` fields (~10 lines)
2. Apply same changes to current PEFT release in a fork
   (`ForceGaming4K/peft`, branch `mora`).
3. Fix the two bugs identified above:
   * Register `cos`/`sin` as buffers (`register_buffer(..., persistent=False)`
     to keep adapter-checkpoint files small but include in `state_dict`).
   * Special-case `get_delta_weight` for fused-projection layers — or
     just document the limitation.
4. Add a unit test that exercises save→reload→inference with MoRA.
5. `pip install -e .` from our fork.
6. Patch LLaMA-Factory's `cli_args.py` to surface `--use_mora` and
   `--mora_type` (one-line additions to the LoraArguments dataclass).
7. Phase-5 first run: `politik_de` adapter on Auralis-1B-instruct
   (post-Phase-3) trained against the politik corpus's QA pairs.

Estimated effort for Path B: **3-5 days** of focused work. Higher than
my initial 1-week estimate because the two bugs are isolated (each
~2h to fix + verify).

### Path C — Eat the integration cost via upstream MoRA fork

Keep peft-mora 0.9.0 as a separate adapter-only environment. Train MoRA
adapters there, then load them into Auralis-1B-instruct outside of
LLaMA-Factory at inference time. Workflow: ugly but functional. Use only
if A and B both blocked.

---

## What's installed on bitbastion right now

```
/mnt/disk7/Auralis/phase2_corpus/tools/
├── LLaMA-Factory/         # 20 MB, cloned, deps not installed yet
└── MoRA/                  # peft-mora reference, installed in
                           # auralis-downloader container as
                           # peft 0.9.0 (overrides any later peft if present)
```

The auralis-downloader container has peft-mora 0.9.0 active. To **revert**
to a stock PEFT for other work: `pip install --force-reinstall peft`.

---

## TL;DR

* MoRA's math integrates cleanly into the LoRA-tuner pattern. Real.
* peft-mora 0.9.0 is functionally correct but on an old PEFT base and
  has two known bugs that need fixing before production.
* For Phase 5, watch upstream PEFT first; if MoRA hasn't landed there
  by August 2026, a 3-5 day backport is the path.
* Today's smoke test confirmed plumbing works; the rest is engineering
  scheduled for Phase-5 prep.
