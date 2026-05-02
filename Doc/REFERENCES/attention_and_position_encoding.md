# Attention Variants & Positional Encoding — Auralis Reference

> **Purpose**: Single source of truth for "which attention/PE choice when".
> Used by future architecture decisions (v3 scaling, MoE pivot, MoRA design).
> **NOT** a tutorial — assumes you know what attention is. Focuses on
> trade-offs and Auralis-specific implications.
>
> **Status**: Living document. Update on every published architecture
> read-through. Last sync: 2026-05-02.

---

## TL;DR — what Auralis uses today, what we'd use later

| Component       | v2 (today)                                | v3 Path A (3B dense hybrid)        | v3 Path B (7B/24B MoE pivot)              |
|-----------------|-------------------------------------------|------------------------------------|-------------------------------------------|
| Attention type  | MHA on the 6 sparse-attention layers      | GQA-4 on the attention layers      | MLA + Partial RoPE                        |
| Position encoding | RoPE on attention layers, NoPE on Mamba/GLA | same (effectively RNoPE)         | Partial RoPE inside MLA                   |
| KV-cache impact | small (only ~21% of layers carry one)     | small-moderate                     | very compressed via MLA latents           |
| Framework risk  | low (mamba_ssm + fla + native attention)  | low (mainstream patterns)          | high (MLA support patchy outside vLLM)    |
| When justified  | now                                       | continuity scaling, ~$5-10k        | only with funding + MoE-research expertise |

The rest of this document explains why those table cells contain what they contain.

---

## 1. Why attention variants exist at all

The original Multi-Head Attention (MHA, "Attention Is All You Need") has one
fatal scaling problem: the **KV-cache**. During autoregressive generation
each new token reads the keys and values of all previous tokens; those have
to be kept in GPU memory.

```
KV-cache size = 2 × n_layers × n_heads × head_dim × seq_len × dtype_bytes
```

For Llama-3-8B at 8k context in bf16: ~16 GB. For DeepSeek-V3 at 128k
context: would be ~100 GB without compression. KV-cache, not weights, is
the binding constraint at long context — that is the whole motivation for
GQA, MQA, MLA, and the hybrid SSM/linear-attention designs we use.

The Auralis v2 hybrid sidesteps most of this because Mamba and GLA layers
have constant-size internal state that does not grow with sequence length.
We have **only 6 of 28 layers** carrying a real KV-cache — about 21% of a
pure-Transformer's footprint. So the "GQA vs MLA" debate is much less
pressing for us than for Llama / DeepSeek.

---

## 2. Attention variants

### 2.1 MHA — Multi-Head Attention (baseline)

* `n_heads` independent (Q, K, V) projections per layer.
* Maximum expressivity, maximum KV-cache cost.
* What "Attention Is All You Need" specified. Used by GPT-2, Llama 1, BERT.
* **When to use today**: small models (≤1B), short context (≤4k), or
  research where expressivity matters more than memory.
* **Auralis v2**: this is what our 6 sparse-attention layers run.

### 2.2 MQA — Multi-Query Attention

* Single shared (K, V) across all heads, but `n_heads` queries.
* Cuts KV-cache by `n_heads`× → typically 8-32× smaller.
* Faster inference, but **noticeable quality regression** (~0.5-1.0 pp on
  most benchmarks) because all heads see the same key/value subspace.
* Used by: PaLM 2 (small), Falcon-7B/40B.
* **When to use**: aggressive inference cost reduction, willing to accept
  small quality hit.

### 2.3 GQA — Grouped Query Attention (current mainstream)

* Compromise: `n_kv_heads` < `n_heads`, queries are grouped to share
  K/V. E.g. Llama-3-8B has 32 query heads, 8 KV heads → ratio 4.
* Cuts KV-cache by `n_heads / n_kv_heads`× while preserving most of
  MHA's quality.
* **Quality vs MHA**: typically <0.2 pp on benchmarks at ratio ≤8.
* Mathematically simple — 50 lines of code change from MHA.
* Used by: Llama 2/3, Qwen 2.5, Mistral, almost everything modern dense.
* **When to use**: default choice for any new dense Transformer ≥3B.
* **Framework support**: excellent. Native in HF Transformers, vLLM,
  llama.cpp, Llama-Factory, nanotron.

### 2.4 MLA — Multi-Latent Attention

* The DeepSeek-V2/V3 contribution. Compresses K and V into a shared **latent
  vector** that gets projected back at compute time ("projection absorption").
* Cache size of one latent per token instead of `n_kv_heads × head_dim` of
  K and V → 4-8× smaller than even GQA-4. Effectively GQA-2.25 equivalent.
* Higher model capacity than GQA at the same cache size, because the latent
  carries more information per byte than `n_kv_heads`-dim key/value pairs.
* **Catch 1 — RoPE incompatibility**: rotational position encoding cannot be
  applied to the compressed latent directly. Needs **Partial RoPE** (see §3.5):
  rotate only a small slice of the dimensions, leave the rest unrotated.
* **Catch 2 — framework support**: vLLM (≥0.5.0) and llama.cpp implement
  inference; **most training frameworks** (HF Transformers ≤4.42, nanotron,
  Megatron-LM as of late 2025) **do not** ship a clean MLA training path.
  You typically need to write your own training loop or fork DeepSeek's code.
* **Catch 3 — implementation complexity**: 5× the line count of GQA, harder
  to debug, more places where numerical instability can hide.
* Used by: DeepSeek-V2, DeepSeek-V3, Kimi K2.
* **When to use**: very long context (≥32k), MoE-style architecture where the
  cache savings let you fit more experts in memory, and you have engineering
  capacity to wrangle a custom training stack.

### 2.5 Sliding-Window / Sparse Attention

* Each token only attends to a fixed window (e.g. 4096 prior tokens) instead
  of the full prefix. Used in Mistral, Phi-3, our `sparse_attention` layers.
* Cuts both compute (quadratic → linear in seq_len) and KV-cache (cap at
  window size).
* Quality cost: small if model has other long-range mechanisms (we do, via
  Mamba+GLA below).
* **When to use**: as a complement to other long-range mechanisms, not as
  the only attention.

### 2.6 Linear / Gated Linear Attention (e.g. GLA, RetNet, RWKV)

* Reformulates attention so it can be computed recurrently, with a constant-
  size hidden state per layer.
* No quadratic compute, no KV-cache that grows with context.
* Quality historically below MHA, but **recent gated variants (GLA, Mamba-2)
  are competitive** at the 1B-3B scale.
* Used by: RetNet, RWKV, Mamba-2, our 16 GLA layers.
* **When to use**: as one slice of a hybrid stack; rarely as the sole
  mechanism (pure linear attention struggles at certain reasoning tasks).

### 2.7 Mamba (state-space, not strictly "attention")

* Recurrent state-space model. No attention at all. Constant compute and
  memory per step regardless of sequence length.
* Strong at long-range information aggregation, weaker at precise recall
  of recent tokens.
* Used by: Mamba, Mamba-2, the 6 Mamba layers in our v2.
* **When to use**: in hybrid stacks for the deep, generalising layers.

---

## 3. Position encoding variants

Why does position encoding matter even more than attention variant?
Because it determines whether your model **extrapolates to context lengths
beyond what it saw in training**. Get this wrong and you can never extend
your context window without retraining.

### 3.1 APE / Sinusoidal (legacy)

* Original Transformer paper. Add a fixed (or learned) position vector
  to the input embedding.
* Cannot extrapolate — performance collapses past trained `seq_len`.
* Nobody ships this in 2026. Mentioned for historical context.

### 3.2 ALiBi (Attention with Linear Biases)

* No explicit position encoding. Adds a linear penalty to attention scores
  proportional to distance.
* Extrapolates to ~2× trained length without quality loss, more with some
  degradation.
* Used by: BLOOM, MPT.
* Largely superseded by RoPE for typical use.

### 3.3 RoPE — Rotary Position Embedding (current mainstream)

* Rotates each (Q, K) pair by an angle proportional to its position.
* Crucial property: **relative position is preserved in the dot product**
  → the model learns "X tokens apart" naturally.
* Extrapolates moderately (1.5-2× trained length) without modification;
  with **YaRN / NTK / Linear scaling** can extend to 4-8× before quality
  drops.
* Used by: Llama, Mistral, Qwen, GPT-NeoX, Auralis (in our 6 attention layers).
* **De-facto standard** for any new Transformer.

### 3.4 NoPE — No Positional Encoding

* Don't add any explicit position information. The causal attention mask
  alone gives the model enough order signal.
* Surprisingly **extrapolates better than RoPE** on long-context tasks
  (recent finding — Kazemnejad et al. 2024 "The Impact of Positional
  Encoding on Length Generalization in Transformers").
* Quality on short context: competitive with RoPE.
* Pure NoPE is rarely shipped; usually combined with RoPE in alternating
  layers (RNoPE) for best of both.

### 3.5 Partial RoPE (a.k.a. Fractional RoPE)

* Apply RoPE to **only a fraction of the head dimensions**, leave the
  rest unrotated. Within a single layer.
* Required by MLA: the latent compression doesn't survive full rotation,
  so DeepSeek-V2/V3 partition `head_dim = d_rope + d_nope`, rotate only
  the `d_rope` slice.
* Used by: DeepSeek-V2/V3, GLM-4.5, MiniMax-01.
* **When to use**: paired with MLA (almost always co-occurs).

### 3.6 RNoPE — RoPE-then-NoPE (alternating layers)

* Mix at the **layer level**: some layers use RoPE, others use no PE at
  all. SmolLM3 uses ratio 3:1 (RoPE in three of every four layers, NoPE
  in one).
* The RoPE layers handle "recency bias" / local context; the NoPE layers
  give global access.
* Cheap to implement (no math change, just skip the rotation).
* Used by: SmolLM3, Llama 4 (with chunked-attention RoPE layers), early
  GPT-NeoX experiments.
* **When to use**: dense Transformers above ~3B where you want long-context
  extrapolation without paying for it at training time.

### 3.7 Auralis "implicit RNoPE"

We don't ship a labelled RNoPE, but our hybrid layer mix achieves the same
effect by construction:

| Layer kind         | Position encoding         | What it contributes               |
|--------------------|---------------------------|-----------------------------------|
| 6× Mamba           | None (state-space order)  | Long-range, smooth aggregation    |
| 16× GLA            | None (gated decay order)  | Mid-range, gated recall           |
| 6× Sparse Attention| **RoPE**                  | Short-range, precise recall       |

22 of 28 layers (78%) carry no explicit positional encoding. This is
*structurally* RNoPE — and was chosen long before RNoPE got its name.

---

## 4. Decision matrix — what to pick at what scale

| Constraint                          | Recommended attention | Recommended PE        | Notes                                           |
|-------------------------------------|-----------------------|-----------------------|-------------------------------------------------|
| Dense ≤3B, ≤4k context              | MHA or GQA-4          | RoPE                  | Stay simple. Frameworks all support this.       |
| Dense 3B-7B, 4-32k context          | GQA-4 or GQA-8        | RoPE + YaRN at infer  | Llama-3-Pattern. Cheapest "good".               |
| Dense 7B+, ≥32k context             | GQA-8 or MLA          | RNoPE or full RoPE    | MLA only if you have engineering capacity.      |
| MoE 8×7B-style                      | MLA                   | Partial RoPE          | DeepSeek-V3-Pattern. The KV-cache savings unlock the active params. |
| Hybrid (Mamba/GLA + Attention)      | MHA or GQA on attn    | RoPE on attn, NoPE on the rest | Auralis v2 pattern. KV-cache problem half-solved by construction. |
| Edge / on-device                    | GQA-4                 | RoPE                  | MLA framework support outside vLLM is patchy.   |
| Research / experiments              | Whatever you want     | NoPE worth trying     | NoPE often beats RoPE on long-context probes.   |

---

## 5. Auralis-specific implications

### 5.1 v2 (today, 1B)

We ship **MHA + RoPE in the 6 attention layers**, no PE in the 22 non-attention
layers. Total KV-cache at 2k context: ~80 MB. We will not ever change this
mid-flight; the architecture is committed to.

### 5.2 v3 Path A — Hybrid Continuity Scaling (3B dense)

Width-up-scale to `d_model=1920`, keep the 6+16+6 layer mix. Only attention
layers are touched.

* **Switch MHA → GQA-4**: trivial change in `helix_model.py` (replace
  `nn.MultiheadAttention` instantiation with a GQA module — fla has one).
* **Keep RoPE** on the attention layers; nothing else changes.
* **Why GQA-4 not MLA**: only 6 attention layers carry KV-cache; the saving
  from MLA over GQA-4 would be tiny in absolute MB, but training-stack
  complexity would multiply.

### 5.3 v3 Path B — MoE Pivot (≥7B active, 24B+ total)

A different beast. Probably we would not call it "Auralis" any more — the
foundational decisions change too much.

* **MLA + Partial RoPE** on a denser attention stack (no Mamba/GLA mix).
* **Active-experts MoE** routing on the FFN.
* **Custom training stack** — DeepSeek's open-source code as a base, not
  HF Transformers.
* Realistic only with a research team and ≥€100k compute budget.

### 5.4 What it does NOT make sense for us to do

* **Convert v2 to MLA mid-life**: pointless. KV-cache isn't our bottleneck.
* **Switch to NoPE on the attention layers**: the recency bias from RoPE is
  valuable on the top layers; don't break that.
* **Adopt YaRN extension before we have ≥4k-context users**: optimisation
  for a problem we don't have yet.

---

## 6. Framework support reality check

| Stack            | MHA | GQA | MQA | MLA | RoPE | NoPE | Partial RoPE | RNoPE |
|------------------|:---:|:---:|:---:|:---:|:---:|:---:|:------------:|:-----:|
| HF Transformers  | ✅  | ✅  | ✅  | ⚠️  | ✅  | ✅  | ⚠️           | ⚠️    |
| vLLM             | ✅  | ✅  | ✅  | ✅  | ✅  | ✅  | ✅           | ✅    |
| llama.cpp        | ✅  | ✅  | ✅  | ✅  | ✅  | ✅  | ✅           | ✅    |
| Llama-Factory    | ✅  | ✅  | ✅  | ❌  | ✅  | ✅  | ❌           | ⚠️    |
| nanotron         | ✅  | ✅  | ✅  | ❌  | ✅  | ⚠️  | ❌           | ❌    |
| Auralis (own)    | ✅  | ⚠️  | ❌  | ❌  | ✅  | ✅* | ❌           | ✅*   |
| flash-linear-attention | n/a | n/a | n/a | n/a | n/a | ✅ | n/a | n/a |
| mamba_ssm        | n/a | n/a | n/a | n/a | n/a | ✅ | n/a | n/a |

`✅*` = via the hybrid construction (no explicit code path needed)
`⚠️`  = supported but with caveats (often depends on model class)

**Implication**: if we ever go MoE+MLA, we cannot stay in HF Transformers'
training pipeline. That would be a several-week migration to a vLLM- or
DeepSeek-fork-based training stack. Worth knowing before committing.

---

## 7. References

* Vaswani et al. 2017 — Attention Is All You Need (MHA baseline)
* Shazeer 2019 — Multi-Query Attention
* Ainslie et al. 2023 — GQA: Training Generalized Multi-Query Transformer
* DeepSeek-AI 2024 — DeepSeek-V2 / DeepSeek-V3 technical reports (MLA + Partial RoPE)
* Su et al. 2021 — RoFormer (RoPE)
* Press et al. 2022 — Train Short, Test Long (ALiBi)
* Kazemnejad et al. 2024 — The Impact of Positional Encoding on Length Generalization (NoPE empirics)
* SmolLM3 technical report — RNoPE-style alternation in dense 3B
* Llama-4 technical disclosures — chunked RoPE + NoPE alternation
* Gu & Dao 2024 — Mamba and Mamba-2 (state-space alternative)
* Yang et al. 2024 — Gated Linear Attention (GLA)

---

## 8. Open questions for v3 decision

1. Do we want context length > 8k for v3? If no → GQA + RoPE end of story.
2. Is there a Forschungs-Co-Investor who can help with MoE+MLA training
   stack? If no → Path A only realistic.
3. Will MoRA adapter capacity matter more than base capacity? If yes →
   stay dense, more compute on Phase-2-Continued + Phase-5 specialists.
4. By the time we plan v3, has HF Transformers shipped a clean MLA training
   path? If yes → Path B becomes much cheaper to try.

These get re-evaluated when Phase-1 finishes (~mid-May 2026) with real
benchmark numbers from `eval/benchmarks_v1.yaml`.
