# Helix v3 — byte-level / small-vocab sketch (research note, NOT the current run)

Status: **idea sketch for a future experiment.** The live foundation run and the
German-data pipeline (fineweb2-v2, german_commons, RedPajama) are the priority and
are unaffected by this. This note exists so the idea is captured concretely and can
be judged on cost/benefit later, not built on impulse.

The seed: "build an AI that understands text closer to the machine level, or finds
a denser pattern/language (DNA analogy)." Translated into a real, testable design:
**drop the 200k tokenizer and operate at (or near) the byte level.** Below is what
that buys, what it costs, and the honest verdict.

---

## 1. Why this is even worth considering — the 200k-vocab tax

Helix v2: `d_model=1280`, `n_layers=28`, `vocab=200000`, tied embeddings.

- Embedding/LM-head table = `200000 × 1280 = 256M` parameters.
- Total model ≈ 900M parameters.
- → **~28% of the entire model is a lookup table**, not reasoning layers. A quarter
  of the "capacity" is spent mapping tokens↔vectors.
- Measured (`bench_model_breakdown.py`, clean 3090): the LM head + CE is **~22% of
  forward+backward time** at our token count — the single most expensive op.

So the 200k vocab is expensive on **both** axes: parameters and compute. A tiny
vocab (256 bytes) would:
- shrink the table to `256 × 1280 = 0.33M` params → **free ~256M params** to either
  make the model smaller/cheaper OR add ~8–10 more real layers (or widen d_model),
- collapse the LM-head compute (predicting over 256 classes instead of 200000).

There is also a **quality** argument, specifically for German:
- Subword tokenizers fragment German **compounds** ("Donaudampfschifffahrts­gesell­schaft")
  and rich morphology inconsistently; a 200k multilingual vocab also spends most of
  its slots on the dominant languages, under-serving German sub-patterns.
- Byte/char-level has **no OOV, no tokenizer language bias, uniform handling of code,
  numbers, typos, scripts.** This is exactly the kind of thing that can move
  `bpb_german`, which is our actual bottleneck.

## 2. The naive byte idea — and its honest catch

Naive "just feed raw bytes" (vocab=256) has a real cost: German is ~4–5 bytes per
current token, so a 2048-token sequence becomes ~8k–10k **byte** positions. Every
mixer/FFN layer runs **per position**, so:

- mixer + FFN compute scales ~**4.5× up** (more positions),
- the LM head gets ~**170× cheaper** (256 vs 200k classes),
- net per-document FLOPs ≈ **~3× MORE expensive** than the token model. The longer
  sequence dominates; the cheap head does not compensate.

That is precisely why production models keep tokenizers — raw bytes trade compute
for purity. So naive bytes is *not* the design. The clever fix is patching.

## 3. The real design — dynamic byte patching (BLT-style), and why Mamba fits

Meta's **Byte Latent Transformer (BLT, 2024)** and **MambaByte** solve the cost: a
small, cheap byte-level model groups raw bytes into **variable-length patches**
(e.g., by next-byte entropy — split where the text gets "surprising"), and the big
backbone runs on the **patch** sequence, not raw bytes. You get byte-level inputs/
outputs with roughly **token-count** sequence lengths → byte benefits without the
4.5× blow-up.

Why this fits Helix specifically:
- We already run a **Mamba-2 + GLA hybrid**. Linear-time mixers (Mamba/GLA) are the
  natural backbone for long/byte sequences — no quadratic attention blow-up. The 6
  **sparse-attention** layers are the part that suffers most from long sequences and
  would need care (smaller window, or move them to operate on patches only).
- A small byte encoder/decoder + entropy patcher bolts onto the existing stack; the
  backbone stays mostly as-is, just fed patches.

## 4. Three concrete tiers (increasing ambition / risk)

| tier | what changes | params freed | compute | risk | German upside |
| --- | --- | ---: | --- | --- | --- |
| **A. Smaller vocab (e.g. 32k–48k)** | retrain SentencePiece smaller; same architecture | ~196M (32k) | ~same / slightly cheaper head | **low** | small — still subword |
| **B. Naive byte/char (vocab 256)** | tokenizer-free, raw bytes | ~256M | **~3× more** (longer seq) | medium | high (no OOV/compounds) but pays compute |
| **C. Patched byte (BLT/MambaByte-style)** | byte encoder + entropy patcher + backbone on patches | ~256M | ~comparable to tokens | **high** (new components, harder to train) | high, *and* compute-neutral |

- **Tier A** is the safe, boring win: a 32k vocab frees ~196M params and is a tiny,
  well-understood change. Likely the best risk-adjusted move if we just want the
  param/compute back. (Downside: still a tokenizer, still fragments German.)
- **Tier C** is the "krasse" version that matches the original intuition and could
  genuinely help German — but it is a real research build (entropy patcher, byte
  encoder/decoder, training stability), i.e. a from-scratch Helix v3, not a tweak.

## 5. The evaluation is *fair by construction* — bits-per-byte

Key elegance: our headline metric, **`bpb` (bits-per-byte), is tokenizer-independent.**
It normalizes by raw bytes, so a byte-level model and a token model can be compared
**directly** — there is no apples-to-oranges problem. We are already measuring
`bpb_german` / `bpb_english`. The whole question reduces to a clean experiment:

> Does a byte/small-vocab Helix reach a **lower `bpb_german`** at the same param and
> compute budget than the 200k-token Helix?

If yes → real win. If no → the tokenizer recipe stays. No hand-waving needed.

## 6. Cheap validation plan (when the 3090 / a GPU is free — NOT now)

Do this small before committing to a v3 build:
1. Train two **tiny** models (~50–100M) on the same German bytes, same compute:
   (a) 200k-token, (b) char/byte-level (Tier B, naive — simplest to stand up).
2. Compare `bpb_german` (fair, byte-normalized) and tokens/sec.
3. If byte-level's `bpb_german` is competitive *despite* the compute handicap, that
   is strong evidence Tier C (patched) — which removes the handicap — would win.
4. Only then prototype the entropy patcher (Tier C) at small scale.

This is days of small-model work, fully isolated from the production run.

## 7. Risks / honest verdict

- **It does not help the current run.** `bpb_german` today is data-limited, not
  representation-limited. A new vocab changes nothing about needing more German data.
- **Tier C is a genuine research bet.** Byte/patched models are recent and not yet a
  clearly-dominant recipe; training stability and the patcher are non-trivial.
- **Compute reality:** only Tier A (and Tier C *if* the patcher works) are
  compute-sane; naive bytes (B) costs ~3× and is only a research probe.
- **Best risk-adjusted first step:** Tier A (32k vocab) for the param/compute win,
  *or* the Tier-B tiny-model bpb probe to decide whether Tier C is worth it.

**Verdict:** Real, on-theme, and the bpb metric makes it cleanly testable — but it is
a **Helix v3 experiment for after** the current run and the German-data scale-up, not
a mid-run change. Capture now, test cheap later, build only if the tiny-model bpb
probe says byte-level beats tokens on German.
