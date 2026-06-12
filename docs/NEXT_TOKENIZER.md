# Next-tokenizer note — add whitespace pieces for code (measured, ~free for German)

**Status:** finding + recommendation for the *next* tokenizer build. The current
Helix-200k tokenizer is frozen (a model is training on it) — this does **not**
apply to the running model.

## The finding

Helix-200k encodes code at 2.77 bytes/token (tab-indented), vs. ~3.55 for
Llama-3 / GPT-4o o200k. A category breakdown of the code tokens showed why:

| category        | % of code tokens | note |
|-----------------|------------------:|------|
| identifier      | 42.9%             | efficient (4.52 B/tok) |
| punct/operator  | 29.6%             | expected |
| **byte-fallback** | **22.5%**       | **1 byte each — pure waste** |
| whitespace      | 3.8%              | tab-indent already fixed this |
| number          | 1.2%              | |

The 22.5% byte-fallback is **99% tab + newline**:

```
0x09 \t   25,209  = 55.3% of byte-fallback
0x0a \n   20,000  = 43.9%
everything else            < 1%
```

Root cause: the vocab has **no piece for tab or newline** (both map to `<unk>` →
byte-fallback). Code is line-oriented and indented, so every `\n` and every `\t`
is spent as a 1-byte fallback token. SentencePiece splits training input on
newlines by default, so `\n` never becomes a piece — which is exactly how this
happened.

## The fix (measured)

A/B experiment: two 48k tokenizers trained on the *same* DE+EN+code corpus,
identical settings, the only difference being that variant B adds
`user_defined_symbols = ["\t", "\n", "\n\t", "\n\t\t", "\n\t\t\t", ...]`.

| sample  | A (baseline) | B (+whitespace) | Δ        |
|---------|-------------:|----------------:|----------|
| Code    | 2.44         | **2.73**        | **+12.0%** |
| German  | 4.36         | 4.36            | −0.0%    |
| English | 4.21         | 4.21            | −0.0%    |

**Conclusion:** whitespace/indent pieces buy measurable code density (+12% here)
at **zero cost to German/English** — because tabs/indent-runs do not occur in
prose. This is **not** the usual zero-sum vocab trade-off; these pieces are
orthogonal to the German advantage.

## Recommendation for the next tokenizer

Include structural-whitespace pieces:
- `\n` and `\t` as atomic pieces.
- newline+indent combos: `\n\t`, `\n\t\t`, … (and space variants `\n    `,
  `\n        ` for space-indented code), like GPT-4 / Llama tokenizers do.
- a few bare space-run pieces (`"    "`, `"        "`).
- train the tokenizer on code with newlines preserved (don't let `\n` be only a
  sentence delimiter).

### Caveats
- +12% is a **conservative floor**: minimal 8-symbol set, small 48k vocab, and
  baseline A already self-learned a few tab pieces. Against the production
  tokenizer (which has *zero* whitespace pieces) the gain should be larger, and a
  richer piece set would add more — not yet measured, so not claimed.
- This only affects encoding **density**, not code **capability** (correctness).
  Capability comes from code data + code SFT + code adapters, separately.

Reproduce: `scripts/eval/tokenizer_fertility.py` (fertility) and the A/B script
referenced in the project history.
