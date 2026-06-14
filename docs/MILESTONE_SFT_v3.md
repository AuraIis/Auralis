# Milestone: Helix v2 corrective SFT v3 — clean instruct model (2026-06-14)

First Helix that answers cleanly in chat: short correct answers, **stops**, routes math to a
tool, abstains honestly on unknowns, no tool over-trigger, no degeneration. Built on the
final pretraining checkpoint (`step_60000`, val_loss 1.843).

## Artifact (NOT in git — checkpoints are gitignored)
- Adapter: `checkpoints/sft_corrective_v3/adapter_best.pt` (173 MB) = LoRA r=64 +
  14 trained special-token embedding rows (ids 4–17). Base: `step_60000.pt`.
- Inference: load base + adapter, then overwrite embedding rows `emb_ids` with `emb_rows`.

## The two fixes that got here
1. **Targeted corrective data** (after SFT v1 drifted): heavy short-fact slice (636
   consensus-verified German facts via a multi-agent workflow) + non-tool anchors
   (lists/greetings/"number question ≠ calculation") + small tool/abstain + stability mix.
   → fixed tool over-trigger and over-abstain.
2. **EOS embedding fix** (the key one): `<|end|>` (id 7) never appeared in pretraining, so
   its tied-embedding row was untrained and LoRA freezes it → the model could never stop.
   Train ONLY the special-token rows (4–17) via a gradient mask, embedding-LR 3e-5.
   → stop_rate 0.0 → 1.0. See memory `sft-eos-embedding-fix`.

## Gate (7 mandatory metrics, strict greedy + hard <|end|> stop)
stop_rate 1.0 · known_facts 1.0 · false_abstain 0.0 · false_tool 0.0 · math_tool 1.0 ·
abstain_unknown 1.0 · short_answer 0.6.

## Free-chat regression (38 real prompts) — 36/38 clean
- stop_rate 0.974 · false_tool 0 · no_math_tool 0 · false_abstain 0 · no_abstain 0 ·
  degenerate 0 · too_long 0 · avg_len 123 chars.
- 2 remaining issues are KNOWLEDGE, not instruct: (a) a harder fact missed
  (Jupiter→"Merkur", a 0.9B reliability limit → use tools/retrieval for facts),
  (b) "Erkläre X" answers are weak/circular (Photosynthese/Atom) → archetype D not built yet.

## Behaviour summary
| Prompt type | Behaviour |
|---|---|
| Known fact | short answer + stop ("Berlin.") |
| Unknown entity | honest "Ich weiß nicht" + stop |
| Math | tool call `print(...)` (harness computes) + stop |
| Chat / list / number-fact | direct answer, NO tool + stop |

## Next (deliberately deferred)
Archetype D (explain) and H (grounded) via the same verified-data workflow; the saved
code-SFT comparison on the final base. Knowledge reliability stays a tool/retrieval problem,
not an SFT one, at 0.9B.
