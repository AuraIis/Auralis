# Auralis SFT Eval Gate Policy

## Disjoint evals are mandatory

A better SFT score is only trustworthy when the eval data is disjoint from the
training data. Disjoint here means: no identical or near-identical prompts,
answers, generator templates, seeds, source snippets, or answer patterns.

Reason: otherwise a model can learn superficial patterns and look good in the score,
without actually being able to do the task. With code SFT this can mean, for example,
that `def`, `return`, brackets, and keywords are rewarded even though the code does not
work correctly.

## Hard rule

- Before every larger SFT, first build a disjoint eval.
- Keep training data and eval data strictly separate.
- No eval from the same file, the same generator template, or the same seed questions as the SFT.
- Check prompt and answer similarity via hash and fuzzy matching.
- Code evals must be checked executably: syntax check, unit tests, expected output.
- Fact evals must come from different QA seeds/sources than the training data.
- Hallucination evals must contain traps that did not appear exactly in training.
- An automatic score alone is not enough: a manual spot check remains mandatory.

## Decision gate

An SFT run may only be considered better when all three points hold:

1. The disjoint eval improves.
2. The manual spot check feels better, not just more formal.
3. No regression in refusal, German, facts, and code basics.
