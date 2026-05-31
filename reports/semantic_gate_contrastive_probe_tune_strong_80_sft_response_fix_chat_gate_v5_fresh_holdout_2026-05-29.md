# Semantic Response Gate: contrastive_probe_tune_strong_80_sft_response_fix_chat_gate_v5_fresh_holdout.json

- semantic_score: 0.833
- passed: 10 / 12

## By Category

- facts_de: 0.667
- hallucination_guard: 0.667
- honesty: 1.000
- qa_de: 1.000

## Failures

### yes_wien_capital
- keyword_score: 0.6
- issues: wrong_polarity:no_expected_yes, forbidden:nein
- answer: Nein. Wien ist die Hauptstadt von Österreich.

### no_hamburg_bavaria
- keyword_score: 0.6
- issues: missing:muenchen, forbidden:hamburg ist die hauptstadt
- answer: Nein. Hamburg ist die Hauptstadt von Bayern.
