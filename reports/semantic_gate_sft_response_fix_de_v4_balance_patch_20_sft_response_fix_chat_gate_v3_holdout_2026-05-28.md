# Semantic Response Gate: sft_response_fix_de_v4_balance_patch_20_sft_response_fix_chat_gate_v3_holdout.json

- semantic_score: 0.833
- passed: 10 / 12

## By Category

- facts_de: 0.667
- hallucination_guard: 0.667
- honesty: 1.000
- qa_de: 1.000

## Failures

### yes_bern_capital
- keyword_score: 1.0
- issues: missing_explicit_polarity:yes, missing:schweiz
- answer: Die Hauptstadt von Bern ist Bern.

### no_bonn_current
- keyword_score: 0.6
- issues: wrong_polarity:yes_expected_no, forbidden:ja
- answer: Ja. Die Hauptstadt von Deutschland ist Berlin.
