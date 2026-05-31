# Semantic Response Gate: sft_response_fix_de_v5_family_from_v4best_100_sft_response_fix_chat_gate_v3_holdout.json

- semantic_score: 0.750
- passed: 9 / 12

## By Category

- facts_de: 0.333
- hallucination_guard: 0.667
- honesty: 1.000
- qa_de: 1.000

## Failures

### yes_wien_capital
- keyword_score: 0.6
- issues: wrong_polarity:no_expected_yes, forbidden:nein
- answer: Nein. Die Hauptstadt von Oesterreich ist Wien.

### yes_bern_capital
- keyword_score: 1.0
- issues: missing:schweiz
- answer: Ja. Bern ist die Hauptstadt von Bern.

### no_bonn_current
- keyword_score: 0.6
- issues: wrong_polarity:yes_expected_no, forbidden:ja
- answer: Ja. Die Hauptstadt von Deutschland ist Berlin.
