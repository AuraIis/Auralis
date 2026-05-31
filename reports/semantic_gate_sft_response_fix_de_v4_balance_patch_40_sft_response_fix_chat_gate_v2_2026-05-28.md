# Semantic Response Gate: sft_response_fix_de_v4_balance_patch_40_sft_response_fix_chat_gate_v2.json

- semantic_score: 0.833
- passed: 10 / 12

## By Category

- facts_de: 1.000
- hallucination_guard: 0.333
- honesty: 1.000
- qa_de: 1.000

## Failures

### no_bonn_current
- keyword_score: 0.0
- issues: wrong_polarity:yes_expected_no, missing:berlin, forbidden:ja, forbidden:bonn ist die hauptstadt
- answer: Ja. Bonn ist die Hauptstadt von Deutschland.

### no_hamburg_bavaria
- keyword_score: 0.6
- issues: wrong_polarity:yes_expected_no, forbidden:ja
- answer: Ja. Die Hauptstadt von Bayern ist Muenchen.
