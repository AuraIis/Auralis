# Semantic Response Gate: sft_response_fix_de_v9_stable_from_v8_20_8_sft_response_fix_chat_gate_v4_fresh_holdout.json

- semantic_score: 0.833
- passed: 10 / 12

## By Category

- facts_de: 1.000
- hallucination_guard: 0.667
- honesty: 1.000
- qa_de: 0.750

## Failures

### no_bonn_current
- keyword_score: 0.0
- issues: wrong_polarity:yes_expected_no, missing:berlin, forbidden:ja, forbidden:bonn ist die hauptstadt
- answer: Ja. Bonn ist die Hauptstadt von Deutschland.

### photosynthesis_definition
- keyword_score: 1.0
- issues: missing:pflanzen
- answer: Die Photosynthese ist eine chemische Reaktion auf Licht, bei der sich Licht aus Licht und Sauerstoff bildet.
