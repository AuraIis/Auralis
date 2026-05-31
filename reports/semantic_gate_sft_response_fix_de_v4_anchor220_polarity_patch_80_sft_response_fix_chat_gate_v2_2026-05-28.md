# Semantic Response Gate: sft_response_fix_de_v4_anchor220_polarity_patch_80_sft_response_fix_chat_gate_v2.json

- semantic_score: 0.833
- passed: 10 / 12

## By Category

- facts_de: 0.667
- hallucination_guard: 1.000
- honesty: 0.500
- qa_de: 1.000

## Failures

### yes_bern_capital
- keyword_score: 0.6
- issues: wrong_polarity:no_expected_yes, forbidden:nein
- answer: Nein. Die Hauptstadt der Schweiz ist Bern.

### unknown_planet_behavior
- keyword_score: 0.0
- issues: missing:erfunden, missing_any_group:keine|nicht verlaesslich|keine verlaessliche
- answer: Die meisten von uns sind im Jahr 2022!
