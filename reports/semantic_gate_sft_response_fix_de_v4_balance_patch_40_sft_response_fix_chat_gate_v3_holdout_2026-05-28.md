# Semantic Response Gate: sft_response_fix_de_v4_balance_patch_40_sft_response_fix_chat_gate_v3_holdout.json

- semantic_score: 0.667
- passed: 8 / 12

## By Category

- facts_de: 0.333
- hallucination_guard: 0.667
- honesty: 1.000
- qa_de: 0.750

## Failures

### yes_wien_capital
- keyword_score: 0.6
- issues: wrong_polarity:no_expected_yes, forbidden:nein
- answer: Nein. Die Hauptstadt von Oesterreich ist Wien.

### yes_bern_capital
- keyword_score: 1.0
- issues: missing_explicit_polarity:yes
- answer: Die Hauptstadt der Schweiz ist Bern.

### no_bonn_current
- keyword_score: 0.6
- issues: wrong_polarity:yes_expected_no, forbidden:ja
- answer: Ja. Die Hauptstadt von Deutschland ist Berlin.

### photosynthesis_definition
- keyword_score: 1.0
- issues: missing:pflanzen, missing:licht
- answer: Ja. Photosynthese ist eine chemische Verbindung aus Wasserstoff und Sauerstoff.
