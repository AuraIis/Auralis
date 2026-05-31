# Semantic Response Gate: sft_response_fix_de_v7_bonn_photo_from_v6_40_45_sft_response_fix_chat_gate_v4_fresh_holdout.json

- semantic_score: 0.667
- passed: 8 / 12

## By Category

- facts_de: 0.333
- hallucination_guard: 1.000
- honesty: 1.000
- qa_de: 0.500

## Failures

### yes_wien_capital
- keyword_score: 1.0
- issues: missing:wien
- answer: Ja. Oesterreich ist die Hauptstadt von Österreich.

### yes_bern_capital
- keyword_score: 0.6
- issues: wrong_polarity:no_expected_yes, forbidden:nein
- answer: Nein. Die Hauptstadt der Schweiz ist Bern.

### water_not_element
- keyword_score: 1.0
- issues: missing_explicit_polarity:no
- answer: Wasser ist eine chemische Verbindung aus Wasserstoff und Sauerstoff.

### photosynthesis_definition
- keyword_score: 1.0
- issues: missing_any_group:zucker|glucose|sauerstoff
- answer: Die Photosynthese ist eine chemische Reaktion auf Licht, bei der sich Pflanzen mit Licht bilden.
