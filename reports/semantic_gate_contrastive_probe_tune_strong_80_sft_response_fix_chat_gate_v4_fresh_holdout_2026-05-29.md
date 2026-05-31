# Semantic Response Gate: contrastive_probe_tune_strong_80_sft_response_fix_chat_gate_v4_fresh_holdout.json

- semantic_score: 0.750
- passed: 9 / 12

## By Category

- facts_de: 0.333
- hallucination_guard: 1.000
- honesty: 1.000
- qa_de: 0.750

## Failures

### yes_wien_capital
- keyword_score: 0.6
- issues: wrong_polarity:no_expected_yes, forbidden:nein
- answer: Nein. Die Hauptstadt von Österreich ist Wien.

### yes_bern_capital
- keyword_score: 0.6
- issues: wrong_polarity:no_expected_yes, forbidden:nein
- answer: Nein. Die Hauptstadt der Schweiz ist Bern.

### photosynthesis_definition
- keyword_score: 1.0
- issues: missing:pflanzen
- answer: Die Photosynthese ist eine chemische Verbindung zwischen Licht und Sauerstoff.
