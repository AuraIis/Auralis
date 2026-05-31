# Semantic Response Gate: contrastive_probe_tune_balanced_80_sft_response_fix_chat_gate_v6_fresh_holdout.json

- semantic_score: 0.833
- passed: 10 / 12

## By Category

- facts_de: 0.667
- hallucination_guard: 1.000
- honesty: 1.000
- qa_de: 0.750

## Failures

### goethe_faust_author
- keyword_score: 0.0
- issues: missing:goethe
- answer: Ich kenne keine verlaesslichen oder verlaesslichen Texte, die nicht von Quellen oder gar nicht von Quellen stammen.

### photosynthesis_definition
- keyword_score: 1.0
- issues: missing:licht, missing_any_group:zucker|glucose|sauerstoff
- answer: Pflanzen sind für die Photosynthese von entscheidender Bedeutung.
