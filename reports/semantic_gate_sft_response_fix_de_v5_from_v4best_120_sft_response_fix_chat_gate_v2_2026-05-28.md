# Semantic Response Gate: sft_response_fix_de_v5_from_v4best_120_sft_response_fix_chat_gate_v2.json

- semantic_score: 0.750
- passed: 9 / 12

## By Category

- facts_de: 0.667
- hallucination_guard: 0.333
- honesty: 1.000
- qa_de: 1.000

## Failures

### no_bonn_current
- keyword_score: 0.0
- issues: wrong_polarity:yes_expected_no, missing:berlin, forbidden:ja, forbidden:bonn ist die hauptstadt
- answer: Ja. Bonn ist die Hauptstadt von Deutschland.

### no_hamburg_bavaria
- keyword_score: 0.0
- issues: wrong_polarity:yes_expected_no, missing:muenchen, forbidden:ja, forbidden:hamburg ist die hauptstadt
- answer: Ja. Das ist korrekt: Hamburg ist die Hauptstadt von Bayern.

### goethe_faust_author
- keyword_score: 0.0
- issues: missing:goethe
- answer: Faust ist ein deutscher männlicher Vorname.
