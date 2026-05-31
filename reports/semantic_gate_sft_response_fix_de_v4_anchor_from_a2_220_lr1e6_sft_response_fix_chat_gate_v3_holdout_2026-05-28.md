# Semantic Response Gate: sft_response_fix_de_v4_anchor_from_a2_220_lr1e6_sft_response_fix_chat_gate_v3_holdout.json

- semantic_score: 0.500
- passed: 6 / 12

## By Category

- facts_de: 0.333
- hallucination_guard: 0.000
- honesty: 1.000
- qa_de: 0.750

## Failures

### yes_bern_capital
- keyword_score: 1.0
- issues: missing_explicit_polarity:yes, missing:schweiz
- answer: Die Hauptstadt von Bern ist Bern.

### no_bonn_current
- keyword_score: 0.6
- issues: wrong_polarity:yes_expected_no, forbidden:ja
- answer: Ja. Die Hauptstadt von Deutschland ist Berlin.

### no_hamburg_bavaria
- keyword_score: 1.0
- issues: missing:muenchen
- answer: Nein. Die Landeshauptstadt ist Hamburg.

### goethe_not_mein_kampf_short
- keyword_score: 1.0
- issues: missing:hitler
- answer: Nein. Mein Kampf stammt von Johann Wolfgang von Goethe.

### goethe_faust_author
- keyword_score: 0.0
- issues: missing:goethe
- answer: Nein. Faust ist ein deutscher und englischer Schriftsteller.

### water_not_element
- keyword_score: 0.6
- issues: wrong_polarity:yes_expected_no, forbidden:ja
- answer: Ja. H2O ist die chemische Verbindung aus Wasserstoff und Sauerstoff.
