# Semantic Response Gate: sft_response_fix_de_v4_anchor_from_a2_220_lr1e6_sft_response_fix_chat_gate_v2.json

- semantic_score: 0.583
- passed: 7 / 12

## By Category

- facts_de: 0.667
- hallucination_guard: 0.000
- honesty: 1.000
- qa_de: 0.750

## Failures

### no_bonn_current
- keyword_score: 0.0
- issues: wrong_polarity:yes_expected_no, missing:berlin, forbidden:ja, forbidden:bonn ist die hauptstadt
- answer: Ja. Bonn ist die Hauptstadt von Deutschland.

### no_hamburg_bavaria
- keyword_score: 0.6
- issues: wrong_polarity:yes_expected_no, forbidden:ja
- answer: Ja. Das ist korrekt. Die Hauptstadt von Bayern ist Muenchen.

### goethe_not_mein_kampf_short
- keyword_score: 1.0
- issues: missing:hitler
- answer: Nein. Mein Kampf ist kein Buch, sondern ein Buch, das erfunden ist.

### goethe_faust_author
- keyword_score: 0.0
- issues: missing:goethe
- answer: Das ist falsch. Faust ist ein deutscher KI-Assistent.

### water_not_element
- keyword_score: 0.0
- issues: wrong_polarity:yes_expected_no, missing:verbindung, missing_any_group:wasserstoff+sauerstoff|h2o, forbidden:ja, forbidden:wasser ist ein chemisches element
- answer: Ja. Wasser ist ein chemisches Element und ein chemisches Element ist Wasser.
