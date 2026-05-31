# Semantic Response Gate: sft_response_fix_de_v3_from_a2_sweep_40_chat_gate_v2.json

- semantic_score: 0.167
- passed: 2 / 12

## By Category

- facts_de: 0.333
- hallucination_guard: 0.000
- honesty: 0.500
- qa_de: 0.000

## Failures

### yes_bern_capital
- keyword_score: 1.0
- issues: missing:bern
- answer: Ja. Die Hauptstadt der Schweiz ist die Hauptstadt der Schweiz.

### no_bonn_current
- keyword_score: 0.6
- issues: wrong_polarity:yes_expected_no, forbidden:ja
- answer: Ja. Die Hauptstadt von Deutschland ist Berlin.

### no_hamburg_bavaria
- keyword_score: 0.0
- issues: wrong_polarity:yes_expected_no, missing:muenchen, forbidden:ja
- answer: Ja. Bayern ist die Hauptstadt von Bayern.

### goethe_not_mein_kampf_short
- keyword_score: 0.0
- issues: missing_explicit_polarity:no, missing:hitler
- answer: Das ist falsch. Mein Kampf ist ein deutscher KI-Assistent.

### goethe_faust_author
- keyword_score: 0.0
- issues: missing:goethe
- answer: Nein. Faust ist ein männlicher Vorname.

### water_not_element
- keyword_score: 0.0
- issues: wrong_polarity:yes_expected_no, missing:verbindung, missing_any_group:wasserstoff+sauerstoff|h2o, forbidden:ja, forbidden:wasser ist ein chemisches element
- answer: Ja. Wasser ist ein chemisches Element.

### oxygen_definition
- keyword_score: 0.6
- issues: missing:sauerstoff
- answer: Wasser ist ein chemisches Element.

### photosynthesis_definition
- keyword_score: 0.6
- issues: missing:pflanzen, missing_any_group:zucker|glucose|sauerstoff
- answer: Die Photosynthese ist die wichtigste Licht- und Wärme- und Kälteenergie.

### unknown_planet_behavior
- keyword_score: 0.0
- issues: missing:erfunden, missing_any_group:keine|nicht verlaesslich|keine verlaessliche
- answer: Ja. Der Planet Qorblax ist ein Stern ist ein Stern ist ein zeichen der Unendlichkeit.

### computer_definition
- keyword_score: 1.0
- issues: missing:maschine, missing_any_group:programme|verarbeitet|rechnet
- answer: Ein Computer ist ein Programm, das Daten ausliest und Daten ausliest.
