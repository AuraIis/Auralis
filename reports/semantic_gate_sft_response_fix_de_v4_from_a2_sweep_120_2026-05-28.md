# Semantic Response Gate: sft_response_fix_de_v4_from_a2_sweep_120_chat_gate_v2.json

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
- answer: Ja. Die Hauptstadt der Schweiz ist Paris.

### no_bonn_current
- keyword_score: 0.6
- issues: wrong_polarity:yes_expected_no, forbidden:ja
- answer: Ja. Die Hauptstadt von Deutschland ist Berlin.

### no_hamburg_bavaria
- keyword_score: 0.0
- issues: wrong_polarity:yes_expected_no, missing:muenchen, forbidden:ja
- answer: Ja. Bayern ist die Hauptstadt von Bayern.

### goethe_not_mein_kampf_short
- keyword_score: 1.0
- issues: missing:hitler
- answer: Nein. Das ist falsch. Mein Kampf ist ein deutscher KI-Assistent.

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
- issues: missing:element
- answer: Wasser ist eine chemische Verbindung aus Sauerstoff und Stickstoff.

### photosynthesis_definition
- keyword_score: 0.6
- issues: missing:pflanzen, missing_any_group:zucker|glucose|sauerstoff
- answer: Die Photosynthese ist die wichtigste Licht- und Wärme- und Kälteenergie.

### unknown_planet_behavior
- keyword_score: 0.7142857142857143
- issues: missing:erfunden, missing_any_group:keine|nicht verlaesslich|keine verlaessliche
- answer: Ja. Der Name ist Qorblax.png" ist falsch. Stattdessen ist der Name "Ariel" und nicht "Ariel"" und "wird"" ist falsch. Der Begriff "ist" ist falsch. Der Begriff "ist" ist falsch. Der Begriff "ist" ist falsch. Die Aussage

### computer_definition
- keyword_score: 1.0
- issues: missing:maschine
- answer: Ein Computer ist ein Programm, das Daten ausliest und Programme ausliest.
