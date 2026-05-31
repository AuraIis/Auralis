# Semantic Response Gate: sft_response_fix_de_v3_from_a2_sweep_60_chat_gate_v2.json

- semantic_score: 0.167
- passed: 2 / 12

## By Category

- facts_de: 0.333
- hallucination_guard: 0.000
- honesty: 0.500
- qa_de: 0.000

## Failures

### yes_bern_capital
- keyword_score: 0.6
- issues: wrong_polarity:no_expected_yes, forbidden:nein
- answer: Nein. Die Hauptstadt der Schweiz ist Bern.

### no_bonn_current
- keyword_score: 0.6
- issues: missing:berlin, forbidden:bonn ist die hauptstadt
- answer: Nein. Bonn ist die Hauptstadt von Deutschland.

### no_hamburg_bavaria
- keyword_score: 1.0
- issues: missing:muenchen
- answer: Nein. Bayern ist die Hauptstadt von Bayern.

### goethe_not_mein_kampf_short
- keyword_score: 1.0
- issues: missing:hitler
- answer: Nein. Goethe schrieb das Gedicht "Der Kampf" stammt von Goethe.

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
- answer: Die Photosynthese ist die wichtigste Licht- und Wärme-Energie-Einheit.

### unknown_planet_behavior
- keyword_score: 0.0
- issues: missing:erfunden, missing_any_group:keine|nicht verlaesslich|keine verlaessliche
- answer: Ja. Der Planet Qorblax ist ein Stern ist ein Stern ist ein zeichen für die Farbe des Himmels.

### computer_definition
- keyword_score: 1.0
- issues: missing:maschine
- answer: Ein Computer ist ein Programm, das Daten ausliest und Programme ausliest.
