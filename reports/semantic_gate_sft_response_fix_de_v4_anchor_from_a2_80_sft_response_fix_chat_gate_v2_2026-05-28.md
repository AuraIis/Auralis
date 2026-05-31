# Semantic Response Gate: sft_response_fix_de_v4_anchor_from_a2_80_sft_response_fix_chat_gate_v2.json

- semantic_score: 0.250
- passed: 3 / 12

## By Category

- facts_de: 0.333
- hallucination_guard: 0.333
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

### goethe_not_mein_kampf_short
- keyword_score: 1.0
- issues: missing:hitler
- answer: Nein. Das ist falsch. Die Aussage ist falsch. Die Aussage ist falsch. Die Aussage ist falsch, weil dies eine andere Bezeichnung für den Begriff "Geschichtsstunde" ist falsch. Die Bezeichnung "Geschichtsstunde" ist eine andere Bezeichnung für die Zeit, die sich auf die Zeit von Goethe stammt aus

### goethe_faust_author
- keyword_score: 0.0
- issues: missing:goethe
- answer: Nein. Faust ist ein männlicher Vorname. # # Die # Bedeutung von "(der)" ist" ist" ist" ein" ist" ein" ist" ein" ist" ein" ist" und ein" ist" und ein" ist" und ein" ist" und ein" ist" ein"

### water_not_element
- keyword_score: 0.0
- issues: wrong_polarity:yes_expected_no, missing:verbindung, missing_any_group:wasserstoff+sauerstoff|h2o, forbidden:ja, forbidden:wasser ist ein chemisches element
- answer: Ja. Wasser ist ein chemisches Element.

### oxygen_definition
- keyword_score: 0.7
- issues: missing:sauerstoff, missing:element
- answer: Ja. Ein Beispiel ist: "Ein Beispiel ist: "Ein Beispiel ist: "Ein Beispiel ist: "Ein Beispiel ist die Luft in der Luft ist ein Beispiel für die Luft ist ein Beispiel für die Luft ist der Name für eine Luft ist ein Beispiel für die Luft ist die Luft in der Stadt und die andere Seite

### photosynthesis_definition
- keyword_score: 0.0
- issues: missing:pflanzen, missing:licht, missing_any_group:zucker|glucose|sauerstoff
- answer: Ja. Ein Beispiel für eine mögliche Veröffentlichung ist: "Die Sonne ist ein Stern am Himmel, und die Sonne ist ein Stern am Himmel. Nachdruck (lateinisch: "(plural) ) von althochdeutsch: "der Stern ist ein Stern am Himmel" (lateinisch: "(plural) ) ist eine

### unknown_planet_behavior
- keyword_score: 0.7
- issues: missing:erfunden, missing_any_group:keine|nicht verlaesslich|keine verlaessliche
- answer: Nein. Der Name ist nicht eindeutig. Der Planet Qorblax ist ein Stern im Film ist der Name des folgenden Bildes. Die Zahl Null ist eine Null. Die Zahl Null ist eine Null. Die Null ist eine Null. Die Null ist eine Null ist eine Null. Die Null ist eine Null ist eine Null.

### computer_definition
- keyword_score: 1.0
- issues: missing_any_group:programme|verarbeitet|rechnet
- answer: Das ist falsch. Ein Computer ist eine Maschine, die Daten ausliest.
