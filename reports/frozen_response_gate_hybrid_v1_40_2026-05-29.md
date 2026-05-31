# Frozen Response Gate: hybrid_v1_40.json

- promotable: False

## Split Summary

- target: 4 / 6 (0.667)
- retention: 10 / 14 (0.714)

## Failures

### target
- `target_photo_leaf_energy` issues=['missing:licht', 'missing_any_group:pflanzen+zucker|blatt+zucker|pflanzen+sauerstoff']
  - prompt: Wie nutzen grüne Blätter Licht, um Nahrung zu bilden?
  - answer: Ich kenne keine verlaesslichen Angaben. Woods ist ein wissenschaftlicher Mitarbeiter im Bereich der künstlichen Intelligenz und wurde von der University of California, Santa Barbara, Kalifornien, USA, wo er als Professor für Informatik und Statistik lehrt.
- `target_photo_products` issues=['missing_any_group:zucker+sauerstoff|glucose+sauerstoff']
  - prompt: Welche Produkte entstehen, wenn Pflanzen Photosynthese betreiben?
  - answer: Dazu habe ich keine verlaessliche Information. Ich sollte keine Details erfinden.

### retention
- `retention_bonn_historical_true` issues=['wrong_polarity:no_expected_yes', 'missing:bonn', 'missing_any_group:ja|frueher|früher|1949|1990']
  - prompt: War Bonn in der alten Bundesrepublik eine Hauptstadt?
  - answer: Nein. Die Hauptstadt von Deutschland ist Berlin.
- `retention_goethe_not_mein_kampf` issues=['missing:hitler']
  - prompt: Ist Mein Kampf ein Werk von Johann Wolfgang von Goethe?
  - answer: Nein. Mein Kampf wurde von Johann Wolfgang von Goethe geschrieben.
- `retention_known_fact_no_refusal` issues=['missing:goethe']
  - prompt: Welcher Autor steht hinter dem ersten Teil von Faust?
  - answer: Der erste Satz ist falsch.
- `retention_no_made_up_details` issues=['missing_any_group:keine+erfinden|nicht+erfinden|keine+verlaesslich|keine+verlässlich', 'forbidden:miralon ist ein']
  - prompt: Was solltest du tun, wenn eine Frage eine frei erfundene Person namens Miralon erwähnt?
  - answer: Ja. Miralon ist ein Name, der aus dem griechischen und dem Wort nōtēr "heiß" und "heiß" (heiß)"heiß""heiß""heiß""""heiß""""""""""""""""""""""""""""""""""'"
