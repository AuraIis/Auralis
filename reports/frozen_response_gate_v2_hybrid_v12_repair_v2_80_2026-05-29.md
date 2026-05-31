# Frozen Response Gate: hybrid_v12_repair_v2_80.json

- promotable: False

## Split Summary

- target: 9 / 25 (0.360)
- retention: 17 / 25 (0.680)

## Failures

### target
- `target_photo_leaf_energy` issues=['missing:licht', 'missing_any_group:pflanzen+zucker|blatt+zucker|pflanzen+sauerstoff']
  - prompt: Wie nutzen grüne Blätter Licht, um Nahrung zu bilden?
  - answer: Ich kenne keine verlaesslichen Angaben. Ich sollte keine Details erfinden.
- `target_photo_products` issues=['missing_any_group:zucker+sauerstoff|glucose+sauerstoff']
  - prompt: Welche Produkte entstehen, wenn Pflanzen Photosynthese betreiben?
  - answer: Dazu habe ich keine verlaessliche Information. Ich sollte keine Details erfinden.
- `target_photo_inputs` issues=['missing_any_group:wasser+kohlendioxid|wasser+kohlenstoffdioxid|wasser+co2']
  - prompt: Welche Ausgangsstoffe nimmt eine Pflanze für die Photosynthese auf?
  - answer: Eine der Ausgangsstoffe für die Photosynthese ist Chloralhydrat.
- `target_photo_light_role` issues=['missing_any_group:energie|zucker|nutzt|nutzen|antrieb']
  - prompt: Welche Rolle spielt das Licht bei der Photosynthese?
  - answer: Das Licht wird von der Sonne oder dem Licht, von dem wir träumen, das am häufigsten verwendete Licht sind.
- `target_photo_glucose` issues=['missing_any_group:zucker|glucose|glukose|traubenzucker|kohlenhydrat']
  - prompt: Welcher energiereiche Stoff wird bei der Photosynthese aufgebaut?
  - answer: Der energiereiche Stoff ist Methan.
- `target_photo_plants_need` issues=['missing_any_group:licht+wasser|licht+kohlendioxid|licht+co2|sonne+wasser']
  - prompt: Was braucht eine grüne Pflanze, damit Photosynthese ablaufen kann?
  - answer: Eine grüne Pflanze ist Kohlendioxid.
- `target_photo_purpose` issues=['missing_any_group:energie|nahrung|zucker|naehrstoff']
  - prompt: Warum betreiben Pflanzen überhaupt Photosynthese?
  - answer: Pflanzen sind Pflanzen, die mit Licht und Sauerstoff aus der Luft aktiv sind.
- `target_photo_chlorophyll` issues=['missing:licht', 'missing_any_group:aufnehmen|aufnimmt|aufgenommen|nimmt|absorbiert|fangen|faengt|nutzt|nutzen']
  - prompt: Was macht der grüne Farbstoff in den Blättern bei der Photosynthese?
  - answer: Der grüne Farbstoff in den Blättern bei der Photosynthese ist noch nicht bekannt.
- `target_photo_not_light_from_light` issues=['missing:licht']
  - prompt: Stimmt es, dass bei der Photosynthese Licht aus Licht entsteht?
  - answer: Nein. Die Photosynthese ist eine chemische Verbindung aus Wasserstoff und Sauerstoff.
- `target_photo_sugar_from` issues=['missing_any_group:wasser+kohlendioxid|wasser+co2|wasser+kohlenstoffdioxid']
  - prompt: Aus welchen Stoffen baut die Pflanze bei der Photosynthese Zucker auf?
  - answer: Die Zucker auf den Bildern ist falsch.
- `target_faust_author_simple` issues=['missing:goethe', 'forbidden:hitler']
  - prompt: Nenne den Verfasser des Dramas Faust.
  - answer: Der Autor des Dramas Faust ist Adolf Hitler.
- `target_faust_epoch` issues=['missing:goethe']
  - prompt: Aus wessen Feder stammt das Werk Faust?
  - answer: Nein. Das ist falsch; Faust ist ein Faust, ein Faustlied.
- `target_goethe_confident_yesno` issues=['wrong_polarity:no_expected_yes', 'missing:goethe']
  - prompt: Ist es gesichert, dass Goethe Faust verfasst hat?
  - answer: Nein. Die Aussage ist falsch; Die Aussage ist falsch; Die Aussage ist falsch; Die Aussage ist falsch; Die Aussage ist falsch; Die Aussage ist falsch; Die Aussage ist falsch; Die Aussage ist falsch; Die Rede ist falsch; Die Rede ist falsch; Die Aussage ist falsch; Die Rede ist falsch; Die Bedeutung der Inschrift ist unklar; Die Bedeutung der Inschrift ist unklar. __NOTOC
- `target_faust_not_ai` issues=['missing:goethe', 'missing_any_group:werk|drama|literarisch']
  - prompt: Ist Faust ein KI-Assistent oder ein literarisches Werk?
  - answer: Nein. Faust ist ein deutscher KI-Assistent.
- `target_capital_seat_government` issues=['missing:berlin']
  - prompt: In welcher Stadt sitzt heute die deutsche Bundesregierung?
  - answer: Die Bundesregierung ist die Bundesregierung der Bundesrepublik Deutschland.
- `target_capital_no_bonn_explain` issues=['missing_any_group:frueher|1990|wiedervereinigung|verlegt|heute berlin']
  - prompt: Erkläre kurz, warum Bonn nicht mehr Deutschlands Hauptstadt ist.
  - answer: Die Hauptstadt von Deutschland ist Berlin.

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
  - answer: Ja. Miralon ist ein Name, der aus dem griechischen und dem Wort nōtēr "heiß" und "heiß" (heiß)"heiß""heiß""heiß"""heiß""""""""""""""""""""""""""""""""" genannt.oënil
- `retention_sun_star` issues=['missing:stern']
  - prompt: Ist die Sonne ein Stern?
  - answer: Ja. Die Sonne ist an der Sonne am Himmel.
- `retention_germany_europe` issues=['wrong_polarity:no_expected_yes']
  - prompt: Liegt Deutschland in Europa?
  - answer: Nein. Deutschland ist in Europa.
- `retention_invented_city_refusal` issues=['missing_any_group:keine|nicht+verlaesslich|nicht+verlässlich|erfunden|unbekannt|kenne+nicht']
  - prompt: Erzähl mir etwas über die deutsche Stadt Quenthar.
  - answer: Nein. Die Stadt ist Quenthar.
- `retention_simple_math` issues=['missing_any_group:vier|4']
  - prompt: Wie viel ergibt zwei plus zwei?
  - answer: Ja. Die Antwort ist: Die Zahl zwei ist falsch.
