# Checkpoint Generation Diagnosis

- checkpoint: `/mnt/c/Users/_Michael_/Auralis_3090_Test/checkpoints/pretrain_1b_bilingual_de55_en45_ramp/step_2000_step2000.pt`
- step: `2000`
- backend: `mamba_ssm`

## Generations

### de_plain_berlin / greedy / rep=0.594
Prompt: `Berlin ist eine Stadt`

im Norden des Landes Nordrhein-Westfalen. Die Stadt ist ein Stadtteil von Hamburg. Die Stadt ist ein Stadtteil von Hamburg. Die Stadt ist ein Stadtteil von Hamburg. Die Stadt ist ein Stadtteil von Hamburg.

### de_plain_berlin / sample_t0.8_k40 / rep=0.000
Prompt: `Berlin ist eine Stadt`

im westafrikanischen Raum. Der mittelalterliche Dorfgemeinschaft der Stadt Zürich ist ein Ortsteil von Bern. Die Stadt bietet seit den 1990er Jahren unter dem Namen František-L’Amré do la

### de_plain_wissen / greedy / rep=0.613
Prompt: `Die Hauptstadt von Deutschland ist`

das Land. Die Stadt ist ein Zentrum der wirtschaftlichen Entwicklung der Region. Die Stadt ist ein Zentrum der wirtschaftlichen Entwicklung der Region. Die Stadt ist ein Zentrum der wirtschaftlichen Entwicklung der Region. Die

### de_plain_wissen / sample_t0.8_k40 / rep=0.030
Prompt: `Die Hauptstadt von Deutschland ist`

in der Schweiz das älteste erhaltene Kloster in Deutschland. Es gibt ein Museum in Wien, das auch der Kirche zu Ehren des damaligen Papstes der Älteren, das in der Schweiz und das von dem Nationalsozialismus

### de_qa_capital / greedy / rep=0.171
Prompt: `Frage: Was ist die Hauptstadt von Deutschland?
Antwort:`

Die Stadt ist ein Ort, in dem sich die Stadt und die Stadt mit ihren Dörfern und Gemeinden in der Region. Die Stadt ist ein Ort der historischen Stadt und der Stadt. Die Stadt ist ein Ort

### de_qa_capital / sample_t0.8_k40 / rep=0.000
Prompt: `Frage: Was ist die Hauptstadt von Deutschland?
Antwort:`

Sie haben die besten Artikel für Sie. Die Welt: Unser Land ist von der Welt der Menschen mit dem Leben. Aber wie ist das in einem Land? Nach dem Krieg werden die Leute in der Region

### de_instruction / greedy / rep=0.000
Prompt: `Schreibe einen einfachen deutschen Satz über Wasser:
`

, s. o. o. , , , , , , , , , , , , , , ,

### de_instruction / sample_t0.8_k40 / rep=0.000
Prompt: `Schreibe einen einfachen deutschen Satz über Wasser:
`

. Nach dem Willen der Vereinigten Staaten von Amerika ist die USA nicht zur Bekämpfung von Terrorismus unter Terrorismus gegen die USA. In der USA war der US-amerikanischen Regierung die USA bereits seit dem Zweiten Vatikanische

### en_plain_capital / greedy / rep=0.706
Prompt: `The capital of Germany is`

the city of Cologne. The city of Cologne is the city of Cologne, and the city of Cologne is the city of Cologne. The city of Cologne is the city of Cologne, and the city of

### en_plain_capital / sample_t0.8_k40 / rep=0.000
Prompt: `The capital of Germany is`

the city of Cologne, with the capital city of the Netherlands, which was the city center of the country. Since then, it is a time of great importance to the community and the new world,

### en_instruction / greedy / rep=0.138
Prompt: `Write one simple sentence about water:
`

, and the first word of the word, the word sn. The word sn. is a word that means the person who is in the same place is in the same place.

### en_instruction / sample_t0.8_k40 / rep=0.000
Prompt: `Write one simple sentence about water:
`

, s.k., for example; s.k., n.k.p., a.k.a. n.k., a.k.

## Top-K Next Tokens

### de_capital_next
Prompt: `Die Hauptstadt von Deutschland ist`
- `▁das` -> `das` p=0.0707
- `▁die` -> `die` p=0.0664
- `▁der` -> `der` p=0.0624
- `▁seit` -> `seit` p=0.0486
- `▁in` -> `in` p=0.0429
- `▁ein` -> `ein` p=0.0378
- `▁eine` -> `eine` p=0.0355
- `▁nicht` -> `nicht` p=0.0277
- `▁heute` -> `heute` p=0.0131
- `▁für` -> `für` p=0.0123
- `▁im` -> `im` p=0.0105
- `▁nur` -> `nur` p=0.0102

### de_faust_next
Prompt: `Faust wurde geschrieben von`
- `▁der` -> `der` p=0.1092
- `▁dem` -> `dem` p=0.0964
- `▁den` -> `den` p=0.0585
- `▁einem` -> `einem` p=0.0202
- `▁Dr` -> `Dr` p=0.0178
- `▁einer` -> `einer` p=0.0098
- `▁` -> `` p=0.0084
- `▁Hans` -> `Hans` p=0.0068
- `▁Karl` -> `Karl` p=0.0062
- `▁seinem` -> `seinem` p=0.0056
- `▁Herrn` -> `Herrn` p=0.0051
- `▁Hermann` -> `Hermann` p=0.0051

### de_water_next
Prompt: `Wasser ist bei Raumtemperatur`
- `en` -> `en` p=0.1169
- `▁nicht` -> `nicht` p=0.1098
- `▁sehr` -> `sehr` p=0.0356
- `▁zu` -> `zu` p=0.0335
- `▁und` -> `und` p=0.0230
- `▁von` -> `von` p=0.0191
- `▁nur` -> `nur` p=0.0179
- `▁die` -> `die` p=0.0158
- `▁bis` -> `bis` p=0.0158
- `▁ein` -> `ein` p=0.0149
- `▁durch` -> `durch` p=0.0140
- `▁in` -> `in` p=0.0131

### en_capital_next
Prompt: `The capital of Germany is`
- `▁the` -> `the` p=0.2153
- `▁a` -> `a` p=0.0544
- `▁located` -> `located` p=0.0511
- `▁in` -> `in` p=0.0213
- `▁called` -> `called` p=0.0194
- `▁part` -> `part` p=0.0151
- `▁now` -> `now` p=0.0142
- `▁situated` -> `situated` p=0.0133
- `▁also` -> `also` p=0.0095
- `▁home` -> `home` p=0.0086
- `▁one` -> `one` p=0.0081
- `▁that` -> `that` p=0.0078

## Contrastive Margins

Positive margin means correct continuation is preferred.

- **capital_berlin_vs_bonn** margin=1.6314 correct_nll=3.3437 wrong_nll=4.9750
- **faust_goethe_vs_hitler** margin=0.5195 correct_nll=5.7766 wrong_nll=6.2961
- **water_liquid_vs_metal** margin=-4.8187 correct_nll=9.9102 wrong_nll=5.0914
- **en_capital_berlin_vs_bonn** margin=1.6382 correct_nll=4.1303 wrong_nll=5.7685