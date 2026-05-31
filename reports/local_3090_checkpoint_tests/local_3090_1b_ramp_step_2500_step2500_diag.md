# Checkpoint Generation Diagnosis

- checkpoint: `/mnt/c/Users/_Michael_/Auralis_3090_Test/checkpoints/pretrain_1b_bilingual_de55_en45_ramp/step_2500_step2500.pt`
- step: `2500`
- backend: `mamba_ssm`

## Generations

### de_plain_berlin / greedy / rep=0.389
Prompt: `Berlin ist eine Stadt`

mit etwa 10.000 Einwohnern. Die Stadt ist ein Zentrum der Stadt und ein Zentrum der Stadt. Die Stadt ist ein Zentrum der Stadt und ein Zentrum für Kultur und Kultur. Die Stadt ist ein Zentrum der Kultur

### de_plain_berlin / sample_t0.8_k40 / rep=0.000
Prompt: `Berlin ist eine Stadt`

im Stadtteil Trier in Hessen. Der Ort liegt in einem Stadtteil von Bonn. Es wird im Norden der Stadt von der Grenze zu Deutschland zu Deutschland (Bayern) und der Schweiz). Das Gebiet um Nürnberg steht

### de_plain_wissen / greedy / rep=0.500
Prompt: `Die Hauptstadt von Deutschland ist`

die Hauptstadt von Deutschland. Die Stadt ist in der Region von der Provinz Posen, die Hauptstadt der Provinz Posen, die Hauptstadt der Provinz Posen, die Hauptstadt der Provinz Posen, die Hauptstadt der Provinz Posen,

### de_plain_wissen / sample_t0.8_k40 / rep=0.000
Prompt: `Die Hauptstadt von Deutschland ist`

seit jeher ein wichtiger Bestandteil der Region. Der Ort des ehemaligen Klosters mit seinen über 150 km langen und etwa 100 km² großen Burgen mit etwa 300 Quadratmetern und einer großen Parkanlage mit einem großen Kulturraum und

### de_qa_capital / greedy / rep=0.400
Prompt: `Frage: Was ist die Hauptstadt von Deutschland?
Antwort:`

Die Stadt ist ein Ort, an dem die Stadt und ihre Bewohner in der Region von der Stadt und ihren Gemeinden. Die Stadt ist ein Ort, an dem die Stadt und ihre Bewohner in der Region von

### de_qa_capital / sample_t0.8_k40 / rep=0.000
Prompt: `Frage: Was ist die Hauptstadt von Deutschland?
Antwort:`

"Die meisten von uns sind zu Hause, das ist die Welt, die die Zukunft der Stadt zu gestalten. Die Stadt hat eine gute Idee: Bei der Suche nach einer gesellschaftlicher Sicht zwischen den

### de_instruction / greedy / rep=0.062
Prompt: `Schreibe einen einfachen deutschen Satz über Wasser:
`

: "Ich habe mich nicht so richtig verstanden, dass ich mich nicht mehr so richtig verstanden habe. Ich habe mich für die Diskussion um die Frage, ob es sich bei der Frage, ob ein Mensch

### de_instruction / sample_t0.8_k40 / rep=0.030
Prompt: `Schreibe einen einfachen deutschen Satz über Wasser:
`

.de: Auf den Seiten über das Portal können keine Punkte auf dem Bildschirm, hier ist der Inhalt auf den Seiten der Seite. Das Problem ist, dass alle Teilnehmer sich auf einen Artikel mit dem Artikel

### en_plain_capital / greedy / rep=0.548
Prompt: `The capital of Germany is`

the city of St. Peter. The city of St. Peter is the largest city in the country. The city of St. Peter is the largest city in the country. The city of St.

### en_plain_capital / sample_t0.8_k40 / rep=0.000
Prompt: `The capital of Germany is`

found to be on the north side of the island of Gonzo in the east of Finland. The north of the country is the border border of the Republic of Finland. An important regional summit for the

### en_instruction / greedy / rep=0.303
Prompt: `Write one simple sentence about water:
`

: The first sentence is the number of words in the word, and the number of words in the word is the sum of the number of words in the word. The word "s" is the

### en_instruction / sample_t0.8_k40 / rep=0.167
Prompt: `Write one simple sentence about water:
`

: $f(0) = 10f(0) + 1f(0) + 0f(1) + 0f(2) = 11f(0) + 1f(0) = -1f(0) + 1f(2) = 5f(2)

## Top-K Next Tokens

### de_capital_next
Prompt: `Die Hauptstadt von Deutschland ist`
- `▁die` -> `die` p=0.0962
- `▁der` -> `der` p=0.0548
- `▁das` -> `das` p=0.0484
- `▁in` -> `in` p=0.0427
- `,` -> `,` p=0.0377
- `▁eine` -> `eine` p=0.0259
- `▁seit` -> `seit` p=0.0243
- `▁ein` -> `ein` p=0.0221
- `▁nicht` -> `nicht` p=0.0202
- `▁mit` -> `mit` p=0.0152
- `.` -> `.` p=0.0122
- `▁für` -> `für` p=0.0105

### de_faust_next
Prompt: `Faust wurde geschrieben von`
- `▁der` -> `der` p=0.0772
- `▁dem` -> `dem` p=0.0564
- `▁einem` -> `einem` p=0.0364
- `▁den` -> `den` p=0.0332
- `▁einer` -> `einer` p=0.0147
- `▁Michael` -> `Michael` p=0.0111
- `:` -> `:` p=0.0108
- `▁` -> `` p=0.0101
- `▁Peter` -> `Peter` p=0.0092
- `▁„` -> `„` p=0.0079
- `▁Andreas` -> `Andreas` p=0.0070
- `▁David` -> `David` p=0.0054

### de_water_next
Prompt: `Wasser ist bei Raumtemperatur`
- `▁und` -> `und` p=0.0428
- `▁nicht` -> `nicht` p=0.0355
- `▁zu` -> `zu` p=0.0313
- `,` -> `,` p=0.0215
- `▁bis` -> `bis` p=0.0202
- `▁in` -> `in` p=0.0202
- `.` -> `.` p=0.0202
- `▁die` -> `die` p=0.0190
- `▁eine` -> `eine` p=0.0178
- `▁(` -> `(` p=0.0168
- `▁sehr` -> `sehr` p=0.0162
- `▁im` -> `im` p=0.0157

### en_capital_next
Prompt: `The capital of Germany is`
- `▁the` -> `the` p=0.2957
- `▁a` -> `a` p=0.0376
- `▁now` -> `now` p=0.0235
- `▁called` -> `called` p=0.0235
- `▁located` -> `located` p=0.0214
- `▁in` -> `in` p=0.0126
- `▁at` -> `at` p=0.0111
- `▁also` -> `also` p=0.0087
- `▁part` -> `part` p=0.0076
- `▁` -> `` p=0.0072
- `▁on` -> `on` p=0.0065
- `▁one` -> `one` p=0.0058

## Contrastive Margins

Positive margin means correct continuation is preferred.

- **capital_berlin_vs_bonn** margin=1.8249 correct_nll=2.6789 wrong_nll=4.5037
- **faust_goethe_vs_hitler** margin=0.9902 correct_nll=5.6971 wrong_nll=6.6873
- **water_liquid_vs_metal** margin=-2.3598 correct_nll=8.2370 wrong_nll=5.8772
- **en_capital_berlin_vs_bonn** margin=1.7461 correct_nll=3.8690 wrong_nll=5.6150