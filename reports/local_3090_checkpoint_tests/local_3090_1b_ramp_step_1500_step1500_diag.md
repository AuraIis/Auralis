# Checkpoint Generation Diagnosis

- checkpoint: `/mnt/c/Users/_Michael_/Auralis_3090_Test/checkpoints/pretrain_1b_bilingual_de55_en45_ramp/step_1500_step1500.pt`
- step: `1500`
- backend: `mamba_ssm`

## Generations

### de_plain_berlin / greedy / rep=0.543
Prompt: `Berlin ist eine Stadt`

in der Stadt. Die Stadt ist seit dem 1. Januar 2017 ein Teil der Stadt. Die Stadt ist seit dem 1. Januar 2016 ein Teil der Stadt. Die Stadt ist seit dem 1. Januar 2016 ein Teil

### de_plain_berlin / sample_t0.8_k40 / rep=0.000
Prompt: `Berlin ist eine Stadt`

in Tirol. Der Verein hat seine eigenen Einrichtungen, die sich für die Einrichtung und Verwaltung der öffentlichen Einrichtungen (B- und F-Bahn- und Wallfahrtskirchen in der Gemeinde Frankfurt, Hamburg, Hamburg

### de_plain_wissen / greedy / rep=0.219
Prompt: `Die Hauptstadt von Deutschland ist`

die größte Stadt in der Region. Die Stadt ist in der Provinz Pommern, die Stadt und die Stadt. Die Stadt ist in der Provinz Pommern. Die Stadt liegt in der Nähe von Ost-Berlin.

### de_plain_wissen / sample_t0.8_k40 / rep=0.029
Prompt: `Die Hauptstadt von Deutschland ist`

seit den 1990er Jahren ein wichtiger Teil der Regionen der Region. In der Provinz Schlesien sind die Städte und Gemeinden in der Provinz Polen sowie der Kreis der Republik Österreich die Gemeinden West-Berlin und der Provinz

### de_qa_capital / greedy / rep=0.353
Prompt: `Frage: Was ist die Hauptstadt von Deutschland?
Antwort:`

Die Stadt ist eine Stadt, die sich in der Stadt befindet. Die Stadt ist in der Nähe von der Stadt. Die Stadt liegt in der Nähe von der Stadt. Die Stadt liegt in der Nähe von

### de_qa_capital / sample_t0.8_k40 / rep=0.000
Prompt: `Frage: Was ist die Hauptstadt von Deutschland?
Antwort:`

Die Österreichischen Städte sind nicht sehr stark. Im Saarland, Sachsen-Anhalt, Baden-Württemberg und Bayern wurde in der DDR oft ein mittelalterliches Unternehmen. In der Folge gab es in den 1990

### de_instruction / greedy / rep=0.138
Prompt: `Schreibe einen einfachen deutschen Satz über Wasser:
`

, der in der Regel nicht in der Lage ist, die Luftfeuchtigkeit zu erreichen. Die Luftfeuchtigkeit ist in der Regel nicht zu erreichen. Die Luftfeuchtigkeit beträgt etwa 30 m. Die Höhe der

### de_instruction / sample_t0.8_k40 / rep=0.000
Prompt: `Schreibe einen einfachen deutschen Satz über Wasser:
`

ist also nicht zu sagen, dass der Begriff "Haus" und "die deutsche Übersetzung" die wissenschaftliche Formulierung "der Gegenwart" (etwa "die Wahrheit") die in der Zusammenstellung von den

### en_plain_capital / greedy / rep=0.657
Prompt: `The capital of Germany is`

the capital of the city of London. The city of London is located in the city of London. The city of London is located in the city of London. The city of London is located in the

### en_plain_capital / sample_t0.8_k40 / rep=0.000
Prompt: `The capital of Germany is`

called the "Gianl" (the "Uerh") is the "I've en-Got" (see "The"). In the en-Go-Go'

### en_instruction / greedy / rep=0.529
Prompt: `Write one simple sentence about water:
`

, the first person to be born and the second person to be born. The first person to be born is a child. The first person to be born is a child. The first person to be

### en_instruction / sample_t0.8_k40 / rep=0.139
Prompt: `Write one simple sentence about water:
`

the answer. If the number of students enrolled in a family of four in a group is a group of four students, and the average number of students is 2. The average number of students enrolled as a

## Top-K Next Tokens

### de_capital_next
Prompt: `Die Hauptstadt von Deutschland ist`
- `▁die` -> `die` p=0.0930
- `▁eine` -> `eine` p=0.0771
- `▁der` -> `der` p=0.0771
- `▁seit` -> `seit` p=0.0600
- `▁in` -> `in` p=0.0564
- `▁ein` -> `ein` p=0.0498
- `▁das` -> `das` p=0.0302
- `▁nicht` -> `nicht` p=0.0284
- `▁mit` -> `mit` p=0.0183
- `▁nur` -> `nur` p=0.0172
- `▁nach` -> `nach` p=0.0162
- `▁auch` -> `auch` p=0.0118

### de_faust_next
Prompt: `Faust wurde geschrieben von`
- `▁der` -> `der` p=0.0625
- `▁dem` -> `dem` p=0.0551
- `▁einem` -> `einem` p=0.0334
- `▁Johann` -> `Johann` p=0.0314
- `▁den` -> `den` p=0.0230
- `▁einer` -> `einer` p=0.0123
- `▁Hans` -> `Hans` p=0.0119
- `▁Dr` -> `Dr` p=0.0105
- `▁Peter` -> `Peter` p=0.0093
- `▁Karl` -> `Karl` p=0.0079
- `▁Richard` -> `Richard` p=0.0077
- `▁Daniel` -> `Daniel` p=0.0075

### de_water_next
Prompt: `Wasser ist bei Raumtemperatur`
- `▁von` -> `von` p=0.0541
- `▁nicht` -> `nicht` p=0.0477
- `▁und` -> `und` p=0.0372
- `▁nur` -> `nur` p=0.0349
- `▁bis` -> `bis` p=0.0328
- `▁eine` -> `eine` p=0.0289
- `▁sehr` -> `sehr` p=0.0255
- `▁zu` -> `zu` p=0.0199
- `▁in` -> `in` p=0.0187
- `▁der` -> `der` p=0.0176
- `,` -> `,` p=0.0165
- `▁die` -> `die` p=0.0155

### en_capital_next
Prompt: `The capital of Germany is`
- `▁the` -> `the` p=0.2198
- `▁a` -> `a` p=0.0714
- `▁located` -> `located` p=0.0407
- `▁in` -> `in` p=0.0359
- `▁one` -> `one` p=0.0204
- `▁also` -> `also` p=0.0192
- `▁to` -> `to` p=0.0180
- `▁not` -> `not` p=0.0170
- `▁now` -> `now` p=0.0150
- `▁an` -> `an` p=0.0117
- `▁called` -> `called` p=0.0109
- `▁part` -> `part` p=0.0109

## Contrastive Margins

Positive margin means correct continuation is preferred.

- **capital_berlin_vs_bonn** margin=1.4875 correct_nll=4.0821 wrong_nll=5.5695
- **faust_goethe_vs_hitler** margin=0.7536 correct_nll=6.0721 wrong_nll=6.8257
- **water_liquid_vs_metal** margin=-3.1123 correct_nll=8.5524 wrong_nll=5.4400
- **en_capital_berlin_vs_bonn** margin=1.3895 correct_nll=5.1095 wrong_nll=6.4991