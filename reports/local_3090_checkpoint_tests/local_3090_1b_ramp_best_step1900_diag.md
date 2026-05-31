# Checkpoint Generation Diagnosis

- checkpoint: `/mnt/c/Users/_Michael_/Auralis_3090_Test/checkpoints/pretrain_1b_bilingual_de55_en45_ramp/best.pt`
- step: `1900`
- backend: `mamba_ssm`

## Generations

### de_plain_berlin / greedy / rep=0.226
Prompt: `Berlin ist eine Stadt`

im Norden der Stadt. Die Stadt ist in der Region Nord-Süd-Richtung von Berlin-Mitte. Die Stadt liegt im Süden der Stadt, die Stadt ist Teil der Stadt. Die Stadt liegt im

### de_plain_berlin / sample_t0.8_k40 / rep=0.000
Prompt: `Berlin ist eine Stadt`

in Nordböhmen mit etwa 70 Einwohnern. In den 1960er Jahren war das historische Nord-Süd-Gebiet zum Teil auch sehr gut erschlossen. Im 19. Jahrhundert wurde der Ort ein Teil des Westb

### de_plain_wissen / greedy / rep=0.367
Prompt: `Die Hauptstadt von Deutschland ist`

die Hauptstadt des Landes. Die Stadt ist in der Region Nord-Süd-Richtung von der Region Nord-Süd-Richtung von Berlin nach Berlin. Die Stadt ist in der Region Nord-Süd-Richtung

### de_plain_wissen / sample_t0.8_k40 / rep=0.000
Prompt: `Die Hauptstadt von Deutschland ist`

in den USA der Name des amerikanischen Wirtschaftssystems. In der deutschen Stadt ist der Name "Les hommes d'Histoire del néque la la Croix de Paris. In diesem Jahr wurde ein

### de_qa_capital / greedy / rep=0.423
Prompt: `Frage: Was ist die Hauptstadt von Deutschland?
Antwort:`

Die Vereinigten Arabischen Emirate ist ein Vereinigten Königreich. Die Vereinigten Arabischen Emirate sind die Vereinigten Arabischen Emirate. Die Vereinigten Arabischen Emirate sind die Vereinigten Arabischen Emirate (Chūs) und

### de_qa_capital / sample_t0.8_k40 / rep=0.000
Prompt: `Frage: Was ist die Hauptstadt von Deutschland?
Antwort:`

Der Deutsche Fußball-Bundesverband (DWD) hat in seiner Sitzung vom 8. November 2014 mit dem Bericht über die Durchführung der Konferenz zur Neuordnung der Menschenrechte in Brüssel zur Stärkung der Menschenrechte von der EU-

### de_instruction / greedy / rep=0.000
Prompt: `Schreibe einen einfachen deutschen Satz über Wasser:
`

. . . . . . . . . . . . . . . . . . . .

### de_instruction / sample_t0.8_k40 / rep=0.000
Prompt: `Schreibe einen einfachen deutschen Satz über Wasser:
`

. Als einen der folgenden Artikel habe ich mir die ganze Zeit die Diskussion um die richtige Antwort angesehen und wir sind gespannt über die Geschichte der letzten Jahrzehnte, während der letzten sechs Jahre die letzten 50 Jahre die

### en_plain_capital / greedy / rep=0.529
Prompt: `The capital of Germany is`

the city of the Netherlands. The city of the Netherlands is the capital of the Netherlands, and the city of the Netherlands, which is the capital of the Netherlands. The city of the Netherlands is the

### en_plain_capital / sample_t0.8_k40 / rep=0.094
Prompt: `The capital of Germany is`

the city of Copenhagen. The city of London is the town of St. Moritz, the town of Berlin and the city of Paris. There are some interesting places on this island (by St. Moritz

### en_instruction / greedy / rep=0.667
Prompt: `Write one simple sentence about water:
`

the t-shirt. The t-shirt is a t-shirt. The t-shirt is a t-shirt. The t-shirt is a t-

### en_instruction / sample_t0.8_k40 / rep=0.000
Prompt: `Write one simple sentence about water:
`

. That is, the most important question is to give women an answer to: - A new question, which is the answer, is that a child who is younger than one of three children is on

## Top-K Next Tokens

### de_capital_next
Prompt: `Die Hauptstadt von Deutschland ist`
- `▁die` -> `die` p=0.1129
- `▁in` -> `in` p=0.0729
- `▁der` -> `der` p=0.0605
- `▁das` -> `das` p=0.0390
- `▁eine` -> `eine` p=0.0304
- `▁seit` -> `seit` p=0.0286
- `▁ein` -> `ein` p=0.0286
- `▁nicht` -> `nicht` p=0.0222
- `▁nach` -> `nach` p=0.0209
- `▁mit` -> `mit` p=0.0184
- `▁im` -> `im` p=0.0184
- `▁heute` -> `heute` p=0.0168

### de_faust_next
Prompt: `Faust wurde geschrieben von`
- `▁der` -> `der` p=0.0446
- `▁dem` -> `dem` p=0.0347
- `▁Peter` -> `Peter` p=0.0124
- `▁den` -> `den` p=0.0113
- `▁Dr` -> `Dr` p=0.0109
- `▁Michael` -> `Michael` p=0.0103
- `▁einem` -> `einem` p=0.0085
- `▁Thomas` -> `Thomas` p=0.0082
- `▁Martin` -> `Martin` p=0.0073
- `▁Richard` -> `Richard` p=0.0073
- `▁Hermann` -> `Hermann` p=0.0071
- `▁Paul` -> `Paul` p=0.0068

### de_water_next
Prompt: `Wasser ist bei Raumtemperatur`
- `▁nicht` -> `nicht` p=0.0389
- `▁und` -> `und` p=0.0389
- `▁in` -> `in` p=0.0365
- `▁die` -> `die` p=0.0343
- `▁von` -> `von` p=0.0323
- `,` -> `,` p=0.0285
- `▁zu` -> `zu` p=0.0267
- `▁eine` -> `eine` p=0.0222
- `▁bis` -> `bis` p=0.0208
- `▁sehr` -> `sehr` p=0.0208
- `▁nur` -> `nur` p=0.0184
- `▁ein` -> `ein` p=0.0184

### en_capital_next
Prompt: `The capital of Germany is`
- `▁the` -> `the` p=0.2413
- `▁a` -> `a` p=0.0506
- `▁in` -> `in` p=0.0348
- `▁to` -> `to` p=0.0254
- `▁at` -> `at` p=0.0198
- `▁that` -> `that` p=0.0159
- `▁now` -> `now` p=0.0136
- `▁located` -> `located` p=0.0124
- `▁still` -> `still` p=0.0106
- `▁on` -> `on` p=0.0091
- `▁about` -> `about` p=0.0091
- `▁from` -> `from` p=0.0075

## Contrastive Margins

Positive margin means correct continuation is preferred.

- **capital_berlin_vs_bonn** margin=1.8980 correct_nll=3.4862 wrong_nll=5.3842
- **faust_goethe_vs_hitler** margin=1.2676 correct_nll=5.8602 wrong_nll=7.1278
- **water_liquid_vs_metal** margin=-4.1819 correct_nll=9.4201 wrong_nll=5.2382
- **en_capital_berlin_vs_bonn** margin=1.5415 correct_nll=4.6362 wrong_nll=6.1777