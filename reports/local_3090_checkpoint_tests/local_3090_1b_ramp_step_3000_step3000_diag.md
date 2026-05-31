# Checkpoint Generation Diagnosis

- checkpoint: `/mnt/c/Users/_Michael_/Auralis_3090_Test/checkpoints/pretrain_1b_bilingual_de55_en45_ramp/step_3000_step3000.pt`
- step: `3000`
- backend: `mamba_ssm`

## Generations

### de_plain_berlin / greedy / rep=0.394
Prompt: `Berlin ist eine Stadt`

im Norden von Berlin. Die Stadt ist seit dem 1. Januar 2006 mit der Stadt Berlin-Brandenburg in der Stadt Berlin-Brandenburg. Die Stadt ist seit dem 1. Januar 2006 mit der Stadt Berlin-Brandenburg

### de_plain_berlin / sample_t0.8_k40 / rep=0.035
Prompt: `Berlin ist eine Stadt`

. Die Stadt hat einen bedeutenden Wirtschaftszweig des Marktes – die Stadt ist über das Land Nordrhein-Westfalen (Schweiz) und die Stadt Bad Kreuznach. Die erste Autobahngesellschaft ist Bad Kreuznach, die zweite in

### de_plain_wissen / greedy / rep=0.100
Prompt: `Die Hauptstadt von Deutschland ist`

die Stadt mit ihren zahlreichen Sehenswürdigkeiten, die sich in der Nähe der Stadt befindet. Die Stadt ist seit dem 1. Januar 2014 Teil des Regierungsbezirks Köln-Mitte. Die Stadt ist seit dem

### de_plain_wissen / sample_t0.8_k40 / rep=0.000
Prompt: `Die Hauptstadt von Deutschland ist`

seit dem 1. April 2021 von der EU als EU-Staaten (EU-Staaten) angenommen. Die Europäische Kommission (WHO) vertritt die Vereinigten Staaten von Amerika, Kanada, die USA und Japan

### de_qa_capital / greedy / rep=0.143
Prompt: `Frage: Was ist die Hauptstadt von Deutschland?
Antwort:`

Die Stadt ist eine Stadt mit vielen Städten und Städten, die sich in der Regel in der Stadt oder in der Stadt angesiedelt. Die Stadt ist ein Ort der Ruhe und Erholung. Die Stadt ist ein Ort

### de_qa_capital / sample_t0.8_k40 / rep=0.062
Prompt: `Frage: Was ist die Hauptstadt von Deutschland?
Antwort:`

Das Klima ist in den letzten Tagen noch lange nicht so weit her. Also, was die Leute von den Vereinigten Arabischen Emiraten haben wir mit den Vereinigten Arabischen Emiraten, die den Aufstieg ins Land

### de_instruction / greedy / rep=0.000
Prompt: `Schreibe einen einfachen deutschen Satz über Wasser:
`

en, die sich auf die Suche nach einem neuen, neuen und neuen Artikel haben. Ich habe mir die Frage gestellt, ob ich die Frage nach dem neuen Artikel noch einmal auf die Idee kommen, dass

### de_instruction / sample_t0.8_k40 / rep=0.000
Prompt: `Schreibe einen einfachen deutschen Satz über Wasser:
`

e . Diese . Bitte schreiben, dann melde Dich hier mit einem . Du bist besuchbar und hast Interesse an einem . Er sucht ihn, wie er Dich in deinem Sex-Team?

### en_plain_capital / greedy / rep=0.353
Prompt: `The capital of Germany is`

the city of Berlin. The city of Berlin was built in the late 19th century, and the city of Berlin was the capital of the German Empire. The city of Berlin was the capital of the

### en_plain_capital / sample_t0.8_k40 / rep=0.000
Prompt: `The capital of Germany is`

the city of St. Moritz in German mythology. At the time of the First English translation of the Bible, the surname of the Greek, Greek, Turkish and Greek-Japanese telecommunications, the

### en_instruction / greedy / rep=0.080
Prompt: `Write one simple sentence about water:
`

. This is a simple example of how the term “serg” is used to describe the relationship between the two. The term “serg” refers to the term “serg

### en_instruction / sample_t0.8_k40 / rep=0.000
Prompt: `Write one simple sentence about water:
`

on the other hand refers to a number of things. This means that one can only imagine a way to have an adult from a different city without having to think about the state of California. Here are

## Top-K Next Tokens

### de_capital_next
Prompt: `Die Hauptstadt von Deutschland ist`
- `▁die` -> `die` p=0.1052
- `▁der` -> `der` p=0.0497
- `▁das` -> `das` p=0.0439
- `▁in` -> `in` p=0.0439
- `▁seit` -> `seit` p=0.0412
- `,` -> `,` p=0.0321
- `▁ein` -> `ein` p=0.0275
- `▁eine` -> `eine` p=0.0207
- `▁mit` -> `mit` p=0.0177
- `▁von` -> `von` p=0.0161
- `▁nicht` -> `nicht` p=0.0161
- `▁im` -> `im` p=0.0156

### de_faust_next
Prompt: `Faust wurde geschrieben von`
- `▁der` -> `der` p=0.0905
- `▁dem` -> `dem` p=0.0705
- `▁einem` -> `einem` p=0.0354
- `▁den` -> `den` p=0.0333
- `▁einer` -> `einer` p=0.0139
- `:` -> `:` p=0.0087
- `▁` -> `` p=0.0079
- `▁"` -> `"` p=0.0072
- `▁Michael` -> `Michael` p=0.0066
- `▁seinem` -> `seinem` p=0.0066
- `▁Peter` -> `Peter` p=0.0058
- `▁Hans` -> `Hans` p=0.0054

### de_water_next
Prompt: `Wasser ist bei Raumtemperatur`
- `en` -> `en` p=0.1561
- `▁und` -> `und` p=0.0395
- `▁in` -> `in` p=0.0371
- `▁nicht` -> `nicht` p=0.0371
- `▁bis` -> `bis` p=0.0327
- `▁zu` -> `zu` p=0.0289
- `▁im` -> `im` p=0.0199
- `▁von` -> `von` p=0.0199
- `▁nur` -> `nur` p=0.0186
- `▁die` -> `die` p=0.0186
- `▁sehr` -> `sehr` p=0.0175
- `▁eine` -> `eine` p=0.0175

### en_capital_next
Prompt: `The capital of Germany is`
- `▁the` -> `the` p=0.2748
- `▁a` -> `a` p=0.0205
- `▁in` -> `in` p=0.0165
- `▁Germany` -> `Germany` p=0.0155
- `▁Berlin` -> `Berlin` p=0.0100
- `▁called` -> `called` p=0.0086
- `▁St` -> `St` p=0.0067
- `▁located` -> `located` p=0.0065
- `▁Poland` -> `Poland` p=0.0052
- `▁now` -> `now` p=0.0049
- `▁an` -> `an` p=0.0047
- `▁The` -> `The` p=0.0040

## Contrastive Margins

Positive margin means correct continuation is preferred.

- **capital_berlin_vs_bonn** margin=1.6237 correct_nll=2.6853 wrong_nll=4.3090
- **faust_goethe_vs_hitler** margin=-0.3685 correct_nll=5.9651 wrong_nll=5.5966
- **water_liquid_vs_metal** margin=-2.6241 correct_nll=8.2515 wrong_nll=5.6273
- **en_capital_berlin_vs_bonn** margin=1.6825 correct_nll=2.8353 wrong_nll=4.5178