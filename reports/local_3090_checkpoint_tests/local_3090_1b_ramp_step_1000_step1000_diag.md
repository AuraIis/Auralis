# Checkpoint Generation Diagnosis

- checkpoint: `/mnt/c/Users/_Michael_/Auralis_3090_Test/checkpoints/pretrain_1b_bilingual_de55_en45_ramp/step_1000_step1000.pt`
- step: `1000`
- backend: `mamba_ssm`

## Generations

### de_plain_berlin / greedy / rep=0.097
Prompt: `Berlin ist eine Stadt`

im Landkreis. Die Stadt liegt im Landkreis und im Landkreis. Die Gemeinde liegt im Landkreis Baden-Württemberg. Die Gemeinde hat eine Fläche von rund Quadratkilometern und ist in der Region. Die Stadt liegt

### de_plain_berlin / sample_t0.8_k40 / rep=0.000
Prompt: `Berlin ist eine Stadt`

, die sich in einem anderen Land, in ein Dorf oder eine Stadt oder in eine andere Art der Region und das Theater des Landes (I.h. des Nationalsozialismus) finden sich die übrigen Länder

### de_plain_wissen / greedy / rep=0.625
Prompt: `Die Hauptstadt von Deutschland ist`

ein wichtiger Teil der wirtschaftlichen und kulturellen und kulturellen Entwicklung der wirtschaftlichen und kulturellen Entwicklung der wirtschaftlichen und kulturellen und kulturellen Entwicklung der wirtschaftlichen und kulturellen Entwicklung der wirtschaftlichen und kulturellen und wirtschaftlichen Entwicklung

### de_plain_wissen / sample_t0.8_k40 / rep=0.000
Prompt: `Die Hauptstadt von Deutschland ist`

seit Jahrzehnten die erste, die der größte Teil der Region ist und der Ort der Stadt seit der Gründung in Paris. Die Gemeinde gehört zur Stadt West, die Stadt der Region. Es gibt viele weitere Städte

### de_qa_capital / greedy / rep=0.261
Prompt: `Frage: Was ist die Hauptstadt von Deutschland?
Antwort:`

Die Veröffentlichung der Veröffentlichung ist nicht der Veröffentlichung des Berichterstatters. Die Veröffentlichung der Veröffentlichung ist nicht veröffentlicht. Die Veröffentlichung der Veröffentlichungen der Veröffentlichungen der Veröffentlichung

### de_qa_capital / sample_t0.8_k40 / rep=0.000
Prompt: `Frage: Was ist die Hauptstadt von Deutschland?
Antwort:`

Was gibt es für mehr als einer der wichtigsten Städte im Ausland, die mehr über das Wasser und die wirtschaftliche und den Alltag von Menschen mit dem Einsatz von Pflanzen, Pflanzen und Milch. Bei uns in den

### de_instruction / greedy / rep=0.000
Prompt: `Schreibe einen einfachen deutschen Satz über Wasser:
`

, die sich in der Vergangenheit als "unverssische" Person" bezeichnet, die "unverssische" Sprache" (z. B. "die "Verz."

### de_instruction / sample_t0.8_k40 / rep=0.000
Prompt: `Schreibe einen einfachen deutschen Satz über Wasser:
`

, Wärme und andere Flächen für das Risiko eines Holzs. Eine sichere, Luftfeuchtigkeit mit wirtschaftlicher oder wirtschaftlichen und wirtschaftlicher und Dienstleistungen für den Einsatz auf der Halbinsel. Die verschiedenen Größen für

### en_plain_capital / greedy / rep=0.371
Prompt: `The capital of Germany is`

the largest city in the world. The city of the United States is the largest city in the world. The city of the United States is located in the northern part of the United States. The area

### en_plain_capital / sample_t0.8_k40 / rep=0.057
Prompt: `The capital of Germany is`

the main source of the country. The United States is also the first to serve and the second language of the country from the United States. It is not the largest city in the world. The United

### en_instruction / greedy / rep=0.314
Prompt: `Write one simple sentence about water:
`

, the number of people who are not in the United States, and the number of people who are in the United States. The average age of the two people is the average age of the two people

### en_instruction / sample_t0.8_k40 / rep=0.000
Prompt: `Write one simple sentence about water:
`

the entire problem: \[ \text{Total number of people using the same number of marbles: \[ 4 \times 1 = \sqrt{9} \] \[ a = 3 \] \[ 2 = \boxed{6}

## Top-K Next Tokens

### de_capital_next
Prompt: `Die Hauptstadt von Deutschland ist`
- `▁ein` -> `ein` p=0.1011
- `▁die` -> `die` p=0.0892
- `▁eine` -> `eine` p=0.0787
- `▁seit` -> `seit` p=0.0787
- `▁in` -> `in` p=0.0613
- `▁der` -> `der` p=0.0478
- `▁nicht` -> `nicht` p=0.0240
- `▁im` -> `im` p=0.0212
- `▁mit` -> `mit` p=0.0199
- `▁das` -> `das` p=0.0199
- `▁für` -> `für` p=0.0165
- `▁von` -> `von` p=0.0165

### de_faust_next
Prompt: `Faust wurde geschrieben von`
- `▁der` -> `der` p=0.0929
- `▁dem` -> `dem` p=0.0770
- `▁einem` -> `einem` p=0.0467
- `▁einer` -> `einer` p=0.0387
- `▁den` -> `den` p=0.0387
- `▁Peter` -> `Peter` p=0.0177
- `▁seinem` -> `seinem` p=0.0138
- `▁ihm` -> `ihm` p=0.0092
- `▁Dr` -> `Dr` p=0.0079
- `▁` -> `` p=0.0076
- `▁ihrem` -> `ihrem` p=0.0074
- `▁seiner` -> `seiner` p=0.0074

### de_water_next
Prompt: `Wasser ist bei Raumtemperatur`
- `,` -> `,` p=0.0654
- `▁nicht` -> `nicht` p=0.0577
- `▁und` -> `und` p=0.0542
- `▁ein` -> `ein` p=0.0329
- `▁von` -> `von` p=0.0290
- `▁mit` -> `mit` p=0.0256
- `▁zu` -> `zu` p=0.0226
- `▁in` -> `in` p=0.0226
- `.` -> `.` p=0.0199
- `▁sehr` -> `sehr` p=0.0187
- `▁eine` -> `eine` p=0.0176
- `▁der` -> `der` p=0.0176

### en_capital_next
Prompt: `The capital of Germany is`
- `▁the` -> `the` p=0.1495
- `▁about` -> `about` p=0.0355
- `▁also` -> `also` p=0.0355
- `▁at` -> `at` p=0.0276
- `▁a` -> `a` p=0.0260
- `▁not` -> `not` p=0.0178
- `▁in` -> `in` p=0.0158
- `▁an` -> `an` p=0.0123
- `▁now` -> `now` p=0.0115
- `▁one` -> `one` p=0.0115
- `▁located` -> `located` p=0.0108
- `▁to` -> `to` p=0.0102

## Contrastive Margins

Positive margin means correct continuation is preferred.

- **capital_berlin_vs_bonn** margin=1.2970 correct_nll=5.4090 wrong_nll=6.7060
- **faust_goethe_vs_hitler** margin=0.1179 correct_nll=5.5102 wrong_nll=5.6281
- **water_liquid_vs_metal** margin=-3.3882 correct_nll=8.3126 wrong_nll=4.9243
- **en_capital_berlin_vs_bonn** margin=1.2696 correct_nll=6.3216 wrong_nll=7.5912