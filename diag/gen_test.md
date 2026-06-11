# Checkpoint Generation Diagnosis

- checkpoint: `checkpoints/pretrain_1b_bilingual_de55_en45_foundation_warmstart_v3/step_9000.pt`
- step: `9000`
- backend: `mamba_ssm`

## Generations

### de_plain_berlin / greedy / rep=0.460
Prompt: `Berlin ist eine Stadt`

im Landkreis Potsdam-Mittelmark in Brandenburg. Die Stadt ist Sitz des gleichnamigen Landkreises im Landkreis Potsdam-Mittelmark in Brandenburg. Die Stadt ist Sitz des gleichnamigen Landkreises im Landkreis Potsdam-Mittelmark in der DDR. Die Stadt ist Sitz des gleichnamigen Kreises im Bezirk Potsdam-Mittelmark in der DDR. Der Ort liegt im Landkreis

### de_plain_berlin / sample_t0.8_k40 / rep=0.159
Prompt: `Berlin ist eine Stadt`

im Landkreis Vorpommern-Greifswald in Mecklenburg-Vorpommern. Das Dorf liegt in der Nähe von Bad Kösen im Landkreis Vorpommern-Greifswald in Mecklenburg-Vorpommern. Der gleichnamige Ortsteil von Bad Kösen, Mecklenburg-Vorpommern, der seit 1993 zusammen mit Bad Kösen auf der Halbinsel Malbork bei Berlin-Weißensee nach

### de_plain_wissen / greedy / rep=0.420
Prompt: `Die Hauptstadt von Deutschland ist`

ein beliebtes Ziel für die Region. Die Stadt ist ein wichtiger Wirtschaftsfaktor in der Region. Die Stadt ist ein wichtiger Wirtschaftsfaktor in der Region. Die Stadt ist ein wichtiger Wirtschaftsfaktor in der Region um die Stadt und die Region um die Stadt von der Insel Rügen. Die Stadt ist ein Zentrum des

### de_plain_wissen / sample_t0.8_k40 / rep=0.062
Prompt: `Die Hauptstadt von Deutschland ist`

für viele in der Entwicklung befindliche Unternehmen in der Industrie ein wichtiges Element, das unsere Gesellschaft seit einigen Jahren eine wichtige Rolle im Gesundheitssystem als Arbeitgeber. Die Führungspositionen der Firma in der Entwicklung und Herstellung von Elektromobilität – und damit auch die Geschäftsbereiche der Firma in der Region, in der jeder

### de_qa_capital / greedy / rep=0.479
Prompt: `Frage: Was ist die Hauptstadt von Deutschland?
Antwort:`

Wie ist die Lage in der Region? Die Stadt ist ein wichtiger Wirtschaftsstandort in der Region. Die Stadt ist ein wichtiger Wirtschaftsstandort. Die Stadt ist ein Zentrum der wissenschaftlichen Forschung und Entwicklung in der Region. Die Stadt ist ein Zentrum der wissenschaftlichen Forschung und Entwicklung in der Region um die

### de_qa_capital / sample_t0.8_k40 / rep=0.000
Prompt: `Frage: Was ist die Hauptstadt von Deutschland?
Antwort:`

Wie ist das Verhältnis von Familie, Beruf und Familie? Wie können wir die gesellschaftlichen Entwicklungen, die unsere Stadt prägen? Das gesellschaftliche und politische Verhältnisse sind von der wirtschaftlichen und wirtschaftlichen Entwicklung der Stadt ein besonderes Anliegen, und deshalb sind in der Tat in Deutschland die Kinder und Jugendlichen von

### de_instruction / greedy / rep=0.480
Prompt: `Schreibe einen einfachen deutschen Satz über Wasser:
`

. Die beiden anderen Beispiele sind die beiden anderen Beispiele: Die Vereinigten Staaten von Amerika und Kanada. Die USA haben sich in den USA und Kanada auf die Vereinigten Staaten von Amerika. Die USA haben sich in den USA und Kanada auf die Vereinigten Staaten von Amerika. Die USA haben sich in den

### de_instruction / sample_t0.8_k40 / rep=0.000
Prompt: `Schreibe einen einfachen deutschen Satz über Wasser:
`

: In diesem Fall sind die Fische in der Luft der Fall. Es muss sehr vorsichtig sein, denn in Wirklichkeit hat das Wasser das Wasser vom Meeresgrund der Elbe, das man auf das Wasser sieht. Wenn Sie eine Sehenswürdigkeit im Wasser schwimmen, ist der richtige Zeitpunkt, um das Schiff zu steuern, so

### en_plain_capital / greedy / rep=0.327
Prompt: `The capital of Germany is`

located in the city of Munich. The city of Munich is the largest city in the world, and the largest city in the world. The city of Munich is a city in the state of Bavaria, Germany. The city of Munich is a city in the United States. The city is known for its

### en_plain_capital / sample_t0.8_k40 / rep=0.019
Prompt: `The capital of Germany is`

the former German federal government in the US as well as in other countries of Europe. A group of scientists, including the Institute for Advanced Study at Columbia University in New York University, and the University of Maryland, College Park, the University of Florida. This report sets out the current status of the National

### en_instruction / greedy / rep=0.462
Prompt: `Write one simple sentence about water:
`

. The word "water" is a combination of the words "water" and "man"" and "people"" and "people"" and "people"" and "people"" and "people"" and "people"" and "people"" and "people""

### en_instruction / sample_t0.8_k40 / rep=0.000
Prompt: `Write one simple sentence about water:
`

n ; to cause an increase in the value of the cost of producing one product. The U.S. is not a government agency, and only states that are able to do business with the highest in quality, industry professionals. The e-book version of this article, a blog

## Top-K Next Tokens

### de_capital_next
Prompt: `Die Hauptstadt von Deutschland ist`
- `▁ein` -> `ein` p=0.0825
- `▁eine` -> `eine` p=0.0728
- `▁die` -> `die` p=0.0684
- `▁das` -> `das` p=0.0389
- `▁der` -> `der` p=0.0303
- `▁nicht` -> `nicht` p=0.0268
- `▁mit` -> `mit` p=0.0251
- `▁in` -> `in` p=0.0236
- `▁seit` -> `seit` p=0.0229
- `▁für` -> `für` p=0.0184
- `▁auch` -> `auch` p=0.0162
- `▁nach` -> `nach` p=0.0148

### de_faust_next
Prompt: `Faust wurde geschrieben von`
- `▁der` -> `der` p=0.0752
- `▁dem` -> `dem` p=0.0605
- `▁einem` -> `einem` p=0.0486
- `▁den` -> `den` p=0.0390
- `▁einer` -> `einer` p=0.0184
- `▁seinem` -> `seinem` p=0.0158
- `▁Heinrich` -> `Heinrich` p=0.0105
- `▁ihm` -> `ihm` p=0.0102
- `▁Hans` -> `Hans` p=0.0090
- `▁Karl` -> `Karl` p=0.0087
- `▁seiner` -> `seiner` p=0.0084
- `▁Peter` -> `Peter` p=0.0082

### de_water_next
Prompt: `Wasser ist bei Raumtemperatur`
- `▁flüssig` -> `flüssig` p=0.0672
- `▁und` -> `und` p=0.0557
- `.` -> `.` p=0.0523
- `▁nicht` -> `nicht` p=0.0407
- `,` -> `,` p=0.0263
- `en` -> `en` p=0.0247
- `▁in` -> `in` p=0.0205
- `▁` -> `` p=0.0160
- `▁zu` -> `zu` p=0.0160
- `▁gut` -> `gut` p=0.0124
- `▁bis` -> `bis` p=0.0120
- `▁aus` -> `aus` p=0.0120

### en_capital_next
Prompt: `The capital of Germany is`
- `▁located` -> `located` p=0.1773
- `▁the` -> `the` p=0.1076
- `▁a` -> `a` p=0.0372
- `▁now` -> `now` p=0.0360
- `▁situated` -> `situated` p=0.0255
- `▁also` -> `also` p=0.0233
- `▁called` -> `called` p=0.0193
- `▁in` -> `in` p=0.0150
- `▁Berlin` -> `Berlin` p=0.0128
- `▁at` -> `at` p=0.0103
- `▁currently` -> `currently` p=0.0091
- `▁still` -> `still` p=0.0091

## Contrastive Margins

Positive margin means correct continuation is preferred.

- **capital_berlin_vs_bonn** margin=1.2226 correct_nll=3.8002 wrong_nll=5.0228
- **faust_goethe_vs_hitler** margin=1.0360 correct_nll=4.4300 wrong_nll=5.4661
- **water_liquid_vs_metal** margin=-3.2822 correct_nll=9.2973 wrong_nll=6.0152
- **en_capital_berlin_vs_bonn** margin=1.1288 correct_nll=2.7684 wrong_nll=3.8973