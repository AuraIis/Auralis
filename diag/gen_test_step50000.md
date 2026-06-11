# Checkpoint Generation Diagnosis

- checkpoint: `checkpoints/pretrain_1b_bilingual_de55_en45_foundation_warmstart_v3/step_50000.pt`
- step: `50000`
- backend: `mamba_ssm`

## Generations

### de_plain_berlin / greedy / rep=0.200
Prompt: `Berlin ist eine Stadt`

im Landkreis Potsdam-Mittelmark. Die Stadt ist Sitz des Amtsgerichts Potsdam, das zum Land Brandenburg gehört und seit dem 1. Januar 2009 zur Stadt Potsdam gehört. Die Stadt ist Sitz des Amtsgerichts Potsdam. Das Amtsgericht Potsdam ist zuständig für den gleichnamigen Bezirk, in dem der Landkreis Potsdam-Mittelmark. Die Stadt ist ein

### de_plain_berlin / sample_t0.8_k40 / rep=0.042
Prompt: `Berlin ist eine Stadt`

mit ca. 40.000 Einwohnern und einer Fläche von ca. 28.000 km² im Großraum von London in England, der Regierungsbezirk ist die zweitgrößte Insel des Vereinigten Königreichs und der mit einer Fläche von 14.700 km² etwa 60.500 km² großen und 20.500 km² großen Inselgruppen umfasst. Eine

### de_plain_wissen / greedy / rep=0.296
Prompt: `Die Hauptstadt von Deutschland ist`

ein beliebtes Ziel für Touristen, die sich für die Region und die Region um die Stadt sind. Die Stadt ist ein beliebtes Ziel für Touristen und Einheimische. Die Stadt ist ein beliebtes Ziel für Touristen und Reisende. Die Stadt ist ein beliebtes Reiseziel für Touristen und die Strände von San Francisco sind die Strände von San

### de_plain_wissen / sample_t0.8_k40 / rep=0.000
Prompt: `Die Hauptstadt von Deutschland ist`

bekannt für ihre beeindruckende Architektur, ihre herausragende Lage am Atlantik und ihre Lage mitten in der Karibik, mit dem Atlantik ist ein Land, das sich von Mexiko bis Brasilien, von der Ost- bis zur Südhalbkugel der Erde. Das Land, in dem sich die meisten Menschen kennen, wie der US-Amerikaner

### de_qa_capital / greedy / rep=0.596
Prompt: `Frage: Was ist die Hauptstadt von Deutschland?
Antwort:`

Die Hauptstadt von Indien ist eine Stadt in der Provinz Punjab, Pakistan. Die Stadt ist eine der größten Städte der Welt. Die Stadt ist eine der größten Städte der Welt. Die Stadt ist eine der größten Städte der Welt. Die Stadt ist eine der größten Städte der Welt. Die Stadt ist eine der größten

### de_qa_capital / sample_t0.8_k40 / rep=0.000
Prompt: `Frage: Was ist die Hauptstadt von Deutschland?
Antwort:`

Ich habe meine Eltern in der Schweiz mit der Bitte um eine Frage zum Thema «Was ist eine Frau in der Politik? Eine Untersuchung zu seiner Bedeutung für den Wandel im öffentlichen Raum und dessen Auswirkungen auf die Entwicklung und den Wandel der Gesellschaft. Die Rolle von Frauen für die Region und die politische Kultur in der

### de_instruction / greedy / rep=0.295
Prompt: `Schreibe einen einfachen deutschen Satz über Wasser:
`

. Die meisten Menschen, die sich für die wissenschaftliche Arbeit interessieren, die sich mit der Frage nach der Bedeutung von Wasser in der Wasserwirtschaft beschäftigt. Die Wasserwirtschaft ist ein sehr komplexes Thema. Die Wasserwirtschaft ist ein sehr komplexes Thema. Die Wasserwirtschaft ist ein sehr komplexes Thema. Die

### de_instruction / sample_t0.8_k40 / rep=0.000
Prompt: `Schreibe einen einfachen deutschen Satz über Wasser:
`

] und einen deutschen Satz. In die Geschichte des englischen Textes. Wie in einigen anderen Ländern Europas werden auch Schweizer, die sich in ihrer Heimat seit einigen Jahren, seit Generationen, mit der Anerkennung des Schweizerdeutschen als Schweizer und einer gewissen Faszination für die Schweiz und mit der Schweiz zusammen. Die Schweizer Armee hat

### en_plain_capital / greedy / rep=0.265
Prompt: `The capital of Germany is`

the capital of the state of Baden-Württemberg. The city of Stuttgart is a city in the state of Baden-Württemberg. The city of Stuttgart is a city in the United States. The city of e-mail is a free software application that allows you to create and use the e-

### en_plain_capital / sample_t0.8_k40 / rep=0.018
Prompt: `The capital of Germany is`

located in the heart of Africa and the largest city in the state of New York. The state of Maine in the winter of 2005-2011. It was a long time ago, in a very long and extensive process with a long list of new inventions, new discoveries and advancements that have made us aware of

### en_instruction / greedy / rep=0.077
Prompt: `Write one simple sentence about water:
`

: The word "water" is a compound word that is used to describe the amount of time it takes to travel from one day to the next day to the time of day that a person is "at the beginning of a new year" and the length of a day, and the length of a

### en_instruction / sample_t0.8_k40 / rep=0.000
Prompt: `Write one simple sentence about water:
`

and water. The problem with these definitions is that it is hard to describe in a non-technical fashion, and as a result of the recent COVID-19 outbreak, there have been numerous instances when the use of the word “life skills” is to be taken seriously. However, we do our best

## Top-K Next Tokens

### de_capital_next
Prompt: `Die Hauptstadt von Deutschland ist`
- `▁ein` -> `ein` p=0.0815
- `▁eine` -> `eine` p=0.0719
- `▁die` -> `die` p=0.0719
- `▁das` -> `das` p=0.0410
- `▁in` -> `in` p=0.0329
- `▁nicht` -> `nicht` p=0.0309
- `▁seit` -> `seit` p=0.0282
- `▁der` -> `der` p=0.0264
- `▁für` -> `für` p=0.0219
- `▁mit` -> `mit` p=0.0213
- `▁auch` -> `auch` p=0.0160
- `,` -> `,` p=0.0160

### de_faust_next
Prompt: `Faust wurde geschrieben von`
- `▁der` -> `der` p=0.0702
- `▁den` -> `den` p=0.0514
- `▁einem` -> `einem` p=0.0498
- `▁dem` -> `dem` p=0.0400
- `▁einer` -> `einer` p=0.0267
- `▁Hans` -> `Hans` p=0.0104
- `:` -> `:` p=0.0104
- `▁Karl` -> `Karl` p=0.0098
- `▁seinem` -> `seinem` p=0.0098
- `▁Peter` -> `Peter` p=0.0089
- `▁Heinrich` -> `Heinrich` p=0.0081
- `▁ihm` -> `ihm` p=0.0079

### de_water_next
Prompt: `Wasser ist bei Raumtemperatur`
- `▁flüssig` -> `flüssig` p=0.1825
- `▁nicht` -> `nicht` p=0.0862
- `▁und` -> `und` p=0.0280
- `▁stabil` -> `stabil` p=0.0218
- `▁` -> `` p=0.0218
- `,` -> `,` p=0.0192
- `▁ein` -> `ein` p=0.0175
- `▁sehr` -> `sehr` p=0.0150
- `▁in` -> `in` p=0.0150
- `▁zu` -> `zu` p=0.0132
- `▁eine` -> `eine` p=0.0124
- `▁fest` -> `fest` p=0.0120

### en_capital_next
Prompt: `The capital of Germany is`
- `▁the` -> `the` p=0.1226
- `▁located` -> `located` p=0.0677
- `▁situated` -> `situated` p=0.0330
- `▁a` -> `a` p=0.0291
- `▁now` -> `now` p=0.0282
- `▁also` -> `also` p=0.0249
- `▁currently` -> `currently` p=0.0161
- `▁not` -> `not` p=0.0142
- `▁Berlin` -> `Berlin` p=0.0129
- `▁called` -> `called` p=0.0125
- `▁in` -> `in` p=0.0111
- `▁to` -> `to` p=0.0089

## Contrastive Margins

Positive margin means correct continuation is preferred.

- **capital_berlin_vs_bonn** margin=0.7802 correct_nll=4.0680 wrong_nll=4.8482
- **faust_goethe_vs_hitler** margin=1.0444 correct_nll=4.3627 wrong_nll=5.4070
- **water_liquid_vs_metal** margin=-4.4228 correct_nll=9.5516 wrong_nll=5.1288
- **en_capital_berlin_vs_bonn** margin=0.7718 correct_nll=2.5869 wrong_nll=3.3588