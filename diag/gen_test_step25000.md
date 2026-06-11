# Checkpoint Generation Diagnosis

- checkpoint: `checkpoints/pretrain_1b_bilingual_de55_en45_foundation_warmstart_v3/step_25000.pt`
- step: `25000`
- backend: `mamba_ssm`

## Generations

### de_plain_berlin / greedy / rep=0.184
Prompt: `Berlin ist eine Stadt`

mit rund 27.000 Einwohnern und ist Sitz der gleichnamigen Stadt- und Gemeindeverwaltung. Die Stadt ist Sitz des Amtsgerichts, das für die Verwaltung des Amtsgerichts in der Stadt Bad Kreuznach. Das Amtsgericht Bad Kreuznach, ein Gericht in Bad Kreuznach, das Amtsgericht Bad Kreuznach, ein Gericht in Deutschland, das Amtsgericht Bad Kreuznach

### de_plain_berlin / sample_t0.8_k40 / rep=0.000
Prompt: `Berlin ist eine Stadt`

in der Metropolregion Rhein-Neckar. Die Stadt wird von der Verbandsgemeinde Bergstraße als Bauherr mit seinen über 1500 Einwohnern nach Heidelberg gekommen. Am 20. April 2003 wurde im Rahmen einer umfassenden Sanierung des Gebäudes im alten Schulhaus der katholischen Pfarrkirche, die in den Jahren von 1468 bis 1640 von Karl IV. als Ort

### de_plain_wissen / greedy / rep=0.283
Prompt: `Die Hauptstadt von Deutschland ist`

die Hauptstadt des Bundesstaates New York. Die Stadt ist Sitz der Countyverwaltung von New York City. Die Stadt ist Sitz der Countyverwaltung von New York City. Die Stadt ist ein Zentrum des New Yorker Stadtteil Manhattans, in der Stadt New York City, New York City, USA, USA, USA,

### de_plain_wissen / sample_t0.8_k40 / rep=0.000
Prompt: `Die Hauptstadt von Deutschland ist`

der Sitz der Freien und Hansestadt Hamburg die Hansestadt Hamburg in der Hansestadt Hamburg. Das Zentrum für medizinische Versorgung (BMHD) ist ein Zusammenschluss der vier Hamburger Hochschulen, um die Hamburger Wirtschaft zu stärken. Eine Studie der Bertelsmann Stiftung hat hier die Chancen und die Bedingungen einer nachhaltigen Stadtentwicklung mit einem starken Service-

### de_qa_capital / greedy / rep=0.040
Prompt: `Frage: Was ist die Hauptstadt von Deutschland?
Antwort:`

Die Hauptstadt von Deutschland ist ein sehr wichtiger Ort, um die Stadt zu erkunden. Die Stadt ist ein Ort, an dem die Menschen in der Stadt von der Hauptstadt der Türkei, die Stadt Istanbul, die in der Türkei liegt. Die Stadt ist ein Zentrum für die Entwicklung der wissenschaftlichen Forschung und die

### de_qa_capital / sample_t0.8_k40 / rep=0.000
Prompt: `Frage: Was ist die Hauptstadt von Deutschland?
Antwort:`

In der Tat ist Deutschland im Vergleich zu den Niederlanden eine der höchsten Staatsquote in der EU. Nach der Wahl in der Türkei ist das Land ein starker Staat: Der Einfluss der türkischen Polizei wurde auf die Stadt durch die Einführung der Todesstrafe durch Präsident Erdogan in Bezug auf die Todesstrafe wurde zum ersten Mal im Jahr

### de_instruction / greedy / rep=0.188
Prompt: `Schreibe einen einfachen deutschen Satz über Wasser:
`

. Die meisten Menschen, die sich für die Arbeit mit dem Fahrrad in der Stadt interessiert sind. Die meisten Menschen, die sich für die Arbeit mit dem Fahrrad interessieren, und dies ist ein weiterer Grund, warum die Stadt in der Region um die Stadt von der Landeshauptstadt München, die Stadt München,

### de_instruction / sample_t0.8_k40 / rep=0.000
Prompt: `Schreibe einen einfachen deutschen Satz über Wasser:
`

, das man sich in den Sinn kommt, wenn man nicht nur an den Wassertropfen eine neue Flasche trinkt, sondern auch ein Glas Mineralwasser trinken. Die meisten Deutschen trinken weniger als andere. Die Flasche mit dem Namen „D.F.A.M.“ (das heißt, die Flasche wird zu Ehren

### en_plain_capital / greedy / rep=0.180
Prompt: `The capital of Germany is`

the city of Munich, Germany. The city is located in the heart of the Bavarian Alps, and is a member of the Bavarian Alps. The town of Munich is a city in the Austrian state of Bavaria. The town of Munich, Germany, is a city in the United States. The city is

### en_plain_capital / sample_t0.8_k40 / rep=0.000
Prompt: `The capital of Germany is`

currently in the process of turning the tide on the issue of the controversial proposal to ban redlining in the city of Birmingham. How could such a great idea could be made just as useful as one might expect from a “business as usual” model” to offer assistance” in getting to grips with

### en_instruction / greedy / rep=0.149
Prompt: `Write one simple sentence about water:
`

: "I'm not sure what I'm talking about here. I'm not sure what the answer is, but I'm sure you'll agree that the best way to get started with the most up-to-date information on the subject of the subject of the subject of

### en_instruction / sample_t0.8_k40 / rep=0.000
Prompt: `Write one simple sentence about water:
`

to the words of the lyrics are in the video below: The best way to get out of the cold. It’s time we took a little time to get back into the “why” and the “why”” of things to do” to a user. The final word. The

## Top-K Next Tokens

### de_capital_next
Prompt: `Die Hauptstadt von Deutschland ist`
- `▁ein` -> `ein` p=0.0923
- `▁die` -> `die` p=0.0923
- `▁eine` -> `eine` p=0.0814
- `▁das` -> `das` p=0.0560
- `▁in` -> `in` p=0.0339
- `▁seit` -> `seit` p=0.0281
- `▁nicht` -> `nicht` p=0.0264
- `▁der` -> `der` p=0.0233
- `▁mit` -> `mit` p=0.0212
- `▁für` -> `für` p=0.0171
- `,` -> `,` p=0.0160
- `▁auch` -> `auch` p=0.0155

### de_faust_next
Prompt: `Faust wurde geschrieben von`
- `▁der` -> `der` p=0.0697
- `▁dem` -> `dem` p=0.0560
- `▁den` -> `den` p=0.0560
- `▁einem` -> `einem` p=0.0479
- `▁einer` -> `einer` p=0.0256
- `▁seinem` -> `seinem` p=0.0104
- `▁Peter` -> `Peter` p=0.0100
- `▁Karl` -> `Karl` p=0.0097
- `:` -> `:` p=0.0094
- `▁Heinrich` -> `Heinrich` p=0.0089
- `▁Hans` -> `Hans` p=0.0089
- `▁ihm` -> `ihm` p=0.0076

### de_water_next
Prompt: `Wasser ist bei Raumtemperatur`
- `▁flüssig` -> `flüssig` p=0.0878
- `▁nicht` -> `nicht` p=0.0825
- `▁` -> `` p=0.0252
- `▁und` -> `und` p=0.0236
- `▁zu` -> `zu` p=0.0157
- `▁sehr` -> `sehr` p=0.0153
- `▁in` -> `in` p=0.0148
- `▁dunkel` -> `dunkel` p=0.0135
- `en` -> `en` p=0.0131
- `,` -> `,` p=0.0126
- `▁nur` -> `nur` p=0.0119
- `▁bei` -> `bei` p=0.0112

### en_capital_next
Prompt: `The capital of Germany is`
- `▁the` -> `the` p=0.1352
- `▁located` -> `located` p=0.0795
- `▁now` -> `now` p=0.0425
- `▁a` -> `a` p=0.0331
- `▁also` -> `also` p=0.0235
- `▁situated` -> `situated` p=0.0221
- `▁Berlin` -> `Berlin` p=0.0172
- `▁called` -> `called` p=0.0111
- `▁not` -> `not` p=0.0095
- `▁Cologne` -> `Cologne` p=0.0092
- `▁currently` -> `currently` p=0.0086
- `▁in` -> `in` p=0.0081

## Contrastive Margins

Positive margin means correct continuation is preferred.

- **capital_berlin_vs_bonn** margin=0.9201 correct_nll=4.0223 wrong_nll=4.9424
- **faust_goethe_vs_hitler** margin=0.8829 correct_nll=4.5047 wrong_nll=5.3876
- **water_liquid_vs_metal** margin=-3.7348 correct_nll=9.3155 wrong_nll=5.5807
- **en_capital_berlin_vs_bonn** margin=0.7746 correct_nll=2.4999 wrong_nll=3.2745