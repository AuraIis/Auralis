# Checkpoint Generation Diagnosis

- checkpoint: `checkpoints/pretrain_1b_bilingual_de55_en45_foundation_warmstart_v3/step_15000.pt`
- step: `15000`
- backend: `mamba_ssm`

## Generations

### de_plain_berlin / greedy / rep=0.340
Prompt: `Berlin ist eine Stadt`

im Landkreis Potsdam-Mittelmark in Brandenburg. Die Stadt ist Sitz des Amtsgerichts Potsdam. Die Stadt Potsdam ist Sitz des Amtsgerichts Potsdam. Die Stadt Potsdam ist eine Stadt im Bezirk Potsdam-Mittelmark in Berlin. Die Stadt ist ein Ortsteil von Berlin im Bezirk Potsdam-Mittelmark. Die Stadt ist ein Ortsteil von Berlin im

### de_plain_berlin / sample_t0.8_k40 / rep=0.062
Prompt: `Berlin ist eine Stadt`

im äußersten Nordosten Polens. Die Stadt im südlichen Teil der Vereinigten Staaten und die Stadt La Jolla (US-Staat Illinois) ist ein Ort in den USA. Das Jahr 2018 hat sich deutlich verändert. Der größte Teil der Industrieproduktion befindet sich in den USA, die Stadt La Jolla ist mit durchschnittlich

### de_plain_wissen / greedy / rep=0.170
Prompt: `Die Hauptstadt von Deutschland ist`

ein beliebtes Ziel für Touristen, die sich für die Region und die Region, die sich in der Region um die Stadt und die Region von Paris, die Hauptstadt der Vereinigten Staaten von Amerika. Die Stadt ist ein Zentrum des Gesundheitswesens. Die Stadt ist ein Zentrum des Gesundheitswesens. Die Stadt

### de_plain_wissen / sample_t0.8_k40 / rep=0.000
Prompt: `Die Hauptstadt von Deutschland ist`

die Hauptstadt der Vereinigten Staaten von Amerika. Er wurde im Jahr 1946 gegründet. Der Ort ist berühmt für seine Weine und seine Weine: Es handelt sich hier um eine der wichtigsten und ältesten Rebflächen Europas. Die Weinproduktion wird durch den Anbau von Weißweinen, die Grundlage des französischen, den französischen und

### de_qa_capital / greedy / rep=0.358
Prompt: `Frage: Was ist die Hauptstadt von Deutschland?
Antwort:`

Die Stadt ist ein beliebtes Reiseziel für Touristen, die sich für die Region und die Region um die Stadt von der Hauptstadt der Vereinigten Staaten von Amerika. Die Stadt ist ein beliebtes Reiseziel für Touristen und Reisende. Die Stadt ist ein beliebtes Reiseziel für Touristen und ist ein beliebtes Reiseziel für Touristen. Die Stadt ist

### de_qa_capital / sample_t0.8_k40 / rep=0.064
Prompt: `Frage: Was ist die Hauptstadt von Deutschland?
Antwort:`

Wie sehen Sie die Situation in diesem Jahr? Die Situation in diesem Jahr ist nicht gerade geschäftig: die Arbeit hat sich verändert, da die Zahl der Flüchtlinge nach Deutschland über das Jahr verteilt. Im vergangenen Jahr wurde mit der Veröffentlichung der durchschnittlichen Zahl der Fälle von Suiziden in Deutschland,

### de_instruction / greedy / rep=0.122
Prompt: `Schreibe einen einfachen deutschen Satz über Wasser:
`

. Die meisten Menschen, die in der Schweiz leben, sind die meisten von uns, die sich für die Schweiz entscheiden, ob sie in der Schweiz oder in der Schweiz, die Schweiz, in der Schweiz und auch in der Schweiz, aber auch in der Schweiz, wo die Gemeinde von der Gemeinde mit der

### de_instruction / sample_t0.8_k40 / rep=0.000
Prompt: `Schreibe einen einfachen deutschen Satz über Wasser:
`

_ auf dem Computer in der Software, mit der man das richtige Maß an Sorgfalt, Präzision und dem richtigen Timing für das richtige Timing von der Maschine zu Hause. Wenn Sie sich nicht sicher sind, wie Sie die erste Version von .exe und .o. Ich habe mein iPhone 8 Plus

### en_plain_capital / greedy / rep=0.400
Prompt: `The capital of Germany is`

the city of Berlin. The city of Berlin is a city in the state of Israel. The city of Jerusalem is a city in the state of Israel. The city of Jerusalem is a city in the center of the family of the trinity of the same name. The trinity of the

### en_plain_capital / sample_t0.8_k40 / rep=0.000
Prompt: `The capital of Germany is`

now called Karlsruhe. The town is now referred to as “the city of the future,” a reference to the state and the fact that it has been a town since the Middle Ages and the first part of the “Sermo” to the present day” (I.P.C.B

### en_instruction / greedy / rep=0.161
Prompt: `Write one simple sentence about water:
`

(A) = A) is a function of the form $f(x) = \frac{x^2 + y^2}{x + y}$, where $f(x)$ is defined as $f(x) = \frac{x^2 + y^2}{d(x) - z)$

### en_instruction / sample_t0.8_k40 / rep=0.000
Prompt: `Write one simple sentence about water:
`

as the first part of the root word to describe the term "the future" or the words "new"" as we do not yet have an idea of what the world is getting older, but it's getting old. In a recent article published in the journal Physics Today, researchers at

## Top-K Next Tokens

### de_capital_next
Prompt: `Die Hauptstadt von Deutschland ist`
- `▁ein` -> `ein` p=0.0776
- `▁eine` -> `eine` p=0.0776
- `▁die` -> `die` p=0.0685
- `▁das` -> `das` p=0.0442
- `▁der` -> `der` p=0.0324
- `▁mit` -> `mit` p=0.0286
- `▁seit` -> `seit` p=0.0277
- `▁in` -> `in` p=0.0277
- `▁nicht` -> `nicht` p=0.0260
- `▁für` -> `für` p=0.0168
- `,` -> `,` p=0.0158
- `▁auch` -> `auch` p=0.0139

### de_faust_next
Prompt: `Faust wurde geschrieben von`
- `▁der` -> `der` p=0.0707
- `▁einem` -> `einem` p=0.0586
- `▁dem` -> `dem` p=0.0568
- `▁den` -> `den` p=0.0442
- `▁einer` -> `einer` p=0.0237
- `▁Heinrich` -> `Heinrich` p=0.0131
- `▁Peter` -> `Peter` p=0.0112
- `:` -> `:` p=0.0093
- `▁Karl` -> `Karl` p=0.0090
- `▁seinem` -> `seinem` p=0.0087
- `▁Hans` -> `Hans` p=0.0084
- `▁` -> `` p=0.0079

### de_water_next
Prompt: `Wasser ist bei Raumtemperatur`
- `▁nicht` -> `nicht` p=0.0799
- `▁flüssig` -> `flüssig` p=0.0622
- `▁und` -> `und` p=0.0485
- `.` -> `.` p=0.0313
- `,` -> `,` p=0.0276
- `en` -> `en` p=0.0259
- `▁in` -> `in` p=0.0178
- `▁sehr` -> `sehr` p=0.0168
- `▁` -> `` p=0.0168
- `▁zu` -> `zu` p=0.0168
- `▁stabil` -> `stabil` p=0.0115
- `▁kalt` -> `kalt` p=0.0105

### en_capital_next
Prompt: `The capital of Germany is`
- `▁the` -> `the` p=0.1870
- `▁located` -> `located` p=0.0830
- `▁a` -> `a` p=0.0473
- `▁now` -> `now` p=0.0392
- `▁situated` -> `situated` p=0.0296
- `▁called` -> `called` p=0.0230
- `▁also` -> `also` p=0.0216
- `▁in` -> `in` p=0.0127
- `▁currently` -> `currently` p=0.0116
- `▁at` -> `at` p=0.0102
- `▁not` -> `not` p=0.0096
- `▁still` -> `still` p=0.0090

## Contrastive Margins

Positive margin means correct continuation is preferred.

- **capital_berlin_vs_bonn** margin=0.9807 correct_nll=4.1543 wrong_nll=5.1349
- **faust_goethe_vs_hitler** margin=1.0974 correct_nll=4.4807 wrong_nll=5.5781
- **water_liquid_vs_metal** margin=-4.4679 correct_nll=9.8056 wrong_nll=5.3378
- **en_capital_berlin_vs_bonn** margin=1.0527 correct_nll=2.9563 wrong_nll=4.0090