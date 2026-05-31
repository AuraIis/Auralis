# Auralis 500M German Response Repair - Hybrid/Neuro Status 2026-05-29

## Kurzfazit

Der neue Live-Lerntrace und die Neuro-Map funktionieren und sind hilfreich:
sie zeigen nicht nur Gate-Scores, sondern welche Zielantworten, Gegenantworten
und gefaehrlichen Kanten sich waehrend des Trainings bewegen.

Es gibt aber noch keinen besseren promotable Checkpoint als:

`checkpoints/sft_response_fix_de_v8_stable_from_v6_40_20/sft_smoke_step_20.pt`

Der Hybrid-Ansatz bewegt die kritischen Konzepte, aber nicht stabil genug. Vor
allem Photosynthese und Faust bleiben fragil; Bonn/Berlin kann durch zu starke
Repair-Daten wieder kippen. Deshalb keine Promotion von v10/v11/contrastive/v12.

## Neu gebaut

- `scripts/sft/hybrid_probe_sft_tune.py`
  - Kombiniert normales Assistant-SFT mit contrastive Probe-Loss.
  - Loss:
    `sft_weight * SFT + probe_weight * (target_nll + contrastive_weight * softplus(target_nll - negative_nll + margin))`
  - Schreibt Learning-Trace JSON/HTML und Neuro-Map HTML.

- `scripts/sft/run_hybrid_probe_sft.sh`
  - Wrapper fuer reproduzierbare Hybridlaeufe im `auralis-blackwell` Container.

- `scripts/data/build_sft_response_fix_de_v12_photo_faust_bridge.py`
  - Baut `data/training/sft_response_fix_de_v12_photo_faust_bridge`.
  - 936 Train Records, 59 Val Records.
  - 424 neue source-disjunkte Bridge-Records fuer Photosynthese und Faust.
  - Exakte Prompts aus Gate v2-v6 werden ausgeschlossen.

- `eval/learning_trace_de_repair_v2.yaml`
  - Erweiterte Lern-Probes.
  - Enthalten Zielantworten, die mit typischen falschen Satzanfaengen beginnen,
    z. B. `Die Photosynthese ist ...`, aber korrekt weitergehen.
  - Wichtig: Diese Datei ist Trainingsdiagnostik, keine Promotion-Eval.

## Erfolgreiche Smoke-Tests

Container-Hilfe und 2-Step-Smoke liefen erfolgreich:

- Checkpoint:
  `checkpoints/hybrid_probe_sft_smoke_2/hybrid_probe_sft_step_2.pt`
- Trace:
  `reports/learning_trace/hybrid_probe_sft_smoke_2.html`
- Neuro:
  `reports/learning_trace/hybrid_probe_sft_smoke_2_neuro.html`

Der Smoke bestaetigte die bekannten roten Kanten:

- Photosynthese: `licht aus licht`
- erfundene Entitaeten: Details wie Abkuerzung/Hintergrundfarbe
- Bern/Wien/Bonn-Polarity teilweise niedrige Margins

## Hybrid v1: v8 + vorhandene Probes

Checkpoint:

`checkpoints/hybrid_probe_sft_v1_40/hybrid_probe_sft_step_40.pt`

Trace:

- `reports/learning_trace/hybrid_probe_sft_v1_40.html`
- `reports/learning_trace/hybrid_probe_sft_v1_40_neuro.html`

Learning-Trace:

- Val-Loss: `0.6835 -> 0.6728`
- Photosynthese-Margin: `-0.550 -> -0.517`, weiterhin rot.
- Unknown-Entity wurde besser: forbidden hits verschwanden zeitweise.
- Bern blieb leicht negativ.

Semantic Gates:

- v2: 12/12
- v3: 12/12
- v4: 11/12, Fail `photosynthesis_definition`
- v5: 12/12
- v6: 11/12, Fail `goethe_faust_author`

Bewertung:

Nicht promoten. Nicht schlechter als v8 in den Hauptgates, aber kein klarer
Fortschritt gegen die bekannten Kernfehler.

## v12 Bridge 60: mehr source-disjunkte Photosynthese/Faust-Daten

Checkpoint:

`checkpoints/hybrid_probe_sft_v12_bridge_60/hybrid_probe_sft_step_60.pt`

Trace:

- `reports/learning_trace/hybrid_probe_sft_v12_bridge_60.html`
- `reports/learning_trace/hybrid_probe_sft_v12_bridge_60_neuro.html`

Learning-Trace:

- Photosynthese-Margin: `-0.550 -> -0.452`, also Bewegung in die richtige Richtung.
- Faust-Margin: `+0.223 -> +0.254`, leicht besser.
- Bonn-Margin fiel aber: `+0.176 -> +0.130`.

Semantic Gates:

- v2: 12/12
- v3: 12/12
- v4: 10/12
  - Fail `no_bonn_current`: `Ja. Bonn ist die Hauptstadt von Deutschland.`
  - Fail `photosynthesis_definition`: weiterhin falsche Licht-Formulierung
- v5: 12/12
- v6: 11/12
  - Fail `goethe_faust_author`: antwortet unsicher statt Goethe

Bewertung:

Nicht promoten. Mehr Konzeptdaten halfen Photosynthese im Trace, oeffneten aber
Bonn/Berlin wieder. Das ist genau die Art von Regression, die die Neuro-Map
sichtbar machen sollte.

## v12 Repair v2 80: erweiterte Decoder-Pfad-Probes

Checkpoint:

`checkpoints/hybrid_probe_sft_v12_repair_v2_80/hybrid_probe_sft_step_80.pt`

Trace:

- `reports/learning_trace/hybrid_probe_sft_v12_repair_v2_80.html`
- `reports/learning_trace/hybrid_probe_sft_v12_repair_v2_80_neuro.html`

Learning-Trace:

- Photosynthese-Core: forbidden `licht aus licht` verschwand ab Step 60/80.
- Aber die freie Antwort blieb semantisch instabil.
- `photosynthesis_plants_do` blieb stark negativ.
- `bonn_or_berlin_choice` blieb negativ, obwohl besser als Start.

Semantic Gates:

- v2: 12/12
- v3: 12/12
- v4: 9/12
  - Fail `no_bonn_current`: `Ja. Bonn ist die Hauptstadt von Deutschland.`
  - Fail `water_not_element`: richtige Einordnung, aber fehlendes explizites `Nein`
  - Fail `photosynthesis_definition`: `... Pflanzen mit Licht bilden`
- v5: 12/12
- v6: 11/12
  - Fail `goethe_faust_author`: `Ich kenne keine verlaesslichen ... Quellen.`

Bewertung:

Nicht promoten. Der Lauf beweist, dass wir den schlechten Phrase-Pfad beeinflussen
koennen, aber das Konzept ist noch nicht wirklich stabil gelernt. Zusaetzlich
wird Bonn wieder fragil.

## Diagnose

1. Das Modell kennt die Konzepte nicht robust genug.
   Photosynthese wird nicht nur falsch formuliert, sondern semantisch bruechig.
   Der Decoder faellt von korrekten Teilwoertern in alte oder generische Phrasen.

2. SFT allein und Contrastive allein sind zu lokal.
   Kleine Reparaturen verbessern einzelne Prompts, verschieben aber Polarity an
   anderer Stelle.

3. Honesty-Training ist inzwischen zu dominant fuer manche Faktenfragen.
   Bei Faust antwortet das Modell lieber unsicher, obwohl Goethe eine einfache
   stabile Tatsache ist.

4. Bonn/Berlin ist ein fragiler Polarity-Knoten.
   Sobald neue Fakten/QA-Daten staerker gewichtet werden, kann die alte Bonn-
   Antwort wieder auftauchen.

## Aktuelle Empfehlung

Nicht weiter versuchen, diese Fehler nur mit kleinen SFT-Patches glattzuziehen.
Der beste sichere Stand bleibt v8:

`checkpoints/sft_response_fix_de_v8_stable_from_v6_40_20/sft_smoke_step_20.pt`

Review-Update nach unabhaengiger Gegenpruefung:

Die haertere Schlussfolgerung ist korrekt: Bei 500M ist hier sehr wahrscheinlich
das Ende der Fahnenstange erreicht. Der Bonn-Kontrast war korrekt gebaut und
bestraft nicht pauschal `Bonn`; trotzdem kippt Bonn/Berlin, sobald Photosynthese
und Faust staerker gedrueckt werden. Das ist kein simpler Daten- oder Loss-Bug,
sondern katastrophale Interferenz: Konzept A wird repariert, Polarity B bricht.

Die richtige Loesung ist deshalb nicht ein weiterer 500M-Mini-Patch, sondern die
saubere Gewichtung im 1B-Pretrain/SFT-Mix. Fuer 500M bleiben die Ergebnisse
Diagnostik, nicht Promotion.

## Frozen Target/Retention Eval

Als Ersatz fuer reines `12/12`-Gate-Denken wurde eine eingefrorene Eval gebaut:

- Aktuelle Eval-Datei: `eval/sft_response_frozen_target_retention_v2.yaml`
- Vorgaenger: `eval/sft_response_frozen_target_retention_v1.yaml`
- Evaluator: `scripts/eval/frozen_response_gate.py`
- Leakcheck: `reports/frozen_response_gate_v2_leakcheck_sft_2026-05-29.json`

Prinzip:

- `target`: fragile Konzepte muessen auf neuen Formulierungen besser werden.
- `retention`: vorher korrekte oder wichtige Gegenfakten duerfen nicht kippen.
- Promotion nur wenn `target` voll besteht und `retention` null Regression hat.
- Eine Retention-Regression bedeutet: nicht promotable.

Hash-/Source-Disjunktheit:

- v2 umfasst 50 Probes: 25 Target, 25 Retention.
- SFT-Leakcheck gegen v1/v8/v10/v11/v12-SFT-Dateien:
  50 Prompts gegen 13,397 Trainingsprompts, 0 Kollisionen.
- 1B-Preflight-Leakcheck gegen vorhandene Pretrain/SFT-Kandidaten:
  70 Eval-Prompts, 382,763 Trainings-Einheiten, 0 Hash-Kollisionen,
  0 Substring-Hits.
- Zwei v2-Prompts wurden vor dem Freeze fuer diese Workspace entschaerft:
  Spanien kollidierte exakt mit SFT, Frankreich kam als Substring im Pretrain vor.

Frozen-Eval Ergebnisse:

| Checkpoint | Target | Retention | Promotable |
|---|---:|---:|---:|
| `v8_safe` | 8/25 | 18/25 | nein |
| `hybrid_v1_40` | 9/25 | 17/25 | nein |
| `hybrid_v12_bridge_60` | 10/25 | 17/25 | nein |
| `hybrid_v12_repair_v2_80` | 9/25 | 17/25 | nein |

Wichtigste Frozen-Fails:

- Target Photosynthese: Modell verweigert oder bleibt semantisch falsch.
- Retention Bonn historisch: Modell antwortet `Nein. Die Hauptstadt von Deutschland ist Berlin.`
  und loescht damit die historische Bonn-Wahrheit.
- Retention Faust/Mein Kampf: Modell sagt zwar `Nein`, haengt dann aber falsch
  `Mein Kampf wurde von Goethe geschrieben` an.
- Retention Known-Fact: Faust I/Goethe bleibt nicht robust confident.
- Retention unknown entity: Modell erfindet bei `Miralon` Details statt sauber zu verweigern.

Diese Eval zeigt noch haerter als v1: v8 ist nur der relativ stabilste
Diagnose-Stand, nicht ein guter Endstand. Hybrid/v12 verbessert Target minimal,
verliert aber Retention. Keine 500M-Variante ist promotable.

## Revidierte Empfehlung

1. Keine weiteren 500M-Mini-Patches auf Photosynthese/Faust/Bonn.
2. v8 als Diagnose-Basis behalten, nicht als final gutes Modell behandeln.
3. Frozen Target/Retention Eval als neue Promotion-Policy verwenden.
4. Fuer 1B:
   - Confident-correct-Anker fuer bekannte Fakten deutlich staerker einplanen.
   - Honesty/Refusal relativ zu Confident-Fakten deckeln.
   - Bonn temporal sauber halten: `Berlin heute`, `Bonn historisch wahr`.
   - Photosynthese als echtes Konzept im Pretrain/SFT-Mix staerken, nicht nur
     spaet per SFT-Patch.
5. Wenn weiter experimentiert wird, dann nur noch zur Diagnose, nicht mit dem
   Ziel, 500M zu promoten.

## Wichtige Artefakte

- Hybrid-Trainer: `scripts/sft/hybrid_probe_sft_tune.py`
- Hybrid-Runner: `scripts/sft/run_hybrid_probe_sft.sh`
- v12-Datenbuilder: `scripts/data/build_sft_response_fix_de_v12_photo_faust_bridge.py`
- v12-Daten: `data/training/sft_response_fix_de_v12_photo_faust_bridge`
- Repair-Probes: `eval/learning_trace_de_repair_v2.yaml`
- Frozen Target/Retention Eval: `eval/sft_response_frozen_target_retention_v2.yaml`
- Frozen Evaluator: `scripts/eval/frozen_response_gate.py`
- Beste sichere Basis: `checkpoints/sft_response_fix_de_v8_stable_from_v6_40_20/sft_smoke_step_20.pt`
