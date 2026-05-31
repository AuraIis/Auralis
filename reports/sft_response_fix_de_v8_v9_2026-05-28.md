# Auralis 500M Response-Fix SFT: v6-v9 Bericht

Datum: 2026-05-28

## Kurzfazit

Der aktuell beste konservative Kandidat ist:

`checkpoints/sft_response_fix_de_v8_stable_from_v6_40_20/sft_smoke_step_20.pt`

Warum: Er haelt die harten Gates v2/v3 voll, repariert den kritischen Bonn-Fehler aus v6 auf v4, und verschlechtert die Kernantworten nicht sichtbar. Er ist aber noch kein finaler "gut genug fuer Produktiv/SFT-Fortsetzung"-Checkpoint, weil auf neuen disjunkten Paraphrasen weiterhin einzelne echte Robustheitsfehler auftreten.

Nicht befoerdern:

- `sft_response_fix_de_v7_*`: keywordseitig gut, semantisch instabil.
- `sft_response_fix_de_v9_stable_from_v8_20_16`: besteht v6 fresh voll, regrediert aber bei v4 wieder auf "Ja. Bonn ist die Hauptstadt von Deutschland." Das ist ein No-Go.

## Geaenderte Artefakte

- `scripts/data/build_sft_response_fix_de_v7_bonn_photo_patch.py`
- `scripts/data/build_sft_response_fix_de_v8_stable_mix.py`
- `scripts/data/build_sft_response_fix_de_v9_stable_reinforce.py`
- `scripts/sft/run_response_fix_v7_sweep.sh`
- `scripts/sft/run_response_fix_v8_stable_sweep.sh`
- `scripts/sft/run_response_fix_v9_stable_sweep.sh`
- `eval/sft_response_fix_chat_gate_v5_fresh_holdout.yaml`
- `eval/sft_response_fix_chat_gate_v6_fresh_holdout.yaml`
- `scripts/eval/semantic_response_gate.py`

Das semantische Gate wurde korrigiert: Es verlangt explizites "Ja/Nein" nur noch bei echten Ja/Nein-Fragen. Bei Auswahl- oder Definitionsfragen wie "Welche Stadt ist ...?" ist die direkte korrekte Entitaet ausreichend.

## Ergebnisse

### v6 Bridge, vorher bester Stand

Checkpoint:

`checkpoints/sft_response_fix_de_v6_bridge_from_v5guardbal_40/sft_smoke_step_40.pt`

Semantic Gates nach korrigiertem Gate:

- v2 hard: 12/12
- v3 old holdout: 12/12
- v4 fresh: 10/12
- v5 fresh: 12/12
- v6 fresh: 11/12

Wichtige v4-Fehler:

- Bonn: antwortet falsch "Ja. Bonn ist die Hauptstadt von Deutschland."
- Photosynthese: falsche/defekte Definition ohne Pflanzen.

### v8 Stable Mix, bester konservativer Kandidat

Checkpoint:

`checkpoints/sft_response_fix_de_v8_stable_from_v6_40_20/sft_smoke_step_20.pt`

Training:

- Basis: v6_40
- Daten: v5 broad mix + v6 bridge + v7 Bonn/Photosynthese Patch
- 512 Train Records, 36 Families
- LR `1e-7`, 20 Steps, family-balanced sampler

Semantic Gates:

- v2 hard: 12/12
- v3 old holdout: 12/12
- v4 fresh: 11/12
- v5 fresh: 12/12
- v6 fresh: 11/12

Offene Fehler:

- v4 Photosynthese: "Die Photosynthese ist eine chemische Reaktion auf Licht, bei der sich Licht aus Licht und Sauerstoff bildet."
- v6 Faust: "Nein. Faust ist ein deutscher KI-Assistent."

Bewertung:

v8_20 ist besser als v6_40, weil der kritische Bonn-Fehler auf v4 repariert ist. Die verbleibenden Fehler sind aber echte Wissens-/Generalisierungsfehler, keine reinen Formatfehler.

### v9 Stable Reinforce

Bester v9:

`checkpoints/sft_response_fix_de_v9_stable_from_v8_20_16/sft_smoke_step_16.pt`

Semantic Gates:

- v2 hard: 12/12
- v3 old holdout: 12/12
- v4 fresh: 10/12
- v5 fresh: 12/12
- v6 fresh: 12/12

Problem:

v9 repariert den v6-Faust-Fall, aber regrediert auf v4 wieder bei Bonn:

`Ja. Bonn ist die Hauptstadt von Deutschland.`

Bewertung:

Nicht befoerdern. Ein Checkpoint, der bei Bonn/Deutschland-Hauptstadt wieder falsch wird, ist riskanter als v8_20.

## Ursache

Das Modell kann viele Einzelantworten korrekt reproduzieren, aber die Robustheit ueber Paraphrasen ist noch bruechig. Besonders empfindlich sind:

- Ja/Nein-Polarity bei Hauptstadtfragen.
- Weltwissen mit aehnlichen Entitaeten, z. B. Faust/Goethe.
- einfache naturwissenschaftliche Definitionen, z. B. Photosynthese.
- Honest-Answer-Formulierungen bei erfundenen Entitaeten.

Kleine Patch-SFTs koennen einzelne Gates verbessern, verschieben aber andere fragile Muster. Das ist ein Signal, dass wir nicht weiter Mikro-Patches auf denselben 12er-Gates optimieren sollten.

## Empfehlung

1. v8_20 als aktuellen Repair-Kandidaten sichern, aber nicht als final bezeichnen.
2. Keine weitere Mini-SFT-Runde auf denselben Gate-Familien starten.
3. Stattdessen ein groesseres, source-disjunktes SFT/Val-Setup bauen:
   - deutsche QA/Fakten-Erklaerungen mit echten Quellen,
   - kurze naturwissenschaftliche Basisdefinitionen,
   - Literatur/Autoren/Werke,
   - Hauptstadt/Geographie mit harten Negativfallen,
   - Honest-Answer-Daten mit erfundenen Entitaeten,
   - komplett getrennte Evalquellen.
4. Fuer Promotion mindestens drei Gates nutzen:
   - Hard regression gate mit bekannten No-Go-Fallen,
   - source-disjunktes Fresh-Gate,
   - manuelle Stichprobe mit freien Prompts.
5. Erst danach weiteres SFT auf dem 500M-Modell oder Wechsel auf 1B.

## Naechster technischer Schritt

Ein Data-Builder fuer ein groesseres source-disjunktes German QA/Instruction Repair-Set ist sinnvoller als ein weiterer 10-30-Step Patch. Ziel: nicht "Bonn/Photosynthese auswendig lernen", sondern die Antwortstruktur und Basiskonzepte breiter verankern.

Bis dahin ist `sft_response_fix_de_v8_stable_from_v6_40_20` der beste vorsichtige Stand.
