# Auralis 1B Readiness Plan - 2026-05-29

## Klartext

Der 1B-Lauf darf erst starten, wenn die Daten und Gates vorher gruen sind.
Der Fehler aus dem 500M-Lauf war nicht nur ein Modellproblem, sondern auch ein
Prozessproblem: zu spaet source-disjunkt geprueft, zu viel Netto-Score-Denken,
zu wenig Retention-Schutz. Das ist jetzt als Preflight/Canary-Policy umgesetzt.

## Neu gebaut

- 1B Readiness Gate:
  `eval/auralis_1b_readiness_gate_v1.yaml`
- Frozen SFT Target/Retention Gate:
  `eval/sft_response_frozen_target_retention_v2.yaml`
- 1B Preflight Config:
  `configs/eval/auralis_1b_readiness_preflight.yaml`
- 1B Preflight Script:
  `scripts/eval/one_b_readiness_preflight.py`
- Guarded Canary Training Config:
  `configs/training/pretrain_1b_canary_readiness.yaml`
- Guarded Canary Runner:
  `scripts/ops/run_pretrain_1b_canary_readiness.sh`

## Was das Gate prueft

Das Gate trennt hart zwischen:

- `target`: bekannte 500M-Schwachstellen muessen besser werden.
- `retention`: alte richtige Fakten duerfen nicht kippen.

Promotion-Regel:

- Target muss bestehen.
- Retention muss 0 Regressionen haben.
- Eine einzige Retention-Regression = nicht promotable.

Target-Achsen:

- Photosynthese als echtes Konzept, nicht nur gute Keywords.
- Faust/Goethe als confident known fact.
- Bonn frueher vs. Berlin heute.
- Bekannte Fakten beantworten, erfundene Dinge verweigern.

Retention-Achsen:

- Bonn historisch wahr halten.
- Berlin heute halten.
- Hamburg/Muenchen halten.
- Wien/Bern positive Polarity halten.
- Wasser/Sauerstoff Chemie halten.
- Goethe nicht mit Mein Kampf verwechseln.
- Faust I nicht verweigern.
- erfundene Entitaeten verweigern.
- einfache Mathe/Code/Computer-Fakten halten.

## Preflight-Ergebnis

Aktueller Report mit v2-Frozen-Gate:

`reports/auralis_1b_readiness_preflight_v2_2026-05-29.md`

Ergebnis:

- `ready_to_launch: False`
- Eval-Prompts: 70
- Trainings-Einheiten gescannt: 382,763
- Hash-Kollisionen: 0
- Substring-Hits: 0

Blocker:

- `configs/data_paths_1b_samples_container.yaml` hat noch leere `cleaned`-Listen.
- `configs/data_paths_1b_samples_container.yaml` hat noch keine `tokenized`-Pfade.

Vorheriger v1-Report:

`reports/auralis_1b_readiness_preflight_2026-05-29.md`

Ergebnis:

- `ready_to_launch: False`
- Eval-Prompts: 40
- Trainings-Einheiten gescannt: 382,763
- Hash-Kollisionen: 0
- Substring-Hits: 0

Blocker:

- `configs/data_paths_1b_samples_container.yaml` hat noch leere `cleaned`-Listen.
- `configs/data_paths_1b_samples_container.yaml` hat noch keine `tokenized`-Pfade.

Wichtig: v2 hatte in dieser Workspace-Gegenpruefung zuerst echte Disjunktheits-
Probleme, die behoben wurden:

- `retention_spain_capital` kollidierte exakt mit SFT-Trainingsprompts.
  Prompt wurde geaendert zu:
  `Welche Stadt ist Spaniens Regierungssitz und Hauptstadt?`
- `retention_france_capital` kam als Substring im Pretrain-Mix vor.
  Prompt wurde geaendert zu:
  `Welche franzoesische Stadt ist Regierungssitz und Hauptstadt?`

Danach:

- SFT-Leakcheck gegen v1/v8/v10/v11/v12-SFT-Dateien: 0 Kollisionen.
- 1B-Preflight-Leakcheck gegen die vorhandenen Kandidaten: 0 Hash-Kollisionen,
  0 Substring-Hits.
- Der Preflight blockt jetzt nicht nur leere Listen, sondern auch nicht
  existierende `cleaned`- und `tokenized`-Pfade inklusive fehlender `.idx` zu
  `.bin`.

Das ist gut: Die Gates leaken jetzt nicht gegen die vorhandenen v6/SFT-Kandidaten.
Der Start ist nur blockiert, weil der echte 1B-Clean/Tokenized-Mix noch nicht
eingetragen ist.

## 500M Baseline gegen v2

Das v2-Gate wurde auf den relevanten 500M-Checkpoints ausgefuehrt. Kein
Checkpoint ist promotable:

| Checkpoint | Target | Retention | Promotable |
|---|---:|---:|---:|
| `v8_safe` | 8/25 | 18/25 | nein |
| `hybrid_v1_40` | 9/25 | 17/25 | nein |
| `hybrid_v12_bridge_60` | 10/25 | 17/25 | nein |
| `hybrid_v12_repair_v2_80` | 9/25 | 17/25 | nein |

Interpretation:

- `v8_safe` bleibt nur relativ am stabilsten, weil Retention am wenigsten
  kaputtgeht.
- `hybrid_v12_bridge_60` verbessert Target minimal, verliert aber Retention.
- Keine 500M-Variante darf als geloest oder produktionsnah behandelt werden.
- Die v2-Eval bestaetigt die harte Diagnose: weitere 500M-Mini-Patches sind
  Diagnose-Arbeit, keine Promotion-Arbeit.

## Guarded Canary

Der Runner:

`scripts/ops/run_pretrain_1b_canary_readiness.sh`

macht absichtlich diese Reihenfolge:

1. `one_b_readiness_preflight.py`
2. `train_phase1.py --dry-run`
3. erst dann 1B-Canary-Training
4. danach automatisch `auralis_1b_readiness_gate_v1`
5. danach `frozen_response_gate.py` mit
   `eval/sft_response_frozen_target_retention_v2.yaml`

Solange der Preflight rot ist, startet kein Training.

## Adaptive Frozen-Gate Live Bridge

Der adaptive Curriculum-Trainer kann das v2-Frozen-Gate jetzt live mitlaufen
lassen:

```bash
python scripts/train/adaptive_curriculum.py \
  --model-config configs/model/helix_v2_1b.yaml \
  --curriculum configs/curriculum/helix_1b_curriculum_v1.yaml \
  --probes eval/adaptive_margin_probes_v1.yaml \
  --frozen-gate eval/sft_response_frozen_target_retention_v2.yaml \
  --output-dir runs/adaptive_1b_v1 \
  --batch-size 8 --seq-length 2048 --grad-accum 4 --max-steps 200000
```

Das schreibt zusaetzlich:

- `<output-dir>/frozen_gate_trace.jsonl`
- `frozen_target_pass`
- `frozen_retention_pass`
- `frozen_target_failures`
- `frozen_retention_failures`
- `frozen_promotable`

Lokale und Container-Tests sind gruen. Ein Mini-Smoke mit 500M-v8 und nur 2
neuen Tokens pro Antwort pruefte die Verdrahtung; die dabei entstehenden Scores
sind nicht inhaltlich relevant.

## Noch zu tun vor 1B-Start

1. Finalen 1B-Clean-Mix bauen.
   - `cleaned.german`, `cleaned.english`, `cleaned.code` in
     `configs/data_paths_1b_samples_container.yaml` befuellen.
   - Tokenized-Pfade eintragen.

2. Datenmix vor dem Tokenisieren auditen.
   - moderne deutsche QA/Fakten staerker als beim 500M.
   - Refusal/Honesty deckeln, nicht dominieren lassen.
   - Confident-correct Fakten explizit einplanen.
   - Books nur niedrig dosieren, nicht als Hauptwissenstraeger.

3. Preflight erneut laufen lassen.

```bash
python scripts/eval/one_b_readiness_preflight.py \
  --config configs/eval/auralis_1b_readiness_preflight.yaml \
  --output-json reports/auralis_1b_readiness_preflight_v2_$(date -u +%F).json \
  --output-md reports/auralis_1b_readiness_preflight_v2_$(date -u +%F).md
```

4. Erst wenn `ready_to_launch: true`, Canary starten:

```bash
bash scripts/ops/run_pretrain_1b_canary_readiness.sh
```

## Stop-Kriterien im 1B-Canary

Sofort stoppen/nicht promoten, wenn:

- Retention auch nur einen Fehler bekommt.
- Bonn historisch wegfaellt.
- Berlin heute kippt.
- Goethe/Faust verweigert oder falsch wird.
- Photosynthese nur gut klingt, aber Zucker/Sauerstoff/Pflanzen/Licht falsch verbindet.
- erfundene Entitaeten ausgeschmueckt werden.
- Target nicht besser wird, obwohl Loss sinkt.

## Entscheidung

Der erste 1B-Clean/Tokenized-Kandidat ist jetzt verdrahtet:

- `tokenized/curated_40b/english.bin`
- `tokenized/curated_40b/german.bin`

Aktueller Mix:

- 55% Englisch
- 45% Deutsch
- 0% Code

Der neue Preflight-Report ist gruen:

`reports/auralis_1b_readiness_preflight_curated_40b_v2_2026-05-29.md`

`train_phase1.py --dry-run` mit
`configs/training/pretrain_1b_canary_readiness.yaml` ist ebenfalls gruen.

Wir starten 1B damit nicht blind: Der Sicherheitsrahmen ist gebaut, der
Kandidatenmix ist verdrahtet, und der Guarded Runner scored nach dem Lauf jetzt
zusaetzlich das echte `eval/sft_response_frozen_target_retention_v2.yaml`.
