# STATUS - Auralis v2

Stand: 2026-06-07

Dies ist die aktuelle Kurz-Wahrheit fuer das Repo. Wenn alte Phasenplaene,
April-/Mai-Statusstaende oder Specs widersprechen, gilt zuerst diese Datei,
dann die Reports vom 2026-05-29, dann die jeweilige Arbeitsdoku.

## Update 2026-06-07 — Tool-Use Mathe END-TO-END (verifiziertes Rechnen statt Raten)

NEUESTER STAND. Helix loest Arithmetik jetzt ueber ein verifiziertes externes Tool,
statt im Kopf zu raten. Strukturell geloest: aus "12 + 15 = 12" (raten) wird
`<tool:python>print(12+15)</tool>` -> Executor 27 -> "12 + 15 ergibt 27."

Gebaut (alles KEY-FREI / self-generating; der Rechner ist die Ground Truth):
- `scripts/sft/tool_harness.py` — AST-Whitelist-Rechner (kein RCE, Selftest 14/14) +
  Generierungs-Loop mit `</tool>`-Stop-Sequenz + Result-Injektion + Resume.
- `scripts/sft/gen_tool_traces.py` — Tool-SFT-Traces (modes call_only/full, --simple-rebump).
- `scripts/sft/tool_gate.py` — DUALES Gate (Mathe->Tool, Fakten->kein Tool) + End-to-End
  (`--mode full`: result_usage_rate, answer_numeric_match), Typ-Breakdown, best-by-GATE.
- `smoke_sft_de.py` — `<result>`-Block aus dem Loss MASKIERT (token-genau verifiziert) ->
  Modell lernt NICHT, Ergebnisse zu faelschen.

Phasen (jede gated, best-by-Gate statt val_loss — val_loss war hier nachweislich irrefuehrend):
- Phase 1 (call_only): Tool-Call + Stop. step_400: tool 100% · false_tool 0% · parse 97% · correct 68%.
- Phase 1.1 (enrichte Uebersetzungs-Traces, Sprache->Formel). step_500: correct 93%.
- Phase 2 (full, Result-Injektion -> finale Antwort).

PROMOTED: **`checkpoints/tool_sft_v12/sft_smoke_step_600.pt`**
  correct **94%** · parse **100%** · fake_result **0%** · false_tool **0%** · answer_match **85%**
  Buckets: percent 24/24 · word 21/21 · speed 10/10 · english 7/7 · time_unit 16/17 · simple 16/21 (76%)

Ehrliche Grenzen: in-distribution (trainierte Aufgabentypen, neue Zahlen; frei formulierte
Fragen ungetestet); `simple`-Bucket schwach wegen sqrt/`hoch 2` (= Operator-Mapping, nicht +-*/);
answer_match konservativ gemessen (deutsches Dezimalkomma "59,5" vs Executor "59.5" zaehlt als
Mismatch). Tool-Use fuegt KEIN Wissen hinzu — Wissensluecken bleiben (Annealing/Skalierung).

Naechste Session (NICHT vorgezogen): (1) Gate-Zahlenvergleich normalisieren (Komma/Punkt/Trailing-0),
(2) simple-Bucket in basic/advanced splitten + Fehlerfaelle extrahieren, (3) gezielte sqrt/power-Traces,
(4) dann Kalibrierung/R-Tuning (key-frei, Self-Labeling gegen Gold/MC/Executor).

## Update 2026-06-06 — 1B-Foundation gelaufen + SFT (Verhalten) + Reasoning-Slice

NEUESTER STAND. Geht ALLEN Abschnitten darunter vor (inkl. 2026-05-31). Die
1B-Policy/Preflight-Gates weiter unten sind erfuellt und damit historisch — der
Foundation-Run IST gelaufen.

Wo wir stehen:

- 1B-Foundation-Warmstart v3 GELAUFEN bis step 50000
  (`checkpoints/pretrain_1b_bilingual_de55_en45_foundation_warmstart_v3/step_50000.pt`).
  Gesundes Training, Sprache + Faktenbindung nachgewiesen (Wissensprofil n=57:
  Geschichte/Geografie stark, Wissenschaft/Uebersetzung schwaecher).

- SFT v1 (~32k diverse DE+EN, gpt-4o-verifiziert [269 Halluzinationen gefangen],
  dekontaminiert) GELAUFEN. Aus dem Base, der kaum antworten konnte, wurde ein
  ANTWORTENDER Assistent (Wien/Madrid korrekt, sauberes Stoppen via
  eos-loss-weight 2.0). SFT lehrt FORM, nicht WISSEN — durch Benchmarks bestaetigt.

- Benchmarks (eigener MC-Loglikelihood-Runner, n=300): Helix-SFT schlaegt auf
  mmlu_de SmolLM2-360M + TinyLlama-1.1B; Qwens MMLU-Vorsprung schrumpft von ~22
  (EN) auf ~7 (DE). Sprachstrategie (200k Vokab, de55/en45) zahlt sich messbar
  aus. Absolutwerte niedrig (Untertrainings-/Groessen-Signal). Details:
  `docs/PROJEKT_STAND.md`.

- Reasoning-Slice gebaut + verifiziert: 2500 DE (nativ generiert) + 2500 EN
  (GSM8K konvertiert). gpt-4o-Verify zu 100% auf Mathe -> ~9.4% falsche Mathe
  gefangen/korrigiert. Sauber im Helix-Format.

- SFT v2 LAEUFT (36.6k = SFT v1 + Reasoning-Slice, ~13.5% Reasoning, 1 Epoche,
  bucket+grad-ckpt). val-Tiefpunkt bei step 2100 (val 2.580 — besser als v1 ~2.81),
  danach Overfit-Uptick. Keeper: `checkpoints/sft_v2/sft_smoke_step_2100.pt`.
  Quicktest + Re-Benchmark als naechstes.

Entschiedene Richtung danach (dreifach trianguliert Michael+GPT+Claude):

1. Tool-Use ZUERST (Mathe-Tool-Harness): kleines Modell lernt PRUEFEN statt raten.
   Spec: `docs/BLUEPRINT_TOOL_USE_VERIFIER.md`.
2. Annealing (FineWeb-2-DE/Cosmopedia/Python-Edu schon geladen) inkl. Code.
3. DoRA Mathe/Logik/Code auf annealtem Base. Spec:
   `docs/BLUEPRINT_DOMAIN_ADAPTERS_DORA.md`.

Reihenfolge gegated (`ZUKUNFT_BACKLOG.md`). Kernprinzip: Adapter verstaerkt
Latentes, installiert nichts -> Code-DoRA gesperrt bis Code-Annealing.

Infra-Hinweis: `data/` und `checkpoints/` liegen NUR auf BITBASTION
(`/workspace/v2data`, 36T), nicht auf der Windows-Box (gitignored, zu gross).
Nur Code synct (U:\ <-> Container). Das ist gewollt, kein Daten-Verlust.

## Update 2026-05-31 — Edu-Daten-Filter (Deutsch) + Multi-GPU

Dieser Block ist der neueste Stand und geht den aelteren 1B-Canary-/500M-
Abschnitten unten vor.

Kontext: Der bilinguale 1B-Ramp (de55/en45) lief bis Step ~3400 (best.pt),
das Lernverhalten war enttaeuschend. Saubere Diagnose (nicht aus dem Bauch):

- NICHT die Eval (Qwen-2.5 auf denselben Probes = sinnvoll, 37/50).
- NICHT die Architektur (All-Plain-Attention-Kontrolle ~ gleichauf mit Helix
  bis Step 300).
- Sondern: Under-Training (~3.4B Tokens ~ 16% Chinchilla) UND ein
  qualitaets-invertierter deutscher Mix (die schwaechste Quelle bekam das
  meiste Budget).

Daten-Qualitaet (FineWeb-Edu-Methodik fuer Deutsch, neu gebaut):

- LLM-Annotation 0-5 auf Bildungswert. Judge: `qwen3-235b-a22b-2507` via
  OpenRouter (non-thinking, ~40x billiger als gemini-3.5-flash, strenger und
  genauer auf Web-Text). 12k Labels, ~1 EUR.
- Cheap Klassifikator: frozen multilingual-e5-large + Ridge-Kopf + kalibrierte
  Schwelle. Val Pearson 0.866, Keep-F1 0.872.
- Korpus-Filter @ Schwelle 2.0: fineweb2_de ~38% behalten, wikipedia_de ganz,
  german_commons GEDROPPT (~2-5% Keep, EuroParl/OCR-Fragmente).
- German-v2 = edu-gefiltertes fineweb2_de + wikipedia_de ~ 2.0B hochwertige
  Tokens (reicht den ~1.8B-DE-Bedarf des Foundation-Runs ohne Wiederholung).
- Config: `configs/data_paths.curated_v2_german.yaml` (re-tokenisiert nur DE).

Multi-GPU / DDP (neu, PR #1, Branch `feat/multigpu-ddp`):

- DistributedDataParallel im Trainer, strikt auf `WORLD_SIZE>1` gegated ->
  Single-GPU-Pfad bit-identisch (verifiziert: py_compile + dry-run).
- DDP-agnostische Checkpoints (kein `module.`-Prefix -> single-GPU-ladbar),
  no_sync bei Grad-Accum, Rank-0-Eval+Barrier, globaler Stop via all_reduce.
- torchrun-Launcher: `scripts/ops/run_pretrain_multigpu.sh`.
- Gemessener Durchsatz: 12.9k tok/s/GPU (1B, Blackwell). Volles 1B (~20B Tok):
  ~18 Tage 1 GPU, ~5 Tage 4 GPU. Noch nicht auf echter Multi-GPU validiert
  (Testbox hat 1 GPU) -> kurzer 2-GPU-Run auf RunPod vor langer Strecke.

Infra-Entscheidung: Training bleibt auf BITBASTION (1 GPU, gratis) fuer den
Foundation-Run; fuer schnelle/grosse Laeufe RunPod-Multi-GPU (Spot, dank
Resume), NICHT Colab (Compute-Units + Session-Limits ungeeignet).

Skalierungs-Quellen (wenn mehr Deutsch noetig): RedPajama-V2-de (3T modern,
mit Quality-Signals) + mehr fineweb2_de, edu-gefiltert. german-commons
verworfen (OCR-historisch, siehe L-020). multitask_german_32k fuer die
spaetere SFT-Phase gesichert.

Offen / als naechstes:

1. fineweb2_de-Voll-Scoring laeuft (~38% Keep) -> dann German-v2 tokenisieren
   (altes `german.bin` sichern, nur DE neu via `curated_v2_german.yaml`).
2. Danach Foundation-Warmstart von ramp `best.pt` auf den besseren Daten.

## Kurzentscheidung

1B wird noch nicht gestartet.

Der Sicherheitsrahmen fuer einen 1B-Canary ist jetzt gebaut, aber der Preflight
ist noch nicht gruen. Der naechste echte Schritt ist kein weiterer 500M-SFT-
Patch, sondern der finale, auditierte 1B-Clean/Tokenized-Mix.

Aktueller 1B-Preflight:

- Report: `reports/auralis_1b_readiness_preflight_v2_2026-05-29.md`
- Ergebnis: `ready_to_launch: False`
- Eval-Prompts: 70
- Trainings-Einheiten gescannt: 382,763
- Hash-Kollisionen: 0
- Substring-Hits: 0

Interpretation:

- Die Leak-/Disjunktheitsseite ist aktuell sauber.
- Der Start ist blockiert, weil der finale 1B-Datenmix noch nicht belastbar
  als Clean/Tokenized-Mix eingetragen und per Preflight freigegeben ist.
- `configs/data_paths_1b_samples_container.yaml` muss vor Start auf echte,
  existierende 1B-Clean- und Tokenized-Pfade zeigen. `.bin`-Tokenfiles brauchen
  die passende `.idx`.

## Verbindliche 1B-Policy

Der 1B-Lauf darf erst starten, wenn Daten und Gates vorher gruen sind.

Verbindliche Dateien:

- 1B Readiness Gate: `eval/auralis_1b_readiness_gate_v1.yaml`
- Frozen Target/Retention Gate: `eval/sft_response_frozen_target_retention_v2.yaml`
- 1B Preflight Config: `configs/eval/auralis_1b_readiness_preflight.yaml`
- 1B Preflight Script: `scripts/eval/one_b_readiness_preflight.py`
- Guarded Canary Config: `configs/training/pretrain_1b_canary_readiness.yaml`
- Guarded Canary Runner: `scripts/ops/run_pretrain_1b_canary_readiness.sh`

Promotion-Regel:

- Target muss bestehen.
- Retention muss 0 Regressionen haben.
- Eine einzige Retention-Regression bedeutet: nicht promotable.
- Eval-Probes werden nicht gelockert, um einen Lauf gruen zu machen.
- Neue Probes nur append-only ergaenzen.

Wichtige Target-/Retention-Achsen:

- Photosynthese als echtes Konzept, nicht nur Keyword-Treffer.
- Faust/Goethe als confident known fact.
- Bonn frueher vs. Berlin heute.
- Bekannte Fakten beantworten, erfundene Entitaeten verweigern.
- Goethe nicht mit `Mein Kampf` verwechseln.
- Faust I nicht wegen ueberdominantem Honesty-Training verweigern.

## 500M-Stand

Kein getesteter 500M-Checkpoint ist promotable.

Frozen-Gate-v2-Ergebnisse:

| Checkpoint | Target | Retention | Promotable |
|---|---:|---:|---:|
| `v8_safe` | 8/25 | 18/25 | nein |
| `hybrid_v1_40` | 9/25 | 17/25 | nein |
| `hybrid_v12_bridge_60` | 10/25 | 17/25 | nein |
| `hybrid_v12_repair_v2_80` | 9/25 | 17/25 | nein |

Aktuelle Schlussfolgerung:

- `v8_safe` bleibt nur relativ am stabilsten, weil Retention am wenigsten
  kaputtgeht.
- Hybrid/v12 bewegt Photosynthese/Faust teilweise, verliert aber Retention.
- Weitere 500M-Mini-Patches sind Diagnose-Arbeit, keine Promotion-Arbeit.
- 500M darf nicht als geloest oder produktionsnah behandelt werden.

## Diagnose

Die aktuellen Fehler sind keine simplen Prompt-, Score- oder Loss-Probleme.
Das Modell zeigt Interferenz:

- Photosynthese/Faust lassen sich lokal verbessern.
- Dabei kippen Bonn/Berlin, Known-Fact-Retention oder sichere Gegenfakten.
- Honesty/Refusal ist bei manchen bekannten Fakten zu dominant.
- Erfundenen Entitaeten werden teilweise trotzdem Details angedichtet.

Das spricht gegen weitere kleine Reparatur-SFTs auf 500M und fuer einen sauber
gewichteten 1B-Pretrain/SFT-Mix.

## Adaptive / Live-Gates

Die adaptive Trainingsschicht kann das v2-Frozen-Gate live mitlaufen lassen:

- Adaptive Frozen-Gate Bridge: `src/auralis/adaptive/frozen_gate.py`
- Adaptive Trainer CLI: `scripts/train/adaptive_curriculum.py`
- Live-Trace: `<output-dir>/frozen_gate_trace.jsonl`

Neue Live-Metriken:

- `frozen_target_pass`
- `frozen_retention_pass`
- `frozen_target_failures`
- `frozen_retention_failures`
- `frozen_promotable`

Teststatus laut Handoff:

- lokale adaptive Tests: gruen
- Container-Smoke fuer Frozen-Gate-Live-Bridge: gruen
- Mini-Smoke mit 500M-v8 pruefte nur Verdrahtung, nicht inhaltliche Scores.

## Naechste Schritte

1. Finalen 1B-Clean-Mix bauen.
   - `cleaned.german`, `cleaned.english`, `cleaned.code` mit echten Pfaden
     befuellen.
   - Tokenized-Pfade eintragen.
   - Existenz von `.bin` und passender `.idx` sicherstellen.

2. Datenmix vor dem Tokenisieren auditen.
   - moderne deutsche QA/Fakten staerker als beim 500M.
   - Refusal/Honesty deckeln, nicht dominieren lassen.
   - confident-correct Fakten explizit einplanen.
   - Books niedrig dosieren, nicht als Hauptwissenstraeger verwenden.

3. Preflight erneut laufen lassen:

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

Sofort stoppen oder nicht promoten, wenn:

- Retention auch nur einen Fehler bekommt.
- Bonn historisch wegfaellt.
- Berlin heute kippt.
- Goethe/Faust verweigert oder falsch wird.
- Photosynthese nur gut klingt, aber Zucker/Sauerstoff/Pflanzen/Licht falsch
  verbindet.
- erfundene Entitaeten ausgeschmueckt werden.
- Target nicht besser wird, obwohl Loss sinkt.

## Aktuelle Referenzen

- `reports/auralis_1b_readiness_plan_2026-05-29.md`
- `reports/codex_handoff_1b_readiness_v2_2026-05-29.md`
- `reports/auralis_1b_readiness_preflight_v2_2026-05-29.md`
- `reports/learning_neuro_hybrid_v12_2026-05-29.md`
- `docs/DOCS_INDEX.md`
- `eval/README.md`

## Was nicht mehr als aktueller Stand gilt

- `STATUS.md` Stand 2026-05-17 als Run-Plan.
- Der alte `pretrain_mix_v4_boosted_500m` als aktueller Hauptpfad.
- Der April-Plan `curated_40b` als aktiver Hauptmix.
- Alte Pfade wie `tokenized/phase1` oder `checkpoints/phase1_pretrain` als
  Default fuer neue Runs.
- SFT als Reparatur fuer ein schwaches/noisy Base-Modell.
