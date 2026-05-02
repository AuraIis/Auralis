# STATUS - Auralis v2

Stand: 2026-04-26

Aktive Phase: Phase 1 ist code-seitig fertig, review-validiert und per canary + 1B sweep abgesichert.
Modellgroesse: 1B final (`helix_v2_1b.yaml`, ~954M Params).
Phase-1-Token-Budget: 25B geplant, ~21B aktuell bereitgestellt (84% Deckung; Rest kann in Phase 2 geschlossen werden).

## Kurzstatus

- Tokenizer fertig und byte-exakt validiert.
- Modell-Architektur fertig.
- Pretraining-Pipeline fertig.
- Blackwell-GPU-Validation PASS.
- Canary-Runden abgeschlossen.
- Code-Review-Findings vom 2026-04-26 komplett behoben.
- Server-Status: **105/105 Tests gruen**.
- 1B batch-size sweep abgeschlossen.
- Empfohlene Hauptlauf-Config: **seq=2048, batch=4, gradient_checkpointing=on**.
- **🚀 1B Phase-1 Hauptlauf läuft seit 2026-04-26 ~18:18 lokal** (PID 225, ETA ~12-19 Tage). Logs: `logs/phase1_pretrain.log`. Mix 70/25/5, --no-wandb, Token-Reads von NVMe-cache, Checkpoints auf disk6.

## Phase 0 - Tokenizer

Artefakte in `tokenizer/`:
- `helix_v2_tokenizer.model`
- `helix_v2_tokenizer.vocab`
- `training_manifest.yaml`
- `quality_report.md`

Qualitaetsprofil:

| Sprache | Tokens/100 Woerter | Tokens/KB | Unknown-Rate | Ziel |
|---|--:|--:|--:|---|
| EN | 123.0 | 203.4 | 0% | <=135 |
| DE | 133.8 | 188.7 | 0% | <=150 |
| Code | 272.2 | 313.6 | 0% | <=350 tok/KB |

Chat-template roundtrip: byte-exakt PASS.

## Phase-1 Datenlage

Root: `//BITBASTION/Auralis/AuralisV2/`

| Datei | Groesse | Tokens est. | Quelle |
|---|--:|--:|---|
| `cleaned/german.txt` | 23.70 GB | ~4.7B | v1 reuse |
| `raw/english/fineweb_edu.txt` | 40.00 GB | ~10.0B | FineWeb-Edu |
| `raw/english/wikipedia_en.txt` | 12.00 GB | ~3.0B | Wikipedia EN |
| `raw/english/openmath.txt` | 8.00 GB | ~2.0B | OpenMathInstruct-2 |
| `raw/code/starcoderdata.txt` | 3.50 GB | ~1.0B | StarCoderData |
| `raw/code/open_web_math.txt` | 0.88 GB | ~0.25B | open-web-math |
| **Total** | **88.08 GB** | **~21B** | |

Nicht eingeflossen:
- SlimPajama
- Dolma
- Proof-Pile-2

Tokenizer-Korpus: `tokenizer_corpus/corpus_clean.txt` mit 15.5 GB im Mix 50/40/10 EN/DE/Code.

## Phase 0.5 - Modell

Implementiert in `src/auralis/model/`:
- Config + layer stack
- RMSNorm
- SwiGLU FFN
- Mamba-2 Referenz
- GLA Referenz
- Sparse Attention
- RoPE
- Scaled-normal init
- KV cache dataclass
- `HelixModel` + `build_model()`

Model configs:
- `configs/model/helix_v2_100m.yaml`
- `configs/model/helix_v2_100m_ref.yaml`
- `configs/model/helix_v2_250m.yaml`
- `configs/model/helix_v2_mid_500m.yaml`
- `configs/model/helix_v2_1b.yaml`

Frueher CPU-Referenzstand:
- 100M forward-loss bei frischer Initialisierung: ~12.37, nahe `ln(200000)=12.20`
- Keine NaN/Inf in Logits oder Gradienten

## Phase 1 - Pretraining-Pipeline

Wichtige Scripts:
- [scripts/data/tokenize_for_pretraining.py](/BITBASTION/Auralis/AuralisV2/scripts/data/tokenize_for_pretraining.py)
- [scripts/pretrain/train_phase1.py](/BITBASTION/Auralis/AuralisV2/scripts/pretrain/train_phase1.py)
- [scripts/pretrain/smoke_test.py](/BITBASTION/Auralis/AuralisV2/scripts/pretrain/smoke_test.py)
- [scripts/utils/batch_size_sweep.py](/BITBASTION/Auralis/AuralisV2/scripts/utils/batch_size_sweep.py)

Wichtige Training-Module:
- [dataset.py](/BITBASTION/Auralis/AuralisV2/src/auralis/training/dataset.py)
- [optimizer.py](/BITBASTION/Auralis/AuralisV2/src/auralis/training/optimizer.py)
- [trainer.py](/BITBASTION/Auralis/AuralisV2/src/auralis/training/trainer.py)
- [utils.py](/BITBASTION/Auralis/AuralisV2/src/auralis/training/utils.py)

Validation:
- CPU smoke test PASS
- Blackwell validation PASS
- Canary Runde 2 und Runde 3 abgeschlossen
- Review-Fixes regressionsicher abgesichert
- Server: **105/105 Tests gruen**

Neue Regressionen decken jetzt ab:
- emergency checkpoint rotation
- plain-attention RoPE build
- sampler last-valid-window
- gradient-checkpointing override
- explizites disable im sweep

## Blackwell Status

GPU: RTX PRO 5000 Blackwell, 47 GB VRAM.

Validiert:
- native backend stabil
- kernel swap numerisch korrekt
- `TRITON_OVERRIDE_ARCH=sm89` als funktionierender Workaround

Fruehere Messpunkte mit allen Kernels aktiv:
- seq=256, batch=4 -> 220 tok/s, 6.68 GB
- seq=512, batch=8 -> 1928 tok/s, 16.36 GB
- seq=1024, batch=4 -> 3628 tok/s, 16.84 GB
- seq=2048, batch=2 -> 2713 tok/s, 17.74 GB

## Canary und 1B Sweep

Canary:
- Baseline-Mix bleibt overall der beste Kandidat fuer den 1B-Hauptlauf.
- DE-heavy verbessert DE etwas, verliert aber overall.
- `de_medium_b16` brachte keinen zusaetzlichen Gewinn gegenueber DE-heavy.

1B batch sweep:
- seq=1024: batch bis 12 OK
- seq=2048: batch bis 8 OK, batch 12 OOM
- top throughput: **seq=2048, batch=4 -> ~11.3k tok/s bei 23.3 GB peak**
- batch=8 liefert nur kleinen tok/s-Gewinn bei stark hoeherem VRAM-Verbrauch

Empfehlung fuer den 1B Phase-1-Hauptlauf:
- `seq=2048`
- `batch=4`
- `gradient_checkpointing=on`

Siehe auch:
- [HISTORY.md](/BITBASTION/Auralis/AuralisV2/HISTORY.md)
- [docs/PHASE_1_LAUNCH.md](/BITBASTION/Auralis/AuralisV2/docs/PHASE_1_LAUNCH.md)

## Offene Blocker vor Hauptlauf

1. RunPod- oder Zielhost-Setup final ausfuehren.
2. Go/No-Go fuer den 1B-Phase-1-Hauptlauf treffen.
3. Zielhardware final festlegen: `1xH200`, `4xA40` oder lokaler Blackwell-Run.

## Naechster Schritt nach Phase 1

Phase 2:
- bilingual continued pretraining
- KL distillation
- Teacher Phase-1-Checkpoint einfrieren
- Student auf 60/30/10 DE/EN/Code weitertrainieren

Siehe:
- [SPEC_PHASE_2_CONTINUED_BILINGUAL.md](/BITBASTION/Auralis/AuralisV2/Doc/SPECs/SPEC_PHASE_2_CONTINUED_BILINGUAL.md)

## Offene Entscheidungen

- Multi-GPU Setup fuer Phase 1
- Phase-2-Daten-Ergaenzung fuer die fehlenden EN-Tokens
- Open-weights vs. proprietaerer Release

## Technische Schulden

Aus dem Review-Pass vom 2026-04-26 sind aktuell keine offenen Findings mehr uebrig.
Die geschlossenen P1/P2/P3-Punkte sind in [HISTORY.md](/BITBASTION/Auralis/AuralisV2/HISTORY.md) dokumentiert.

## Daten-Qualitaet — Phase-2-Vorbereitung

Stichproben-Audit der aktiven Trainings-Daten (`tokenized/curated_40b/`, 2026-04-26):

- **EN: gruen.** Mix sauber (fineweb_edu 53%, dolma 20%, wikipedia 17%, openmath 10%). Web-/Wissens-/Tutorial-Text, kein Alarmierendes.
- **DE: gruen.** Mix sauber (german_commons 46%, fineweb2_de 39%, wikipedia 15%). Sauberes allgemeines Deutsch.
- **Code: gelb.** Effektiv StarCoder + OpenWebMath als Fallback. Mittlere Doc-Laenge 85 Bytes/Zeile (vs EN/DE 2.4 KB) deutet auf snippets/fragments statt File-Korpus.

Zwei silent-Drops im aktiven Snapshot (siehe `training/curated_40b/mix_manifest.json`):

- **`fineweb2_en` = 0 Bytes** — sollte Teil des EN-Mixes sein, hat aber 0 beigetragen.
- **`the_stack_v2` = 0 Bytes** — der primaere Code-Korpus, komplett gedroppt → Code-Schwaeche.

Fuer **Phase 1** kein Blocker (1B-Hauptlauf laeuft mit gegebener Datenbasis, Code wird "leicht familiarisiert"). Fuer **Phase 2 (60/30/10 DE/EN/Code, doppelter Code-Anteil)** muss vorher geklaert werden:

1. **Root-cause-Analyse warum `fineweb2_en` und `the_stack_v2` auf 0 gelandet sind.** Vermutlich filter_quality.py-Threshold oder download/tokenize-Pipeline-Issue.
2. **Code-Korpus haerten:** mehr echte Multi-Line-Dateien, weniger Kommentare/Strings/Docs-Fragmente.
3. **Optional Code-Filter:** extrem kurze oder offensichtlich nicht-codeartige Zeilen rausfiltern.

Bewertung passt zu Smoke-Eval-Loss-Trajektorien (Code-Eval-Loss konvergiert deutlich schlechter als EN/DE).

## Phase-2-Backlog (Architektur + Optimizer)

### Muon Optimizer Evaluation für Phase 2 (TODO 2026-04-29)

**Hintergrund:** DeepSeek V4 (Apr 2026) nutzt Muon-Optimizer fuer 25-40% Wallclock-Speedup vs AdamW. Architektur-orthogonal — kein Modell-Aenderung notwendig.

**Was ist Muon:** "MomentUm Orthogonalized by Newton-schulz" (Keller Jordan et al., 2024). Standard-Pattern: Muon fuer 2D-Matrix-Parameter (Linear, Attention), AdamW als Fallback fuer 1D-Parameter (biases, norms, embeddings). Newton-Schulz orthogonalisiert die Momentum-Update-Matrix → alle Update-Richtungen werden gleichgewichtet, schnellere Konvergenz.

**Empfohlener Adoption-Slot:** Phase 2 (Continued Pretraining) — Phase 2 startet mit frischem Optimizer-State von Phase-1-best.pt, also kein State-Inkompatibilitaets-Issue. Mid-Phase-1-switch ist NICHT empfohlen.

**Action-Items:**
1. Vor Phase-2-Start: kleine Ablation (250M-Canary, ~5k steps Muon vs AdamW) um Konvergenz auf der Helix-Hybrid-Architektur (Mamba+GLA+Sparse) zu verifizieren — die meisten Muon-Validierungen sind auf Standard-Transformern, nicht Hybrid-Stacks.
2. Falls Ablation positiv: Muon in `scripts/pretrain/train_phase1.py` als Optional-Backend hinter env-flag (`AURALIS_USE_MUON=1`) einbauen.
3. Hyperparameter-Tuning: Muon LR ~5-10x hoeher als AdamW-LR (z.B. 0.02 statt 3e-4 fuer Matrix-Params, AdamW-Default fuer Rest).
4. Implementation-Referenz: [github.com/KellerJordan/Muon](https://github.com/KellerJordan/Muon).
5. Memory-Vorteil: Muon braucht nur `m` (momentum), kein `v` (variance) — ~33% weniger Optimizer-State pro Matrix-Parameter. Bei 1B-Modell relevant.

**Erwarteter Gain bei Phase 2:** falls Ablation +30% Wallclock confirms, sparen wir bei Phase-2 (10-15B Tokens auf RTX PRO 5000) ca. 2-4 Tage.

### CSA / HCA / mHC — Architektur-Innovationen aus DeepSeek V4 (Backlog v3)

**Nicht fuer Auralis v2.** Architektur-Wechsel mid-version waere Run-zerstoerend.

- **CSA (Compressed Sparse Attention):** KV-Cache-Kompression entlang Sequenz-Dimension + DeepSeek Sparse Attention. Macht lange Kontexte (>32k) effizient.
- **HCA (Heavily Compressed Attention):** aggressivere KV-Cache-Kompression als CSA, fuer extrem lange Kontexte (>128k).
- **mHC (Manifold-Constrained Hyper-Connections):** Replacement fuer klassische residual connections.

**Auralis-spezifischer Reality-Check:** Auralis v2 hat bereits 22 von 28 Layern in linear-time Formaten (6 Mamba + 16 GLA), die O(1)-Speicher pro Token haben. CSA/HCA loesen das Problem von Standard-Transformern, Auralis loest es schon ueber andere Theorie. CSA/HCA waeren nur relevant wenn:
- echte 1M+ Token-Kontexte Use-Case-Priority werden
- Skalierung > 10B Params (dann sind die 6 Sparse-Attention-Layer teurer)

**Action fuer v3-Planning:** evaluieren ob v3 Architektur-Refresh komplett-V4-style (CSA/HCA/mHC) machen sollte ODER bei Hybrid-Stack mit eigenem Long-Context-Approach bleiben. Entscheidung NACH Phase-1/2/3 Auralis-v2 Abschluss, basierend auf konkreten Long-Context-Anforderungen.
