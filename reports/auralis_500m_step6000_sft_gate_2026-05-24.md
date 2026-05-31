# Auralis 500M v5 - Step-6000/SFT Gate Report

Stand: 2026-05-24, ca. 14:30 Europe/Berlin

Dieser Bericht ist als Übergabe für einen zweiten Codex-Agenten gedacht. Er beschreibt Infrastruktur, echte Checkpoint-Stände, bisherige Tests, SFT-Experimente und die wichtigsten Risiken. RunPod-Pretraining läuft weiter; die lokalen Tests wurden getrennt auf BITBASTION im neuen Blackwell-Docker gemacht.

## Kurzfazit

- Der RunPod-A100-Pretrain ist technisch stabil und läuft weiter.
- Der beste echte getestete Pretrain-Checkpoint ist Step 6000 mit `val_loss=1.5821`.
- Es gab einen wichtigen Import-/Benennungsfehler: Der alte lokale Ordner `pretrain_mix_v5_boosted_500m_a100_latest` enthielt in `best.pt` noch Payload-State `step=5000`, obwohl Dateizeit/Ordnername nach Step 6000 aussah.
- Der echte Step-6000-Checkpoint wurde danach neu importiert und verifiziert.
- SFT funktioniert grundsätzlich: Val-Loss sinkt stark und der Backbone reagiert.
- Kurzer, sauberer Deutsch-SFT lernt sogar `<|end|>`/Stop-Verhalten.
- Code-SFT ist aktuell gefährlich: Loss sinkt, aber allgemeine Antworten werden mit Code-Mustern kontaminiert und Code bleibt qualitativ schwach.
- Empfehlung: RunPod weiter bis mindestens Step 10000/12000 laufen lassen; echten neuen `best.pt` dann erneut importieren und Gate wiederholen. Kein großer SFT/LoRA auf Step 6000 als Produktionsbasis.

## Infrastruktur

### BITBASTION

- Host: `BITBASTION`
- SSH: `ssh root@BITBASTION`
- Aktiver Trainingsdaten-/Codepfad für Docker-Mounts:
  `/mnt/user/Auralis/NEWGPT/v2data`
- Windows-SMB/Arbeitsrepo, wo die neuen Docker-Skripte liegen:
  `\\BITBASTION\Auralis\AuralisV2`
- Docker-Container für lokale Blackwell-Tests:
  `auralis-blackwell`
- Docker-Image:
  `auralis-blackwell:cu13`
- GPU:
  `NVIDIA RTX PRO 5000 Blackwell`, Compute Capability `(12, 0)`

### Neuer Blackwell-Docker

Der alte `auralis-training` Stack hatte alte Triton/CUDA-Probleme mit Blackwell. Deshalb wurde ein separater Docker gebaut:

- CUDA 13.0
- `torch 2.12.0+cu130`
- `triton 3.7.0`
- `mamba-ssm 2.3.2.post1`
- `causal-conv1d 1.6.2.post1`
- `flash-linear-attention 0.5.0`
- `flash-attn 2.8.3`

Smoke-Tests bestanden:

- Auralis-Modell lädt auf Blackwell.
- `flash_attn_func` läuft direkt auf CUDA.
- Mamba/causal-conv/FLA/flash-attn importieren sauber.
- Checkpoint-Forward ist finite.

Wichtige Skripte im SMB-Repo:

- `docker/blackwell/Dockerfile`
- `docker/blackwell/entrypoint.sh`
- `scripts/ops/build_auralis_blackwell_image.sh`
- `scripts/ops/run_auralis_blackwell_container.sh`
- `scripts/ops/smoke_auralis_blackwell_container.sh`

### RunPod

Zugriff über BITBASTION:

```bash
ssh root@BITBASTION
ssh -i /root/.ssh/bitbastion-runpod -p 13344 root@157.157.221.29
```

RunPod-Projekt:

```bash
/workspace/v2data
```

Run:

- Config: `/workspace/v2data/configs/training/pretrain_mix_v5_boosted_500m_a100.yaml`
- Runner: `/workspace/v2data/scripts/ops/run_pretrain_mix_v5_boosted_500m_a100.sh`
- Log: `/workspace/v2data/logs/pretrain_mix_v5_boosted_500m_a100.log`
- PID: `/workspace/v2data/logs/pretrain_mix_v5_boosted_500m_a100.pid`
- Checkpoints: `/workspace/v2data/checkpoints/pretrain_mix_v5_boosted_500m_a100`

Letzter geprüfter RunPod-Status:

- Prozess läuft.
- GPU: A100-SXM4-80GB, ca. 99% Utilization.
- VRAM: ca. `50.8/81.9 GB`.
- Temperatur: ca. `51C`.
- Durchsatz: ca. `28.7k tok/s`.
- Step 6500 eval: `val_loss=1.620`, `german=1.648`.
- Step 6500 war schlechter als Step 6000, daher bleibt Step 6000 aktuell best.
- Keine OOM/NaN/Tracebacks gesehen.

## Checkpoint-Wahrheit

### Alter lokaler Import - irreführend

Alter lokaler Pfad:

```bash
/workspace/v2data/checkpoints/runpod_import/pretrain_mix_v5_boosted_500m_a100_latest/best.pt
```

Payload:

```json
{
  "step": 5000,
  "best_val_loss": 1.6267899894714355,
  "tokens_seen": 2621440000
}
```

Dieser Checkpoint wurde zuerst versehentlich als "Step 6000 latest" interpretiert. Das war falsch. Ergebnisse daraus sind als Step-5000-basierte Ergebnisse zu behandeln.

### Echte Step-6000-Importe

Neu importierter lokaler Pfad:

```bash
/workspace/v2data/checkpoints/runpod_import/pretrain_mix_v5_boosted_500m_a100_step6000/best.pt
/workspace/v2data/checkpoints/runpod_import/pretrain_mix_v5_boosted_500m_a100_step6000/step_6000.pt
/workspace/v2data/checkpoints/runpod_import/pretrain_mix_v5_boosted_500m_a100_step6000/MANIFEST.yaml
```

Payload beider `.pt`-Dateien:

```json
{
  "step": 6000,
  "best_val_loss": 1.582100075483322,
  "tokens_seen": 3145728000,
  "consecutive_val_increases": 0
}
```

## Capability-Gate Ergebnisse

Probe-Datei:

```bash
eval/capability_probes_v4_v5_gate.yaml
```

Modellconfig:

```bash
configs/model/helix_v2_mid_500m_smart.yaml
```

Tokenizer:

```bash
tokenizer/helix_v2_tokenizer.model
```

| Checkpoint / Variante | Basis | Aggregate | Wichtigste Kategorien | Ergebnisdatei |
|---|---:|---:|---|---|
| Base alter Import | Step 5000 | 21.8% | cleanliness 35, longform 50, math 35, facts 0, code 0 | `data/eval/checkpoint_tests/blackwell_latest_import/pretrain_mix_v5_boosted_500m_a100_latest_blackwell.json` |
| Base echter Import | Step 6000 | 28.2% | facts 70, math 50, cleanliness 35, code 0 | `data/eval/checkpoint_tests/pretrain_mix_v5_boosted_500m_a100_step6000_blackwell/pretrain_mix_v5_boosted_500m_a100_step6000_blackwell.json` |
| Clean-SFT 50 Steps, alter Import | Step 5000 | 31.8% | facts 70, longform 70, qa 70, code 0 | `data/eval/checkpoint_tests/sft_smoke_de_step6000_v1/sft_smoke_de_step6000_v1_step50_blackwell.json` |
| Clean-SFT 50 Steps, echter Step 6000 | Step 6000 | 34.5% | cleanliness 70, facts 70, math 50, code 0 | `data/eval/checkpoint_tests/sft_smoke_de_step6000_true_v1/sft_smoke_de_step6000_true_v1_step50_blackwell.json` |
| Kurzantwort-SFT 200 Steps, echter Step 6000 | Step 6000 | 34.5% | cleanliness 70, facts 70, math 50, code 0 | `data/eval/checkpoint_tests/sft_short_answer_step6000_true_v1/sft_short_answer_step6000_true_v1_step200_blackwell.json` |
| Code-SFT 200 Steps, echter Step 6000 | Step 6000 | 34.5% | code 35, facts 70, math 50; manuell schlecht | `data/eval/checkpoint_tests/sft_code_step6000_true_v1/sft_code_step6000_true_v1_step200_blackwell.json` |

Wichtig: Der Gate-Score ist nur ein grober Indikator. Manuelle Outputs sind entscheidend, weil Repetition und Kontamination teilweise nicht ausreichend bestraft werden.

## SFT Experimente

Alle SFT-Tests liefen lokal auf BITBASTION im `auralis-blackwell` Container. RunPod wurde nicht gestoppt oder verändert.

### 1. Clean-SFT Smoke, echter Step 6000

Input:

- Checkpoint: `checkpoints/runpod_import/pretrain_mix_v5_boosted_500m_a100_step6000/best.pt`
- Train: `data/training/sft_clean_de_v1/train.helix.jsonl`
- Val: `data/training/sft_clean_de_v1/val.helix.jsonl`
- Schritte: 50
- LR: `1e-5`
- Batch: 1
- Grad accum: 8

Output:

```bash
checkpoints/sft_smoke_de_step6000_true_v1/sft_smoke_step_50.pt
logs/sft_smoke_de_step6000_true_v1.log
```

Loss:

```text
initial val_loss=4.7658
step 50 val_loss=3.5313
```

Beobachtung:

- Lernt schnell Richtung Deutsch/Instruktionsformat.
- Antworten werden verständlicher.
- Repetition bleibt stark.
- Code bleibt unbrauchbar.

### 2. Kurzantwort-SFT, echter Step 6000

Ziel: Testen, ob Stop-Token/kurze Antworten lernbar sind.

Datensatz wurde aus `sft_clean_de_v1` gefiltert:

- Kategorien u.a. `factual_qa`, `rewrite`, `translation`, `concept_explain`, `technical_explanation`, `format_following`, `honest_refusal`, `factual_correction`, `math_reasoning`, `code_explain`.
- Antwortlänge grob <= 90 Wörter.
- Keine starke 3-Gramm-Repetition.
- Train: 2048 Beispiele.
- Val: 47 Beispiele.

Pfad:

```bash
data/eval/checkpoint_tests/sft_short_answer_step6000_true_v1/train.helix.jsonl
data/eval/checkpoint_tests/sft_short_answer_step6000_true_v1/val.helix.jsonl
data/eval/checkpoint_tests/sft_short_answer_step6000_true_v1/filter_stats.json
```

Training:

- Schritte: 200
- LR: `8e-6`
- Max length: 768
- Grad accum: 8

Output:

```bash
checkpoints/sft_short_answer_step6000_true_v1/sft_smoke_step_200.pt
logs/sft_short_answer_step6000_true_v1.log
```

Loss:

```text
initial val_loss=4.2560
step 25  val_loss=2.9501
step 50  val_loss=2.4360
step 100 val_loss=2.1121
step 150 val_loss=1.9488
step 200 val_loss=1.9037
```

Manuelle Outputs nach SFT:

```text
PROMPT: Was ist die Hauptstadt von Deutschland?
Die Hauptstadt von Deutschland ist Berlin.
<|end|>

PROMPT: Erkläre kurz, was Wasser ist.
Das Wasser ist ein wichtiger Bestandteil der menschlichen Ernährung. Es ist wichtig für die Gesundheit und das Wohlbefinden.
<|end|>
```

Bewertung:

- Starkes positives Signal: `<|end|>` wird gelernt, Antworten stoppen.
- Repetition sinkt für Chat-Prompts deutlich.
- Fakten sind teilweise falsch/oberflächlich.
- Mathe und Code weiterhin schwach.
- Dies ist die beste lokale Diagnose-Variante bisher.

### 3. Code-SFT, echter Step 6000

Ziel: Prüfen, ob Code-Fähigkeit schnell trainierbar ist oder Backbone noch zu schwach ist.

Gefiltert aus `sft_clean_de_v1`:

- Kategorien mit `code`/`coding`.
- Train: 1536 Beispiele.
- Val: 31 Beispiele.

Pfad:

```bash
data/eval/checkpoint_tests/sft_code_step6000_true_v1/train.helix.jsonl
data/eval/checkpoint_tests/sft_code_step6000_true_v1/val.helix.jsonl
data/eval/checkpoint_tests/sft_code_step6000_true_v1/filter_stats.json
```

Training:

- Schritte: 200
- LR: `8e-6`
- Max length: 1024

Output:

```bash
checkpoints/sft_code_step6000_true_v1/sft_smoke_step_200.pt
logs/sft_code_step6000_true_v1.log
```

Loss:

```text
initial val_loss=5.8445
step 50  val_loss=2.8452
step 100 val_loss=2.2262
step 150 val_loss=2.1001
step 200 val_loss=2.0430
```

Manuelle Outputs:

- Allgemeine Fragen wurden mit Code-Mustern kontaminiert.
- Python-Funktion blieb invalid/repetitiv.
- Formaler Gate-Code-Score stieg auf 35%, aber manuell ist die Variante nicht brauchbar.

Bewertung:

- Code-Spezialisierung auf diesem frühen Backbone ist noch nicht ratsam.
- Nicht als Produktionsweg verwenden.
- Erst nach mehr Pretraining und mit besseren disjunkten Code-Evals erneut testen.

## Decode-Matrix / Repetition

Ergebnisdateien:

```bash
data/eval/checkpoint_tests/deep_decode_step6000/decode_matrix.json
data/eval/checkpoint_tests/sft_short_answer_step6000_true_v1/decode_short_matrix.json
```

Wichtige Befunde:

### Base Step 6000

- Greedy-Decoding hat hohe Repetition.
- No-repeat reduziert Schleifen, aber Antworten werden oft sachlich falsch.
- Chat-Template wird teilweise nur wiederholt oder falsch verstanden.
- Plain-Prompts sind instabil.

Beispiele:

```text
Base Step 6000, Hauptstadt:
teilweise Berlin/Deutschland-Muster, aber schnell Drift/Repetition.

Base Step 6000, Python:
keine gültige Funktion.
```

### Clean-SFT Step 6000 50 Steps

- Stärker deutsch, mehr Antwortformat.
- Greedy-Repetition steigt teils sogar.
- Stop-Token noch nicht zuverlässig.

### Kurzantwort-SFT Step 6000 200 Steps

- Chat-Prompts stoppen oft korrekt mit `<|end|>`.
- Repetition bei Chat-Greedy deutlich besser.
- Plain-Prompts bleiben schlecht.
- Mathe/Code/Hallucination Guard bleiben nicht belastbar.

Beispiele:

```text
capital chat greedy:
Die Hauptstadt von Deutschland ist Berlin. ... <|end|>

water chat greedy:
Wasser ist ein Gas ... <|end|>
```

Das erste Beispiel ist formattechnisch gut; das zweite zeigt, dass sachliche Korrektheit noch fehlt.

## Was daraus folgt

### Gute Signale

- 500M v5 lernt besser als frühere rohe 100M/500M-Tests.
- Step 6000 ist objektiv besser als Step 5000 im Gate.
- SFT greift sofort: Loss fällt und Antwortform verbessert sich.
- Stop-/Repetition-Verhalten ist durch zielgerichtetes SFT trainierbar.
- Blackwell-Inferenzstack ist jetzt zuverlässig nutzbar.

### Warnsignale

- Der Backbone ist bei Step 6000 noch unreif.
- Hallucination Guard bleibt 0%.
- Code bleibt auch nach Code-SFT schwach.
- Spezial-SFT kann allgemeines Verhalten schnell kontaminieren.
- Scores allein sind gefährlich: Code-SFT sieht im Gate besser aus, ist manuell aber klar schlecht.
- Plain-Prompts ohne Chat-Template sind sehr instabil; SFT sollte konsequent auf ein einheitliches Chat-Format trainieren.

### Vergleich zur 1B-Fehlergefahr

Der wichtigste Schutz gegen den alten Fehler:

1. Checkpoint-Payload prüfen, nicht nur Dateiname/mtime.
2. Val-Loss plus manuelle Outputs plus Gate-Score betrachten.
3. Spezial-SFT erst nach General-SFT und disjunkten Evals.
4. Kein großer SFT, wenn Stop/End-Token nicht zuverlässig funktioniert.
5. Keine Entscheidung nur auf einem besseren Aggregate-Score.

## Empfehlung

### Jetzt

- RunPod-Pretraining weiterlaufen lassen.
- Step 6000 bleibt aktuell bester lokal verifizierter Pretrain-Checkpoint.
- Step 6500 hat schlechteren val_loss (`1.620`), daher nicht importieren/testen, außer zur Regression-Analyse.

### Nächster Gate-Zeitpunkt

Bei Step 7000/7500 oder beim nächsten `best.pt`:

1. Checkpoint von RunPod importieren.
2. Payload `state.step` und `best_val_loss` verifizieren.
3. Base Capability-Gate laufen lassen.
4. Kurzantwort-SFT 200 Steps wiederholen.
5. Decode-Matrix mit Chat-Prompts prüfen.
6. Nur wenn Hallucination/Code nicht regressieren: längeren SFT planen.

### Kein großer SFT jetzt

Nicht empfohlen auf Step 6000:

- großer all-purpose SFT
- Code-SFT/Code-LoRA
- Topic-LoRA
- DPO/RLHF

Empfohlen nur als Diagnose:

- kurzer Kurzantwort-SFT
- Stop-Token-/Chat-Template-Gate
- kleine disjunkte Probe-Sets

## Wichtige Befehle

### RunPod Status prüfen

```bash
ssh root@BITBASTION
ssh -i /root/.ssh/bitbastion-runpod -p 13344 root@157.157.221.29
cd /workspace/v2data
tail -80 logs/pretrain_mix_v5_boosted_500m_a100.log
nvidia-smi
```

### Checkpoint-Payload prüfen

```bash
docker exec auralis-blackwell bash -lc 'cd /workspace/v2data && python - <<PY
import torch
p="checkpoints/runpod_import/pretrain_mix_v5_boosted_500m_a100_step6000/best.pt"
payload=torch.load(p,map_location="cpu",weights_only=False)
print(payload.get("state"))
PY'
```

### Base Gate wiederholen

```bash
docker exec auralis-blackwell bash -lc 'cd /workspace/v2data && \
AURALIS_USE_MAMBA_KERNEL=1 AURALIS_USE_GLA_KERNEL=1 \
python scripts/eval/run_capability_probes.py \
  --probes eval/capability_probes_v4_v5_gate.yaml \
  --results-dir data/eval/checkpoint_tests/pretrain_mix_v5_boosted_500m_a100_step6000_blackwell \
  --tag pretrain_mix_v5_boosted_500m_a100_step6000_blackwell \
  --model-config configs/model/helix_v2_mid_500m_smart.yaml \
  --checkpoint checkpoints/runpod_import/pretrain_mix_v5_boosted_500m_a100_step6000/best.pt \
  --tokenizer tokenizer/helix_v2_tokenizer.model \
  --device cuda \
  --max-new-tokens 96'
```

### Beste lokale SFT-Diagnose bisher

```bash
checkpoints/sft_short_answer_step6000_true_v1/sft_smoke_step_200.pt
logs/sft_short_answer_step6000_true_v1.log
```

## Offene To-dos

- Nächsten echten `best.pt` nach Step 7000/7500/8000 importieren und nicht nach Dateiname vertrauen.
- Kurzantwort-SFT auf neuem `best.pt` wiederholen.
- SFT-Skript erweitern: optional `--save-every`, `--generation-no-repeat-ngram`, `--generation-repetition-penalty`, `--stop-on-end`.
- Capability-Gate erweitern: Stop-Token/Antwortlänge/Repetition stärker gewichten.
- Disjunkte Code-Eval bauen, damit Code-SFT nicht durch oberflächliche Muster besser aussieht.
- Hallucination-Guard-Set erweitern: Goethe/Mein Kampf, falsche Jahreszahlen, Sicherheitsfragen.
- Erst nach stabilem Base + Kurz-SFT einen längeren SFT planen.

