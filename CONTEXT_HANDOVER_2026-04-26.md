# Auralis v2 — Context-Handover für neue Chat-Session
**Stand:** 2026-04-26 ~13:00 lokal (CEST)
**Vorherige Session-Länge:** ~2 Tage

## Was Auralis v2 ist
- 1B bilinguales (DE/EN) Hybrid-LLM (Helix v2: 6 Mamba + 16 GLA + 6 Sparse Attention)
- Eigener 200k SentencePiece-Tokenizer
- Repository: `/mnt/user/Auralis/AuralisV2/` (Server) = `\BITBASTION\Auralis\AuralisV2` (Windows SMB)
- ⚠️ **NICHT** `I:\AuralisV2` verwenden — ist eine veraltete lokale NTFS-Kopie und Quelle vieler Sync-Probleme

## Infrastruktur
| | |
|---|---|
| Server | 192.168.178.5 (BITBASTION, Unraid) |
| GPU | RTX PRO 5000 Blackwell, 47 GB VRAM |
| Container | `auralis-training` (Python 3.11.12, torch 2.7.0+cu128) |
| Data root im Container | `/workspace/v2data` (= /mnt/user/Auralis/AuralisV2/) |
| SSH | `ssh root@192.168.178.5` |

## Pflicht-Env-Vars für jedes Trainings-Run
```bash
export PYTHONUNBUFFERED=1
export AURALIS_USE_CUDA_KERNELS=1     # mamba_ssm + fla aktivieren
export TRITON_OVERRIDE_ARCH=sm89       # Blackwell sm_120 Triton-Workaround
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # gegen OOM-Fragmentierung
```

## Aktuell laufend (Stand jetzt)
1. **Canary Runde 3** `canary_runde3_de_medium_b16` — batch=12 seq=1024, mix 70/25/5
   - Started 12:44 lokal, ETA fertig ~13:25
   - Log: `logs/runde3_de_medium_b16.log`
   - Ckpt-Dir: `checkpoints/canary_runde3_de_medium_b16/`
2. **Chained: 1B Batch-Size-Sweep** (`scripts/utils/batch_size_sweep.py`)
   - Wartet auf runde3 Ende, startet automatisch
   - Testet helix_v2_1b mit batch [1,2,4,6,8,12] × seq [1024,2048]
   - Output: `logs/batch_sweep_1b.log`

## Sieger-Verdikt der vorherigen 3-Wege-Ablation (batch16, alle FERTIG)
| | baseline (12/3/1) | de_heavy (10/5/1) | code_heavy (10/3/3) |
|---|---|---|---|
| val_loss | 3.286 | 3.652 | 3.912 |
| EN | 2.336 | 2.381 | 2.523 |
| DE | 6.500 | 6.280 | 6.293 |
| Code | 5.541 | 5.594 | 5.451 |

Empfehlung: **Mix 70/25/5** für 1B-Hauptlauf (de_medium = Zwischenstufe, läuft jetzt zur Validierung)

## Daten-Pipeline-Status
### Pretrain (curated_40b, fertig + getestet)
- `tokenized/curated_40b/` — english 11.79B, german 5.60B, code 0.71B Tokens
- Gefiltert mit verbessertem `filter_quality.py` (PROTECTED_PREFIXES + max_repetition CLI)

### SFT-Daten gesammelt (heute):
| Stream | Records | Status |
|---|---|---|
| QA (SQuAD + MS MARCO) → `seeds/sft/qa/qa_combined.sft.jsonl` | 330,319 | ✅ einsatzbereit |
| Coding-Troubleshoot → `seeds/sft/coding_troubleshoot/clean.jsonl` | 19,276 | ✅ einsatzbereit |
| Safety Hard-No → `seeds/sft/safety/safety_hard.jsonl` | 567 | ✅ einsatzbereit |
| Safety Softable → `safety_softable.jsonl` | 13,254 | ⏳ braucht Qwen-Rewrite |
| Safety Normal → `safety_normal.jsonl` | 560,091 | ✅ einsatzbereit |

### Safety-Policy (NEU heute, kritisch):
- `docs/AURALIS_SAFETY_POLICY.md` — 5 Hard-No-Kategorien + Owner-Mode-Mechanismus
- 5 Hard-No: CSAM, WMD-Synthese, konkrete Anschlagsplanung, Doxxing, deployment-ready Malware
- Owner-Mode: System-Prompt-Flag `[OWNER_MODE: true]` schaltet Soft-Refusals ab

## Heute fertig gestellte Tools
- `scripts/data/filter_quality.py` (gepatched)
- `scripts/data/download_qa_seeds.py` (SQuAD + MS MARCO)
- `scripts/data/download_german_speeches.py` (German Political Speeches)
- `scripts/data/download_safety_seeds.py` (HH-RLHF, OASST1/2, WildChat, AdvBench, HarmBench, JailbreakBench)
- `scripts/data/download_kaggle_seeds.py` (license-aware, default class=commercial)
- `scripts/data/process_troubleshoot_seeds.py` (Stack Exchange → SFT)
- `scripts/data/process_qa_seeds.py` (SQuAD/MSMARCO → chat-SFT)
- `scripts/data/categorize_safety_seeds.py` (hard/softable/normal Bins)
- `scripts/data/synth/qwen_client.py` (async OpenAI-API Client)
- `scripts/data/synth/qwen_synth_sft.py` (Refactor-Pipeline für Softable)
- `scripts/utils/runde2_master.sh`, `runde2_b16_master.sh` (Trainings-Master)
- `scripts/utils/batch_size_sweep.py` (gerade in Wartestellung)
- `scripts/utils/diagnose_layer_memory.py` (per-Layer VRAM-Tracker)

## Offene Todos für nach 1B-Sweep
1. 4-Wege-Vergleich kompilieren (baseline, de_medium, de_heavy, code_heavy)
2. 1B Phase-1 Hauptlauf-Config bauen mit Sweep-Empfehlung
3. Vor 1B-Start: `docs/AURALIS_SAFETY_POLICY.md` reviewen + bestätigen
4. Qwen-Endpoint aufsetzen für Safety-Softable-Rewrite (vLLM oder DeepSeek-API)
5. Songtext-4-Säulen-Pipeline (Theory, Public Domain, Qwen-Generation, Reception-Discourse)

## Wichtigste v1-Lektionen (in v2 schon adressiert)
- L-001 Prompt-Format-Konsistenz: noch zu validieren bei SFT-Start
- L-003 Tokenizer locked (200k SP, identity normalization, byte_fallback)
- L-005 `eval/baseline_questions.yaml` (339 Zeilen) existiert, muss in Trainer integriert werden
- L-006 Optimizer-Reset = Default für SFT (noch nicht relevant, da kein SFT bisher)
- L-008 NUL-Strip Pflicht vor Tokenizer (in pipeline aktiv)
- L-012 Blackwell-Triton-Bug → `TRITON_OVERRIDE_ARCH=sm89`

## Working Style für die neue Chat-Session
- Server-Side direkt arbeiten via `ssh root@192.168.178.5 ...`
- Files schreiben über base64-transport (Beispiel im git-history sichtbar)
- Lange Trainings-Runs detached starten: `docker exec -d auralis-training bash -c ...`
- Fortschritt monitoren via Wakeup-Schedule
- ALLE Dateien direkt nach `/mnt/user/Auralis/AuralisV2/` schreiben — nicht I:
## Wichtigste Files zum Kontext-Aufbau (bei Bedarf lesen)
- `docs/AURALIS_SAFETY_POLICY.md`
- `LESSONS.md` (~150 Zeilen, append-only)
- `HISTORY.md` (chronologisch, append-only)
- `STATUS.md`
- `configs/training/canary_runde2_*_b16.yaml` (3 Mix-Varianten)
- `configs/training/canary_runde3_de_medium_b16.yaml` (gerade laufend)
- `configs/model/helix_v2_1b.yaml` (Ziel-Architektur)
