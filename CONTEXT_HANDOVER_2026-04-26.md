# Auralis v2 — Context handover for a new chat session
**As of:** 2026-04-26 ~13:00 local (CEST)
**Previous session length:** ~2 days

## What Auralis v2 is
- 1B bilingual (DE/EN) hybrid LLM (Helix v2: 6 Mamba + 16 GLA + 6 Sparse Attention)
- Custom 200k SentencePiece tokenizer
- Repository: `/mnt/user/Auralis/AuralisV2/` (server) = `\BITBASTION\Auralis\AuralisV2` (Windows SMB)
- ⚠️ Do **NOT** use `I:\AuralisV2` — it's an outdated local NTFS copy and the source of many sync problems

## Infrastructure
| | |
|---|---|
| Server | 192.168.178.5 (BITBASTION, Unraid) |
| GPU | RTX PRO 5000 Blackwell, 47 GB VRAM |
| Container | `auralis-training` (Python 3.11.12, torch 2.7.0+cu128) |
| Data root in the container | `/workspace/v2data` (= /mnt/user/Auralis/AuralisV2/) |
| SSH | `ssh root@192.168.178.5` |

## Mandatory env vars for every training run
```bash
export PYTHONUNBUFFERED=1
export AURALIS_USE_CUDA_KERNELS=1     # enable mamba_ssm + fla
export TRITON_OVERRIDE_ARCH=sm89       # Blackwell sm_120 Triton workaround
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # against OOM fragmentation
```

## Currently running (as of now)
1. **Canary Round 3** `canary_runde3_de_medium_b16` — batch=12 seq=1024, mix 70/25/5
   - Started 12:44 local, ETA done ~13:25
   - Log: `logs/runde3_de_medium_b16.log`
   - Ckpt dir: `checkpoints/canary_runde3_de_medium_b16/`
2. **Chained: 1B batch-size sweep** (`scripts/utils/batch_size_sweep.py`)
   - Waits for round 3 to end, starts automatically
   - Tests helix_v2_1b with batch [1,2,4,6,8,12] × seq [1024,2048]
   - Output: `logs/batch_sweep_1b.log`

## Winner verdict of the previous 3-way ablation (batch16, all DONE)
| | baseline (12/3/1) | de_heavy (10/5/1) | code_heavy (10/3/3) |
|---|---|---|---|
| val_loss | 3.286 | 3.652 | 3.912 |
| EN | 2.336 | 2.381 | 2.523 |
| DE | 6.500 | 6.280 | 6.293 |
| Code | 5.541 | 5.594 | 5.451 |

Recommendation: **Mix 70/25/5** for the 1B main run (de_medium = intermediate step, now running for validation)

## Data pipeline status
### Pretrain (curated_40b, done + tested)
- `tokenized/curated_40b/` — english 11.79B, german 5.60B, code 0.71B tokens
- Filtered with the improved `filter_quality.py` (PROTECTED_PREFIXES + max_repetition CLI)

### SFT data collected (today):
| Stream | Records | Status |
|---|---|---|
| QA (SQuAD + MS MARCO) → `seeds/sft/qa/qa_combined.sft.jsonl` | 330,319 | ✅ ready to use |
| Coding troubleshoot → `seeds/sft/coding_troubleshoot/clean.jsonl` | 19,276 | ✅ ready to use |
| Safety Hard-No → `seeds/sft/safety/safety_hard.jsonl` | 567 | ✅ ready to use |
| Safety Softable → `safety_softable.jsonl` | 13,254 | ⏳ needs Qwen rewrite |
| Safety Normal → `safety_normal.jsonl` | 560,091 | ✅ ready to use |

### Safety policy (NEW today, critical):
- `docs/AURALIS_SAFETY_POLICY.md` — 5 Hard-No categories + Owner-mode mechanism
- 5 Hard-No: CSAM, WMD synthesis, concrete attack planning, doxxing, deployment-ready malware
- Owner-mode: system-prompt flag `[OWNER_MODE: true]` turns off soft refusals

## Tools finished today
- `scripts/data/filter_quality.py` (patched)
- `scripts/data/download_qa_seeds.py` (SQuAD + MS MARCO)
- `scripts/data/download_german_speeches.py` (German Political Speeches)
- `scripts/data/download_safety_seeds.py` (HH-RLHF, OASST1/2, WildChat, AdvBench, HarmBench, JailbreakBench)
- `scripts/data/download_kaggle_seeds.py` (license-aware, default class=commercial)
- `scripts/data/process_troubleshoot_seeds.py` (Stack Exchange → SFT)
- `scripts/data/process_qa_seeds.py` (SQuAD/MSMARCO → chat-SFT)
- `scripts/data/categorize_safety_seeds.py` (hard/softable/normal bins)
- `scripts/data/synth/qwen_client.py` (async OpenAI-API client)
- `scripts/data/synth/qwen_synth_sft.py` (refactor pipeline for softable)
- `scripts/utils/runde2_master.sh`, `runde2_b16_master.sh` (training masters)
- `scripts/utils/batch_size_sweep.py` (currently on standby)
- `scripts/utils/diagnose_layer_memory.py` (per-layer VRAM tracker)

## Open todos for after the 1B sweep
1. Compile 4-way comparison (baseline, de_medium, de_heavy, code_heavy)
2. Build 1B Phase-1 main-run config with the sweep recommendation
3. Before the 1B start: review + confirm `docs/AURALIS_SAFETY_POLICY.md`
4. Set up Qwen endpoint for the safety-softable rewrite (vLLM or DeepSeek API)
5. Lyrics 4-pillar pipeline (Theory, Public Domain, Qwen generation, reception discourse)

## Most important v1 lessons (already addressed in v2)
- L-001 Prompt-format consistency: still to be validated at SFT start
- L-003 Tokenizer locked (200k SP, identity normalization, byte_fallback)
- L-005 `eval/baseline_questions.yaml` (339 lines) exists, must be integrated into the trainer
- L-006 Optimizer reset = default for SFT (not relevant yet, since no SFT so far)
- L-008 NUL strip mandatory before tokenizer (active in the pipeline)
- L-012 Blackwell Triton bug → `TRITON_OVERRIDE_ARCH=sm89`

## Working style for the new chat session
- Work server-side directly via `ssh root@192.168.178.5 ...`
- Write files via base64 transport (example visible in git history)
- Start long training runs detached: `docker exec -d auralis-training bash -c ...`
- Monitor progress via wakeup schedule
- Write ALL files directly to `/mnt/user/Auralis/AuralisV2/` — not I:
## Most important files for building context (read as needed)
- `docs/AURALIS_SAFETY_POLICY.md`
- `LESSONS.md` (~150 lines, append-only)
- `HISTORY.md` (chronological, append-only)
- `STATUS.md`
- `configs/training/canary_runde2_*_b16.yaml` (3 mix variants)
- `configs/training/canary_runde3_de_medium_b16.yaml` (currently running)
- `configs/model/helix_v2_1b.yaml` (target architecture)
