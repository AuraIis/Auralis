# HISTORY — Auralis v2

Chronological, milestones. Append-only.

---

## 2026-04-22 — Project start Auralis v2
- Brief + 7 phase SPECs + `SPEC_DATASETS.md` finalized in `Doc/`
- Project skeleton created (`pyproject.toml`, directory tree, `.gitignore`)
- `STATUS.md`, `LESSONS.md`, `HISTORY.md` initialized
- Memory system for Claude Code populated (user / project / feedback / v1 lessons / v1 datasets)
- **Decision: model size 1B** (instead of 2-3B) — faster, cheaper, doubles v1 capacity but stays small.
- **Decision: data root `//BITBASTION/Auralis/AuralisV2/`** (NAS, 25 TB free); v1 SFT pool `I:/Auralis/NEWGPT/data/` stays local.
- Phase 0 preliminary work complete: baseline (50 questions), byte-identical prompt builder + tests, data config + 3 Phase-1 download scripts + v1 inventory script.
- Guideline anchored: synthetic data generation is desired when open-source sources are thin (DeepSeek V3 / Qwen 3.5 30B local — v1-proven).

## 2026-04-22 — Phase 0 completed (tokenizer + Phase-1 data)
- **v1 DE reuse:** 23.7 GB deduplicated German pretraining material on NAS (~4.7 B tokens, 8.87 M docs), 9:35 min.
- **EN downloads:** Wikipedia EN (12 GB), FineWeb-Edu sample-10BT (40 GB), OpenMathInstruct-2 (8 GB) — together ~15 B tokens.
- **Code downloads:** StarCoderData 9-language subset (3.5 GB) + open-web-math (0.88 GB) — together ~1.25 B tokens.
- **Not loaded** (dataset HTTP 404 or `datasets` v4+ script ban): SlimPajama, Dolma, Proof-Pile-2. Total Phase 1 coverage = **~21 B / 25 B = 84 %**, gap reserved for Phase 2.
- **Tokenizer corpus:** 15.5 GB mix (50 EN / 40 DE / 10 Code), NUL-cleaned.
- **Tokenizer training (SentencePiece Unigram, 200 k vocab, 32 threads):** 14.6 min.
- **Quality report PASS:** EN 123 tok/100 w (target ≤135), DE 133.8 (≤150, v1 was ~220), Code 313.6 tok/KB (≤350), Unknown 0 %, **chat template roundtrip byte-exact**.
- **New Lessons L-007..L-012** in `LESSONS.md`: SP normalization `identity` mandatory, NUL strip required, `num_threads ≥ 1`, `input_sentence_size = 5 M` with 32-GB RAM, HF v4 `Dataset-scripts` ban, code metric switched to `tokens/KB`.

## 2026-04-23 — Phase 0.5 completed (model architecture)
- `src/auralis/model/` fully implemented: config dataclass, RMSNorm, SwiGLU FFN, Mamba-2 (pure torch), Gated Linear Attention (pure torch), Sparse Attention with sliding window + global tokens, RoPE, scaled-normal init, KV cache dataclass.
- `helix_model.py`: `HelixBlock` pre-norm style, `HelixModel` with heterogeneous stack from config, `build_model(yaml)` factory, tied-embedding LM head.
- Two configs: **`helix_v2_100m.yaml`** (8 layers, 134 M params, test model for CPU) and **`helix_v2_1b.yaml`** (28 layers, d=1280, ~954 M params — hits the 1B target within 5 %).
- **50/50 tests green** in 2.7 s on CPU: config loading + validation, all layers (shapes, backward, causal masking, window masking, global-tokens bypass, RoPE-norm preservation), end-to-end forward/backward on 100M model.
- Forward on a freshly-initialized 100M model yields loss **12.37** — close to the theoretical value of `ln(200000) = 12.20` for a uniform prior over the 200k vocab. No NaN/Inf.
- Pure-Python variants of Mamba-2 and GLA provide the reference semantics for CPU tests. For real GPU pretraining, Phase 1 will additionally wire in `mamba_ssm` and `flash-linear-attention` (config flag, interface stays the same).

## 2026-04-23 — Phase 1 pretraining pipeline finished (launch-ready)
- `scripts/data/tokenize_for_pretraining.py`: batched SentencePiece encoding → uint32 .bin + int64 .idx per language, atomic writes + manifest, resume-safe.
- `src/auralis/training/` complete: `PretrainDataset` (memmap), `MixedDataLoader` with largest-remainder partitioning for mix ratios, `build_optimizer` with decay split for norms/biases, `build_scheduler` (cosine + constant_with_warmup), `PretrainTrainer` with gradient accumulation, grad clip, checkpoint rotation, NaN abort, val-loss alarm after 3 regressions.
- `scripts/pretrain/train_phase1.py`: CLI entry with preflight + resume + device override + `torch.compile` flag.
- `scripts/pretrain/smoke_test.py`: end-to-end validation in 30 s (134M model, 20 steps, synthetic tokens, checkpoint roundtrip).
- `configs/training/phase1_pretrain.yaml`: 80 k steps × 128 effective batch × 2048 tokens = ~21 B tokens (matched to the actual data coverage).
- **64/64 tests green in 4 s** (+14 new training tests: `dataset`, `optimizer`, `trainer`).
- Tokenization of the 88 GB Phase-1 data started in parallel (background) — throughput ~6 MB/s over SMB, estimated duration ~4 h.
- **Launch guide** `docs/PHASE_1_LAUNCH.md` covers RunPod setup, preflight, monitoring, rollback procedures, and milestone expectations.
- **The GPU launch itself was NOT started automatically** — that costs $500-800 on RunPod and needs Michael's explicit decision + account setup.

## 2026-04-23 — Blackwell GPU validation on Unraid
- Auralis Docker container (Ubuntu 22.04, Python 3.11, torch 2.7.0+cu128) used on the Unraid host with RTX PRO 5000 Blackwell (47 GB VRAM).
- V2 data directory mounted into the container via `mount --bind /mnt/user/Auralis/AuralisV2 /mnt/user/Auralis/NEWGPT/v2data` + `docker restart` (SHFS does not propagate running mounts).
- Libraries installed: `flash-linear-attention`, `mamba-ssm 2.3.1`, `causal-conv1d 1.6.1`, `flash-attn 2.8.3` (all cu128-compatible). After the Triton 3.6 upgrade, both Triton-based libraries (mamba-ssm + fla) compile.
- **Library-swap hooks built in**: activatable per layer type via env var (`AURALIS_USE_CUDA_KERNELS`, `AURALIS_USE_MAMBA_KERNEL`, `AURALIS_USE_GLA_KERNEL`, `AURALIS_USE_FLASH_ATTN`). Interface for `HelixBlock` unchanged; default stays native pure-torch. The GLA backend supports the same parameter shapes native/fla, the Mamba backend does not (architectural difference).
- **Smoke-test results (250M, bf16, batch=4):**
  - seq=256 native: 147 tok/s, 13.0 GB VRAM, loss 12.16 → 11.59 (Δ+0.57, learning)
  - seq=512 native: 82 tok/s, 24.85 GB VRAM
  - seq=512 gla-kernel: 88 tok/s, **21.27 GB VRAM (-14 %)**, numerically identical to native
  - seq=512 gla+flash: identical results (Sparse only covers 3/12 layers)
- **Blackwell insight:** the main benefit of the kernels is **VRAM savings** (chunkwise instead of a materialized [B,L,H,D,D] state), not primarily tok/s. The tok/s speedup becomes more pronounced at the seq=2048 Phase-1 config.
- **Mamba kernel currently problematic on Blackwell** — Triton compile bug in mamba_ssm itself with Triton 3.6. OK for RunPod H100/H200; for Blackwell, leave Mamba native.
- **(Later the same day) Blackwell fix found: `TRITON_OVERRIDE_ARCH=sm89`** — emulates Ada (compute capability 8.9) on Blackwell (sm_120). Neither sm90 (WGMMA intrinsics missing) nor default (sm_120 unknown) work; sm89 is the first backward-compatible target that Blackwell accepts. All three kernels (mamba_ssm + fla + flash-attn) now run simultaneously, numerically identical to the native reference.
- **Final measurements Blackwell, 250M bf16, ALL kernels active:**
  - seq=256 batch=4: 220 tok/s, 6.68 GB VRAM (-49 % vs native)
  - seq=512 batch=8: 1 928 tok/s, 16.36 GB (23× vs 3090 pure-python)
  - seq=1024 batch=4: 3 628 tok/s, 16.84 GB
  - **seq=2048 batch=2: 2 713 tok/s, 17.74 GB — Phase-1 config validated, loss Δ +1.52 in 15 steps**
- Docs (`docs/PHASE_1_LAUNCH.md`) extended with the sm89 workaround and per-hardware kernel setup.

---

## 2026-04-26 — Canary round 3 + code-review bugfixes + 1B sweep
- **Canary round 2 b16 completed** (3 mix variants, baseline=10:34Z): val_loss baseline 3.286 / de_heavy 3.652 / code_heavy 3.912. Baseline (12/3/1) wins overall, de_heavy catches up on DE (6.500 → 6.280), code_heavy only minimally on Code (5.541 → 5.451) with a clearly worse overall picture.
- **Round 3 `de_medium_b16` (mix 70/25/5, 11/4/1)** run as an intermediate-stage validation. Eval @ step 5000: **val_loss 3.653, EN 2.380, DE 6.280, Code 5.592** — practically identical to de_heavy, no expected gain from the intermediate stage. Baseline remains the better candidate overall for the 1B main run, but the DE gap (6.5 → 6.28) is real.
- **Disk crisis at step ~1175 of round 3:** tok/s collapsed from 33k to 2-4k, data_wait rose to 90%+. Root cause: a parallel-running v1-lessons audit with `shuf -n N` on the 56 GB EN training files blocked sdc at 100% util. Cleanup: all `shuf` + parent-bash loops killed (PIDs 444973/465158/492246), the trainer recovered to 27-29k tok/s, the run finished cleanly. Lesson L-013 created. Collateral damage: the chained sweep wrapper was in the same process group → had to be restarted manually later.
- **Code-review pass over trainer + model + dataset** produced 3 findings:
  - **P1 ([trainer.py:359-360](src/auralis/training/trainer.py#L359))** — `_rotate_checkpoints()` parsed `step_<n>_emergency.pt` position-based → `ValueError`, crashing the auto-stop path right after the emergency save. Fix: regex-based step extraction. → L-014
  - **P1 ([helix_model.py:108-117](src/auralis/model/helix_model.py#L108))** — the shared `RotaryEmbedding` was only built when ≥1 `sparse_attention` layer existed; pure `plain_attention` stacks with `use_rope=true` ran without position encoding. Fix: the build condition now checks all layer types. → L-015
  - **P3 ([dataset.py:71-74](src/auralis/training/dataset.py#L71))** — sampler off-by-one due to the exclusive upper bound of `Generator.integers()`, the last legal window was never drawn. Fix: `+1` on the upper bound.
  - 3 regression tests added (`test_emergency_checkpoint_does_not_break_rotation`, `test_pretrain_dataset_can_sample_last_valid_window`, RoPE plain-attention smoke). Server run: **100/100 green** (97 old + 3 new).
- **1B batch-size sweep** restarted manually (PID 932424) — tests `helix_v2_1b` × batch [1,2,4,6,8,12] × seq [1024,2048] with mandatory env block. Output: `logs/batch_sweep_1b.log`. The result steers the 1B Phase-1 main-run config.
- **L-014 follow-up fix:** a second review pass revealed that the regex `r"step_(\d+)(?:_.*)?$"` removes the `ValueError`, but regular and emergency ckpts at the same step fall under the same sort key — on a same-step collision, a snapshot was silently rotated depending on glob order (emergency could disappear). Fix: rotation narrowed to strictly `r"step_(\d+)$"`, emergency snapshots are now fully rotation-exempt. Regression test `test_emergency_checkpoint_survives_same_step_rotation` added. Server: **101/101 green**.
- **2 remaining P2 findings documented in STATUS.md → Known technical debt** (gradient_checkpointing override contract + sweep `--no-grad-ckpt` ineffective) — both without production impact for current configs.
- **P2 backlog resolved:** both findings fixed in a second pass by extracting the shared logic into [src/auralis/training/utils.py](src/auralis/training/utils.py) (`resolve_gradient_checkpointing()` as a real override + `apply_gradient_checkpointing()` for explicit enable/disable). [train_phase1.py](scripts/pretrain/train_phase1.py) and [batch_size_sweep.py](scripts/utils/batch_size_sweep.py) now use the shared path. 4 new regression tests in [test_utils.py](tests/training/test_utils.py). Server: **105/105 green**.
- **1B batch-size sweep completed** (`helix_v2_1b`, mix 70/25/5):
  - seq=1024: max batch=12 OK, top tok/s at batch=8 (3.4k tok/s, 19 GB)
  - seq=2048: max batch=8 OK, batch=12 → OOM
  - **Top throughput: seq=2048 batch=4 → 11.3k tok/s @ 23.3 GB peak** (23.7 GB reserve on a 47 GB budget)
  - Notable: batch=8 only gains +6% tok/s over batch=4 but costs +84% VRAM → **batch=4 is the sweet spot**.
  - Recommendation for the 1B Phase-1 main-run config: **seq=2048, batch=4**, gradient_checkpointing=on. Full table in `logs/batch_sweep_results.json`.
- **Zombie wrapper cleaned up:** PID 931991 (an old chained `wait && start_sweep` wrapper from the previous chat) had a self-referential bug — the wait condition `pgrep -f "train_phase.*runde3"` matched its own command-line string (the python argument contains both strings) and would never have terminated. It survived the shuf cleanup but never triggered the sweep; my manual sweep was luckily the only one running. Lesson: the wait-wrapper pattern must phrase `pgrep` so its own command line does not match (e.g. `pgrep -f "[t]rain_phase1.py.*runde3"` or via a PID file instead of pgrep).
- **Pre-main-run config audit (3 findings addressed):**
  - **Critical:** [phase1_pretrain.yaml:74](configs/training/phase1_pretrain.yaml#L74) still had the old `monitoring.alert_on:` schema with fields (`val_loss_increase`, `grad_norm_explosion`, `nan_in_loss`) that the trainer (`trainer.py:181` reads `monitoring.health.<field>`) would have completely ignored — i.e. the user thresholds for the 1B main run would have been silently dropped, only HealthConfig defaults active. Fix: complete rewrite to the real HealthConfig schema (`monitoring.health.grad_explosion_threshold: 100.0`, `monitoring.health.val_regression_stop_k: 3`). Verified via `HealthConfig(**cfg)` roundtrip on the server.
  - **Sweep result incorporated:** [phase1_pretrain.yaml:44-45](configs/training/phase1_pretrain.yaml#L44) `batch_size_per_device: 8 → 4`, `gradient_accumulation: 16 → 32`. Effective batch stays 128, but the microbatch now uses the 1B sweep sweet spot (11.3k tok/s @ 23 GB instead of a suboptimal batch of 8).
  - **Doc drift:** [docs/PHASE_1_LAUNCH.md](docs/PHASE_1_LAUNCH.md) — install command `pip install -e ".[train]"` (extra does not exist in pyproject.toml) → `.[all-linux]` (pulls pretrain + posttrain + lora + inference + dev including mamba-ssm/flash-attn/fla via the pretrain extra). Plus stale "64/64 tests" → "105/105", stale "tokenization still running" → "tokenization finished (~21B tokens)".
- **Remaining open recommendation (not executed):** copy the token bins via `rsync` from SMB to local NVMe in the container before the 1B main run — the biggest operational stability gain after the shuf/SMB crisis. Pending decision.
- **NVMe stage pulled through:** 69 GB token bins via rsync from `disk6` (HDD) to `/mnt/cache/auralis_tokens_local/curated_40b/` (NVMe btrfs). SHA256 verify byte-identical, host-side `mount --bind` over `/mnt/user/Auralis/AuralisV2/tokenized/curated_40b`, container restart, visible inside the container as `/dev/nvme0n1p1 on /workspace/v2data/tokenized/curated_40b type btrfs`. Sequential read benchmark: **NVMe 1.3 GB/s vs HDD 30.5 MB/s = 42× speedup**. Reboot persistence deliberately deferred (User Scripts entry later, after a successful main run). Collateral: another chat meanwhile ran `OpenText`+ReadLine loops on the 87 GB raw text files in parallel (a second L-013 replay in one day) — rsync throughput dropped from 124 MB/s to 17 MB/s, stopped immediately after a user notice, rsync recovered to 124 MB/s. Memory `feedback_disk_diagnosis.md` sharpened with an active cross-chat coordination rule.
- **Pre-main-run smoke pass (50 steps, production-equivalent settings):** new [phase1_smoke.yaml](configs/training/phase1_smoke.yaml) as a fork of phase1_pretrain.yaml with only `total_steps: 50`, eval/save/wandb disabled, separate `output_dir`. Smoke v1 uncovered a hidden mix bug: with micro-batch=4 and code=5%, `_partition_rows()` (largest-remainder) rounded down to 0 code rows per micro-batch → across 80k×32 micro-batches **never saw code**. Another chat made the fix: stratify small mix fractions fairly across multiple micro-batches instead of hard-rounding down per micro-batch, with expectation-value logging in the trainer (`train expected rows/batch per language`). 27 tests + server suite 106/106 green. Smoke v2 with the patched DataLoader: **tok/s 13.0k (vs sweep 11.3k), data_wait 0.3-0.5%, VRAM 17.7 GB peak, loss 12.41→8.85 in 50 steps** — all health guards quiet, code rows now appear at the expected fraction in the mix.
- **3 more pre-main-run config findings in the smoke-preparation pass:**
  - **Critical:** `data.data_dir` pointed to `//BITBASTION/Auralis/AuralisV2/tokenized/phase1` (SMB path AND old predecessor subdir!) — the real main run would have crashed immediately. Fix: `/workspace/v2data/tokenized/curated_40b` (container path, via NVMe bind).
  - **Mix inconsistency:** `mix_ratios` were 75/20/5 instead of the winner 70/25/5 — fixed.
  - **`external_backup.path`** pointed to the SMB path `//BITBASTION/...checkpoints/phase1` (same pattern). The backup would have run with no effect (the trainer catches backup errors). Fix: `/checkpoints/phase1_pretrain_backup` (container mount on the disk6 array, write test verified).
- **🚀 1B Phase-1 main run started** (~18:18 local): PID 225 in the container, detached. Config: helix_v2_1b (0.90B params), seq=2048 batch=4 grad_accum=32 (effective 128), mix 70/25/5, 80k steps (~21B tokens), gc=on, torch_compile=on, --no-wandb (wandb was verified not authenticated in the container, safer path chosen). Token reads from NVMe cache, checkpoints on the disk6 HDD (cache reserved for tokens). Expected wall clock: **~12-19 days** on RTX PRO 5000 Blackwell. Health thresholds active (grad_explosion=100, val_regression_stop_k=3). Logs: `logs/phase1_pretrain.log`, primary ckpts: `checkpoints/phase1_pretrain/`, backups every 10k steps to `/checkpoints/phase1_pretrain_backup/`.

---

## 2026-04-29 — Phase-3 SFT data pipeline + WSL inference setup
- **Trainer trajectory very healthy:** the 80k main run at step 8340 (10.4%), val_loss trajectory on plan: 1k=3.44, 2k=2.37, 3k=2.05, 4k=1.92, 5k=1.84, 6k=1.76, 7k=1.74, 8k=1.68. EN val_loss in particular below 0.9 (perplexity ~2.46) — the model learns EN at ~50% top-1 next-token accuracy. tok/s steady 12.9k, data_wait 1-2%, no health trigger.
- **OpenRouter + DeepSeek V4 Flash/Pro pipeline** built for Phase-3 SFT data generation: [scripts/data/synth/deepseek_v4_client.py](scripts/data/synth/deepseek_v4_client.py) async client with task-type-based Pro/Flash routing. Pro for code-engineering (idiomatic patterns), Flash for tutorial/explainer (preference-confirmed via A/B test 2026-04-28). Resume-safe, cost-tracking, optional reasoning_content extract.
- **Phase-3 Batch1** (980 examples, $0.37): smoke-quality validation across 11 task_types. Quality audit: 100% correct on code/math, 70% auto-clean refusals (the rest also OK on manual inspection), no obvious hallucinations. Insight: step_by_step_reason avg 5421 tokens — far too long.
- **Phase-3 Batch2** (5600 examples, $1.96 + $0.04 retry, all errors transient 504): with max_tokens=1500 cap on step_by_step_reason (verbosity 5421→1305 tok, -76%) plus DE-deep topics (law, history, literature, language, DACH, ~1500 additional prompts). 132 transient errors fully resolved via retry, the dedup pass kept only 5600 clean unique IDs. But: 2 real hallucinations discovered in honest_refusal — both on the "Who designed Goethe's office chair?" prompt (Funk/Bertuch confabulated). 7/9 samples of the same prompt correctly refused.
- **Anti-hallucination A/B test** (310 examples, $0.024): NEW system prompt for honest_refusal with (a) explicitly forbidden speculation markers (vermutlich/wahrscheinlich/soll/angeblich), (b) few-shot good-vs-bad examples, (c) allowed verifiable context debunk. **Result: 0% hallucination rate** (vs ~3% baseline), 91% explicit refusal markers, avg-out 143 tok instead of 241 (more concise thanks to forbidden filler waffle). → Lesson L-017 documented. New prompt adopted in [generate_phase3_inputs_v2.py](scripts/data/synth/generate_phase3_inputs_v2.py).
- **WSL2 + RTX 3090 inference setup:** complete Linux inference environment locally for regular Phase-1/2/3 iterations without server disruption. Stack: WSL2 Ubuntu 24.04, Python 3.12, torch 2.11.0+cu128, mamba_ssm 2.3.1, causal-conv1d 1.6.1, flash-linear-attention 0.5.0 (all Linux-only libs without issues). flash-attn not installed (the server trainer uses sparse_attention:native, we don't need it). best.pt copied locally via scp. The inference script loads 0.90B in 5.6s, generates at **30-40 tok/s on the 3090**, peak GPU mem 2.10 GB. First outputs at step 7000: classic pretrain-state behavior — DE/EN grammar perfect, factual knowledge unreliable, topic drift on longer generation. Expected at only 8.75% of the main run.
- **Cost balance Phase-3 data today:** $2.37 of the ~$11 OpenRouter budget, 7188 SFT examples in the pipeline (980 + 5598 + 300 v1 + 310 v2 honest_refusal), production-quality validated.
- **Doc updates:** [LESSONS.md](LESSONS.md) L-017 (helpful-elaboration trap), [STATUS.md](STATUS.md) Phase-3 data status pending.

## 2026-04-30 → 2026-05-30 — Interim status (reconstructed from git log + `reports/`)

> These ~29 days were not captured in HISTORY. **Reconstructed** from the git commit log (up to HEAD `0cfe26f`, 2026-05-04 — nothing was committed after that) and the dated files in `reports/` and `docs/` (05-24 .. 05-30). No invention — where run metrics are missing, only what the artifacts substantiate is stated.

**~04-24 → 05-04 — Phase-2 prep + industrial data pipeline (git-substantiated):**
- `feat: Phase-2 prep, DE-Chain, Politik-Korpus, infra hardening` + three Codex review passes (P1-P3, 6 findings, 2 edge cases).
- **Industry-standard data + eval pipeline** wired up, **Track-2 benchmark suite** (`feat(eval)`), opt-in speed knobs for Phase-2+ (`feat(perf)`).
- Reference docs: attention variants + positional encoding, MoRA integration plan, data pipeline / framework evaluation (Tier-1 tested), `data_pipeline_v1.md`.
- **Ask-LLM quality scorer** (Nemotron-CC style): `rewrite_low_quality.py`, `ask_llm_code.py`, `ask_llm_local.py` (direct-HTTP for LocalAI/vLLM), chunked streaming + `--resume`, `--min/--max-chars` prefilter.
- **Scorer-model debugging:** DeepSeek garbage loop → temperature 0→0.05 → default `llama-3.3-70b` → later `qwen3.6-35b-a3b` (local bitbastion model). Matches the later judge lesson **L-019**.
- **RunPod:** phase1-resume config + pod setup script.

**~05-24 → 05-26 — 500M-v5 forensics + v6 data plan:**
- `500m_step6000_sft_gate` (05-24); `pretrain_v5_500m_a100_root_cause` (05-26) — root-cause analysis of the weak v5-500M.
- `pretrain_v6_data_eval_plan`, `candidates_audit` / `manifest_combined`, `canary_500m_bitbastion` (05-26): new v6 data-candidate pool defined + audited.

**~05-27 — v6 data mixes built + contamination-checked:**
- Gutenberg-Books clean_v2/v1 (contamination + manifests + tokenized manifests), `books_augmented` / `books_lowratio` / `expanded_test` 500M mixes, `strict_mix`, `instruction_pool` / `instruction_de_strict`, extra candidates.
- First `sft_response_fix_de` v1/v2 (microfit, curriculum-mixed, diag eos/weighted, guard/core-only, stabilize-from-core).

**~05-28 — large SFT-repair sweep (500M):**
- `sft_response_fix_de` **v3 → v9** with dozens of variants (anchor, balance/guard-patch, family-from-v4best, bridge, bonn_photo, stable-from-v6/v8) against the interference axes (Bonn/Berlin, photosynthesis, Faust/Goethe).
- Semantic-gate sweeps (a2, v3..v9, contrastive/balanced/strong-probe-tune) over the `sft_response_fix_chat_gate` v2..v6 holdouts.

**~05-29 — pivot to 1B readiness + frozen/live gates:**
- `1b_readiness_plan`, `staged_training_plan`, `codex_handoff_1b_readiness_v2`; **1B preflight** (v2 + curated_40b_v2) → `ready_to_launch: False`.
- **Frozen-response gate v2** over the 500M checkpoints (v8_safe, hybrid_v1_40, hybrid_v12_bridge_60, hybrid_v12_repair_v2_80) + leak checks → **no checkpoint promotable** (target weak, retention breaks).
- **Adaptive live bridge** (frozen gate live in training), `learning_neuro_map` + `learning_trace_system`, v10 (source-disjoint) / v11 (contrastive) SFT.
- STATUS snapshot 05-29: 500M not production-ready; diagnosis = **interference**, way forward = a cleanly weighted 1B mix instead of further 500M mini-patches.

**~05-30 — curated_40b canary / 1B mix A/B (transition to this session):**
- `1b_language_tree_plan`, `1b_lr2e4_mix_ab_smoke`, `1b_readiness_preflight_curated_40b_canary`, `frozen_response_gate sft_response_v2 curated_40b_canary`.
- Transition into the bilingual **1B ramp (de55/en45)**, which was analyzed in this session at step ~3400 (see next entry).

## 2026-05-30/31 — German-Edu filter (FineWeb-Edu methodology) + multi-GPU/DDP

- **Starting point:** bilingual 1B ramp (de55/en45) up to step ~3400 (best.pt), learning behavior disappointing. **Clean diagnosis:** not the eval (Qwen-2.5 on the probes 37/50 = sensible, not broken), not the architecture (all-plain-attention control ~ on par with Helix up to step 300), but **under-training** (~3.4B tok ~16% Chinchilla) **+ a quality-inverted DE mix** (weakest source = largest budget).
- **German-data audit:** heuristic refilter pointless (data not trashed, ~0.01% drops). The real lever = an **educational-value filter** like fineweb_edu (English had an edu score, German never did).
- **Edu annotation built** (`scripts/data/score_german_edu.py`, OpenAI-compatible): first tested **gemini-3.5-flash** → **€24 cost shock** (thinking tokens eat `max_tokens` + are billed as expensive output) → run killed. Switched to **`qwen/qwen3-235b-a22b-2507` via OpenRouter** (non-thinking, ~40× cheaper, **stricter AND more accurate** on web text — Gemini was too lax on EuroParl fragments). 12k labels, **~€1**.
- **Distributions (Qwen, ≥3):** wikipedia_de 85 % · fineweb2_de 25 % · german_commons **4.8 %** (almost only parliament/OCR fragments).
- **Cheap classifier** (`edu_embed.py` frozen multilingual-e5-large + `train_edu_classifier.py` Ridge head + threshold calibration): val **Pearson 0.866, Keep-F1 0.872**; reproduces the LLM judgment on held-out (294 docs/s). `score_corpus_edu.py` filters the full corpus.
- **German-v2:** fineweb2_de @≥2.0 (~38 % keep) + wikipedia_de in full, **german_commons dropped** → ~2.0B high-quality DE tokens (`configs/data_paths.curated_v2_german.yaml`). Enough for the ~1.8B DE need of the foundation run without repetition.
- **Multi-GPU/DDP** built into the trainer (`trainer.py`, `train_phase1.py`, `scripts/ops/run_pretrain_multigpu.sh`), strictly `WORLD_SIZE>1`-gated → single-GPU bit-identical (verified). DDP-agnostic checkpoints, `no_sync`, rank-0 eval+barrier, global stop. **Measured: 12.9k tok/s/GPU** → full 1B (~20B tok) ~18 days (1 GPU) / ~5 days (4 GPU). Not yet validated on real multi-GPU (test box = 1 GPU).
- **Dataset review** (four user suggestions): RedPajama-V2-de = a real modern-DE scaling lever (3T, quality signals); **german-commons rejected** (stream front-loaded with OCR historica — BLBooks/DiBiLit/GermanPD, ppl 500-1000+; reinforces L-004 → L-020); babylm-german too small; **multitask_german_32k** saved as SFT data to `raw/sft/`.
- **Infra:** Colab vs RunPod analyzed → RunPod cheaper + more practical for multi-day runs (Spot thanks to resume); Colab unsuitable. The 1B foundation runs for free on BITBASTION (1 GPU).
- **Versioning:** two focused commits on `feat/multigpu-ddp` (DDP `eb4f833`, Edu pipeline `95d71ba`), pushed, **PR #1** open.
- **New Lessons:** L-018..L-022.
