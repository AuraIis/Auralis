# Auralis Docs Index

This index separates current working docs, project idea, historical specs and
experiments. When files contradict each other, `STATUS.md` takes precedence first, then
this index, then the respective current working doc.

## Current Truth

- [../STATUS.md](../STATUS.md) - current state, active direction, current
  runs and open tasks.
- [../README.md](../README.md) - entry point and link hub.
- [../eval/README.md](../eval/README.md) - how checkpoints are evaluated.

## Core Idea And Model Specs

- [../Doc/AURALIS_V2_PROJECT_BRIEF.md](../Doc/AURALIS_V2_PROJECT_BRIEF.md) -
  the big idea: small base model, modular adapters, tools, router,
  memory/LoRA system, training phases.
- [../Doc/SPECs/SPEC_PHASE_0.5_MODEL_ARCHITECTURE.md](../Doc/SPECs/SPEC_PHASE_0.5_MODEL_ARCHITECTURE.md) -
  technical Helix-v2 architecture: hybrid stack of Mamba/SSM, GLA and
  Sparse Attention.
- [../tokenizer/quality_report.md](../tokenizer/quality_report.md) -
  tokenizer efficiency and roundtrip quality.

## Data And Pipeline

- [data_cleaning_pipeline_v3.md](data_cleaning_pipeline_v3.md) - current
  cleaning approach for better pretraining data.
- [dataset_market_app.md](dataset_market_app.md) - dataset search, evaluation
  and mix planning.
- [DATA_PIPELINE_V2.md](DATA_PIPELINE_V2.md) - older pipeline state,
  still useful as a reference.
- [../data/eval/pretrain_clean_v2_audit_v3.md](../data/eval/pretrain_clean_v2_audit_v3.md) -
  important data audit from the rescue phase.
- [../data/eval/training_data_cleaning_report.md](../data/eval/training_data_cleaning_report.md) -
  cleaning report (local; data/eval is gitignored).
- German Edu filter (FineWeb-Edu methodology, 2026-05-31): LLM annotation
  `scripts/data/score_german_edu.py` -> cheap classifier
  `scripts/data/train_edu_classifier.py` (+ `scripts/data/edu_embed.py`) ->
  corpus filter `scripts/data/score_corpus_edu.py`. Source mix of the filtered
  German v2 data: `configs/data_paths.curated_v2_german.yaml`. Judge:
  `qwen3-235b-2507` via OpenRouter; see LESSONS L-018..L-021.

## Training / Multi-GPU

- DDP / Multi-GPU: `scripts/ops/run_pretrain_multigpu.sh` (torchrun launcher).
  The trainer (`src/auralis/training/trainer.py`) is single-process by default
  and activates DDP only when `WORLD_SIZE>1` -> single-GPU path unchanged
  (see LESSONS L-022). Spec:
  [../Doc/SPECs/SPEC_MULTI_GPU_TRAINING.md](../Doc/SPECs/SPEC_MULTI_GPU_TRAINING.md)
  and the "Update 2026-05-31" block in [../STATUS.md](../STATUS.md).

## Blueprints (decided future, gated)

- [BLUEPRINT_TOOL_USE_VERIFIER.md](BLUEPRINT_TOOL_USE_VERIFIER.md) - tool use &
  self-verification: math tool first, harness (stop sequence/sandbox/loop),
  hidden tests as a data gate, Code-DoRA last. Triple-triangulated
  (Michael + GPT + Claude, June 2026). Order in `ZUKUNFT_BACKLOG.md` phase 3-4.
- [BLUEPRINT_DOMAIN_ADAPTERS_DORA.md](BLUEPRINT_DOMAIN_ADAPTERS_DORA.md) - DoRA
  domain adapters (math/logic/code) on a frozen base. Core principle: an adapter
  amplifies latent ability, installs no new one → Code-DoRA locked until
  code annealing. Targeting on the hybrid arch, multi-adapter, gates. (Phase 5)

## Experiments

- [experimental/knowledge_dna_v2.md](experimental/knowledge_dna_v2.md) -
  Knowledge-DNA with `<memory>`/`<recall>` as a booster idea.
- [experimental/knowledge_kernel.md](experimental/knowledge_kernel.md) -
  separate knowledge-kernel test.
- [experimental/memory_kernel.md](experimental/memory_kernel.md) -
  memory-kernel prototype.
- [experimental/math_reasoning_dna.md](experimental/math_reasoning_dna.md) -
  parked idea for compute/reasoning DNA with a mental workspace.

Experiment rule: Nothing from `docs/experimental/` goes into the real
pretraining mix before an ablation shows a clear signal.

## Historical Specs

These files are valuable, but not automatically the current run plan:

- [../Doc/SPECs/SPEC_PHASE_0_TOKENIZER.md](../Doc/SPECs/SPEC_PHASE_0_TOKENIZER.md)
- [../Doc/SPECs/SPEC_PHASE_1_PRETRAINING.md](../Doc/SPECs/SPEC_PHASE_1_PRETRAINING.md)
- [../Doc/SPECs/SPEC_PHASE_2_CONTINUED_BILINGUAL.md](../Doc/SPECs/SPEC_PHASE_2_CONTINUED_BILINGUAL.md)
- [../Doc/SPECs/SPEC_PHASE_3_SFT.md](../Doc/SPECs/SPEC_PHASE_3_SFT.md)
- [../Doc/SPECs/SPEC_PHASE_4_ORPO_ALIGNMENT.md](../Doc/SPECs/SPEC_PHASE_4_ORPO_ALIGNMENT.md)
- [../Doc/SPECs/SPEC_PHASE_5_LORA_SYSTEM.md](../Doc/SPECs/SPEC_PHASE_5_LORA_SYSTEM.md)
- [../Doc/SPECs/SPEC_MULTI_GPU_TRAINING.md](../Doc/SPECs/SPEC_MULTI_GPU_TRAINING.md)

## References

- [../Doc/REFERENCES/attention_and_position_encoding.md](../Doc/REFERENCES/attention_and_position_encoding.md)
- [../Doc/REFERENCES/data_pipeline_v1.md](../Doc/REFERENCES/data_pipeline_v1.md)
- [../Doc/REFERENCES/data_pipelines_and_frameworks.md](../Doc/REFERENCES/data_pipelines_and_frameworks.md)
- [../Doc/REFERENCES/mora_integration.md](../Doc/REFERENCES/mora_integration.md)

## Known Cleanup Debt

- Some old Markdown files contain mojibake/encoding remnants
  (broken umlaut/dash sequences). That's doc dirt, not a
  training blocker.
- Old phase specs contain sizes like 2-3B/3B, while current
  canary work is at 500M.
- Old paths like `curated_40b`, `phase1_pretrain` and `tokenized/phase1`
  may be historically correct, but are no longer the current default.
