# Auralis 1B Readiness Preflight

- ready_to_launch: False
- eval_prompts: 40
- train_units_scanned: 382763
- hash_collisions: 0
- substring_hits: 0

## Blocking Issues

- data_path_config_not_ready:/workspace/v2data/configs/data_paths_1b_samples_container.yaml:['cleaned_paths_empty', 'tokenized_paths_empty']

## Train Files

- /workspace/v2data/data/training/pretrain_v6_expanded_test_mix/mix_full.txt: exists=True bytes=75177275
- /workspace/v2data/data/training/pretrain_v6_book_sources_gutenberg_v1/book_sources.clean_v2.txt: exists=True bytes=1460573853
- /workspace/v2data/data/training/sft_clean_de_v1/train.helix.jsonl: exists=True bytes=26493815
- /workspace/v2data/data/training/sft_response_fix_de_v8_stable_mix/core_train.helix.jsonl: exists=True bytes=290485

## Data Path Configs

- /workspace/v2data/configs/data_paths_1b_samples_container.yaml: exists=True issues=['cleaned_paths_empty', 'tokenized_paths_empty']

## Source-Disjoint Manifests

- /workspace/v2data/data/training/pretrain_v6_candidates/source_disjoint_manifest_v2.jsonl: exists=True rows=41478 issues=[]
- /workspace/v2data/data/training/pretrain_v6_extra_candidates/source_disjoint_manifest.jsonl: exists=True rows=28550 issues=[]
- /workspace/v2data/data/training/pretrain_v6_book_sources_gutenberg_v1/source_disjoint_manifest_clean_v2.jsonl: exists=True rows=323794 issues=[]

## Policy

- max_eval_prompt_collisions: 0
- max_retention_regressions: 0
- require_cleaned_data_paths: True
- require_source_disjoint_manifest: True
- stop_if_target_not_improving: True
- stop_if_retention_regresses: True
