# Auralis 1B Readiness Preflight

- ready_to_launch: True
- eval_prompts: 70
- train_units_scanned: 11558
- fast_text_files_scanned: 2
- hash_collisions: 0
- substring_hits: 0

## Blocking Issues

None.

## Train Files

- /workspace/v2data/data/training/curated_40b/english.txt: exists=True bytes=56895440970
- /workspace/v2data/data/training/curated_40b/german.txt: exists=True bytes=29662112686
- /workspace/v2data/data/training/sft_clean_de_v1/train.helix.jsonl: exists=True bytes=26493815
- /workspace/v2data/data/training/sft_response_fix_de_v8_stable_mix/core_train.helix.jsonl: exists=True bytes=290485

## Large Text Scan Mode

- /workspace/v2data/data/training/curated_40b/english.txt: mode=large_text_literal_grep bytes=56895440970
- /workspace/v2data/data/training/curated_40b/german.txt: mode=large_text_literal_grep bytes=29662112686

## Data Path Configs

- /workspace/v2data/configs/data_paths_1b_samples_container.yaml: exists=True issues=[]
  - data_root: /workspace/v2data
  - cleaned_counts: {'english': 1, 'german': 1, 'code': 0}
  - tokenized_count: 2

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
