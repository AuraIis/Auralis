#!/usr/bin/env bash
# Transient watcher: when v3 reaches step 15000 / 25000, run the generation
# diagnostic on that checkpoint for a before/after vs the step-9000 baseline.
set -u
cd /workspace/v2data || exit 1
D=checkpoints/pretrain_1b_bilingual_de55_en45_foundation_warmstart_v3
for T in 15000 25000; do
  CK="$D/step_$T.pt"
  echo "[$(date +%H:%M)] waiting for $CK ..."
  while [ ! -f "$CK" ]; do sleep 600; done
  sleep 20
  echo "[$(date +%H:%M)] found $CK -> running generation diagnostic"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python -u scripts/eval/diagnose_checkpoint_generation.py \
      --model-config configs/model/helix_v2_1b.yaml \
      --checkpoint "$CK" \
      --tokenizer tokenizer/helix_v2_tokenizer.model \
      --output "diag/gen_test_step$T.json" \
      --max-new-tokens 60 \
    && echo "[$(date +%H:%M)] DONE step $T -> diag/gen_test_step$T.md" \
    || echo "[$(date +%H:%M)] FAILED step $T"
done
echo "ALL_DONE"
