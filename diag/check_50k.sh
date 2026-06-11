#!/usr/bin/env bash
# Fires when v3 reaches step 50000: runs the full check (gen-test + rigorous
# fact-recall margins). val_loss/bpb are already in the training log; SFT is a
# separate step afterwards. By 50k the GPU is free (training done).
set -u
cd /workspace/v2data || exit 1
CKDIR=checkpoints/pretrain_1b_bilingual_de55_en45_foundation_warmstart_v3
CK="$CKDIR/step_50000.pt"
echo "[$(date +%H:%M)] waiting for $CK ..."
for i in $(seq 1 420); do
  [ -f "$CK" ] && break
  if ! pgrep -f "warmstart_v3.yaml" >/dev/null 2>&1; then
     CK=$(ls -t $CKDIR/step_*.pt 2>/dev/null | grep -v emergency | head -1)
     echo "[$(date +%H:%M)] training ended; using newest: $CK"
     break
  fi
  sleep 600
done
sleep 30
echo "[$(date +%H:%M)] running 50k check on $CK"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u scripts/eval/diagnose_checkpoint_generation.py \
  --model-config configs/model/helix_v2_1b.yaml --checkpoint "$CK" \
  --tokenizer tokenizer/helix_v2_tokenizer.model --output diag/gen_test_step50000.json --max-new-tokens 60 \
  && echo "[$(date +%H:%M)] gen-test done -> diag/gen_test_step50000.md"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u scripts/eval/fact_recall_eval.py \
  --model-config configs/model/helix_v2_1b.yaml --checkpoint "$CK" \
  --tokenizer tokenizer/helix_v2_tokenizer.model --output diag/fact_recall_step50000.json > diag/fact_recall_step50000.log 2>&1 \
  && echo "[$(date +%H:%M)] fact-recall done -> diag/fact_recall_step50000.json"
echo "ALL_DONE_50K"
