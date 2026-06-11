cd /workspace/v2data
export TRITON_OVERRIDE_ARCH=sm89 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for MODEL in \
  "helix:checkpoints/sft_v1/sft_smoke_step_2000.pt" \
  "helix:checkpoints/pretrain_1b_bilingual_de55_en45_foundation_warmstart_v3/step_50000.pt" \
  "hf:Qwen/Qwen2.5-0.5B" \
  "hf:HuggingFaceTB/SmolLM2-360M" \
  "hf:TinyLlama/TinyLlama_v1.1" ; do
  echo "##### $MODEL #####"
  python scripts/eval/benchmark_mc.py --model "$MODEL" --tasks mmlu,arc_challenge,hellaswag --limit 300 2>&1 | grep -E "RESULT|Error|Traceback|FEHLER" 
done
echo "ALL_BENCH_DONE"
