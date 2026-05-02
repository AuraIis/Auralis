#!/bin/bash
# Master chain for Phase-2 corpus prep:
#   wait for already-running download → clean → next download → clean → ...
#
# Runs detached in the auralis-downloader container. Safe to leave for hours.
# Logs progress to /staging/logs/master_chain.log
#
# Each stage is idempotent: if cleaned/<src>.filtered.txt already exists with
# matching manifest, the clean stage is skipped on re-run.

set -uo pipefail

LOG=/staging/logs/master_chain.log
exec >> "$LOG" 2>&1

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }

echo "============================================================"
echo "=== master_chain start: $(ts) ==="
echo "============================================================"

wait_for_dl() {
    local src="$1"
    while pgrep -f "[d]ownload_phase2_pretrain.py --source ${src}" > /dev/null; do
        sleep 60
    done
}

start_dl() {
    local src="$1"
    echo "--- DL ${src} START: $(ts) ---"
    export HF_TOKEN="$(cat /root/.hf_token)"
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
    export PHASE2_RAW_ROOT=/staging/raw
    nohup python /staging/scripts/data/download_phase2_pretrain.py --source "${src}" \
        > /staging/logs/dl_${src}.log 2>&1 &
    local pid=$!
    echo "    PID ${pid}"
    wait_for_dl "${src}"
    echo "--- DL ${src} END: $(ts) ---"
}

clean_one() {
    local src="$1"
    local out="/staging/cleaned/${src}.filtered.txt"
    if [ -f "${out}.manifest.json" ] && [ -s "$out" ]; then
        echo "--- CLEAN ${src} SKIP (already done): $(ts) ---"
        return 0
    fi
    echo "--- CLEAN ${src} START: $(ts) ---"
    if bash /staging/scripts/clean_phase2_source.sh "${src}"; then
        echo "--- CLEAN ${src} END: $(ts) ---"
    else
        echo "--- CLEAN ${src} FAILED (rc=$?): $(ts) ---"
        return 1
    fi
}

# Step 1: wait for the in-flight fineweb_10bt
echo ">> step 1: wait fineweb_10bt"
wait_for_dl fineweb_10bt
echo ">> fineweb_10bt finished at $(ts)"

# Step 2: clean fineweb (in parallel with starting smollm DL)
echo ">> step 2a: start smollm DL (background)"
export HF_TOKEN="$(cat /root/.hf_token)"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export PHASE2_RAW_ROOT=/staging/raw
nohup python /staging/scripts/data/download_phase2_pretrain.py --source smollm_python_edu \
    > /staging/logs/dl_smollm_python_edu.log 2>&1 &
echo "    smollm DL PID $!"

echo ">> step 2b: clean fineweb_10bt (parallel to smollm DL)"
clean_one fineweb_10bt || echo "WARN: clean fineweb_10bt failed, continuing"

echo ">> step 3: wait smollm_python_edu"
wait_for_dl smollm_python_edu
echo ">> smollm_python_edu finished at $(ts)"

# Step 4: clean smollm in parallel with stack_v2 DL
echo ">> step 4a: start the_stack_v2_python DL (background)"
nohup python /staging/scripts/data/download_phase2_pretrain.py --source the_stack_v2_python \
    > /staging/logs/dl_the_stack_v2_python.log 2>&1 &
echo "    stack_v2 DL PID $!"

echo ">> step 4b: clean smollm_python_edu (parallel to stack_v2 DL)"
clean_one smollm_python_edu || echo "WARN: clean smollm_python_edu failed, continuing"

echo ">> step 5: wait the_stack_v2_python"
wait_for_dl the_stack_v2_python
echo ">> the_stack_v2_python finished at $(ts)"

echo ">> step 6: clean the_stack_v2_python"
clean_one the_stack_v2_python || echo "WARN: clean the_stack_v2_python failed, continuing"

echo "============================================================"
echo "=== master_chain end: $(ts) ==="
echo "============================================================"

echo ">> SUMMARY"
echo "raw sizes:"
du -sh /staging/raw/* 2>/dev/null
echo "cleaned sizes:"
du -sh /staging/cleaned/* 2>/dev/null
echo
echo "manifests:"
for m in /staging/cleaned/*.manifest.json; do
    [ -f "$m" ] || continue
    echo "--- $m ---"
    cat "$m"
done
