#!/usr/bin/env bash
# Parallel CPU pre-filter (stage 1) for RedPajama + HPLT. Tuned for a 16-PHYSICAL-
# core host: 6 chunks/source = 12 concurrent cleaners, leaving ~4 cores for v3 +
# system. nice -n 15 makes cleaners yield CPU to v3 (training keeps priority).
# Pure CPU — never touches the GPU.
set -u
cd /workspace/v2data || exit 1
N=6
SC=scripts/data/structure_clean_pretrain.py
mkdir -p cleaned/_chunks
rm -f cleaned/_chunks/*

clean_source () {  # $1=raw_src  $2=name
  split -d -n "l/$N" "$1" "cleaned/_chunks/$2_"
  for f in cleaned/_chunks/${2}_*; do
    nice -n 15 python "$SC" --input "$f" --output-jsonl /dev/null --output-text "$f.clean" >/dev/null 2>&1 &
  done
  wait
  cat cleaned/_chunks/${2}_*.clean > "cleaned/$2.struct.txt"
  rm -f cleaned/_chunks/${2}_*
  echo "[$(date +%H:%M)] $2 DONE: $(wc -l < cleaned/$2.struct.txt) docs"
}

echo "[$(date +%H:%M)] start parallel prefilter N=$N/source (12 cleaners, niced) on $(nproc) threads / 16 cores"
clean_source raw/german/redpajama_de.txt rp_de &
clean_source raw/german/hplt_de.txt hplt_de &
wait
echo "ALL_DONE"
