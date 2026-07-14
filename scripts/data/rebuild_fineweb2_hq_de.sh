#!/usr/bin/env bash
# Conservative FineWeb2-HQ German ingest and tokenization.
#
# This source is already globally deduplicated and language/quality filtered
# upstream. Do not reapply generic web hard filters or flatten JSONL first.
set -euo pipefail

ROOT="${ROOT:-/workspace/v2data}"
WORK="${WORK:-$ROOT/data/de_rebuild_v2}"
SRC="${SRC:-$ROOT/data/fineweb2_hq_de.dedup.jsonl}"
FILTER="${FILTER:-$ROOT/scripts/data/filter_quality.py}"
TOKENIZER_SCRIPT="${TOKENIZER_SCRIPT:-$ROOT/scripts/data/tokenize_for_pretraining.py}"
TOK="${TOK:-$ROOT/tokenizer/helix_v2_tokenizer.model}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-curated_de_v2}"
EXPECTED_TOK_SHA="a24fbea439bc8b78c78653b9febf708d96cf023745199d8f6e7c0b3f6285f2bc"

mkdir -p "$WORK"
ts() { date -u +%H:%M:%S; }

echo "[$(ts)] STAGE0 provenance and preflight"
python - "$SRC" "$FILTER" "$TOKENIZER_SCRIPT" "$TOK" "$WORK/run_manifest.json" \
  "$EXPECTED_TOK_SHA" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

src, filt, tok_script, tok, output, expected = map(Path, sys.argv[1:7])
dedup_report = src.with_suffix(".dedup_report.json")
for path in (src, dedup_report, filt, tok_script, tok):
    if not path.is_file():
        raise SystemExit(f"missing required input: {path}")

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

tok_sha = sha256(tok)
if tok_sha != str(expected):
    raise SystemExit(f"tokenizer mismatch: {tok_sha}")
payload = {
    "started_at": datetime.now(timezone.utc).isoformat(),
    "source": {
        "path": str(src),
        "size_bytes": src.stat().st_size,
        "mtime_ns": src.stat().st_mtime_ns,
    },
    "filter": {"path": str(filt), "sha256": sha256(filt)},
    "tokenizer_script": {"path": str(tok_script), "sha256": sha256(tok_script)},
    "tokenizer": {"path": str(tok), "sha256": tok_sha},
    "dedup_report": {
        "path": str(dedup_report),
        "sha256": sha256(dedup_report),
    },
}
Path(output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps(payload, indent=2))
PY

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "[$(ts)] PREFLIGHT_ONLY complete; full ingest was not started"
  exit 0
fi

echo "[$(ts)] STAGE1 conservative FineWeb2-HQ ingest (direct JSONL)"
PYTHONPATH="$ROOT" python "$FILTER" \
  --input "$SRC" --input-format jsonl --text-field text \
  --source-profile fineweb2-hq --language german \
  --output "$WORK/de_filtered.txt"

echo "[$(ts)] STAGE2 tokenize with verified 200k SentencePiece"
cat > "$WORK/data_config.yaml" <<YAML
data_root: "$ROOT"
cleaned:
  german: ["$WORK/de_filtered.txt"]
YAML

cd "$ROOT"
python "$TOKENIZER_SCRIPT" \
  --data-config "$WORK/data_config.yaml" --tokenizer "$TOK" \
  --languages german --output-subdir "$OUTPUT_SUBDIR" --required-free-gb 100

echo "[$(ts)] ALL_DONE -> $ROOT/tokenized/$OUTPUT_SUBDIR/german.bin"
