#!/usr/bin/env python3
"""Run German cross-dataset dedup one reference at a time.

This orchestration keeps the retained set equivalent to querying one combined
reference index while bounding peak RAM to the largest individual reference.
Exact/near counts are attributed to the first matching reference in the
declared order, which is recorded in the cumulative report.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from scripts.data.dedup_de_fresh import (
    load_reference_manifest,
    main as dedup_main,
)


def report_path(output: Path) -> Path:
    return output.with_suffix(".dedup_report.json")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh", required=True, type=Path)
    parser.add_argument("--ref", required=True, nargs="+", type=Path)
    parser.add_argument("--ref-manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--keep-intermediate", action="store_true")
    parser.add_argument("--verify-ref-hashes", action="store_true")
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--num-perm", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--shingle-size", type=int, default=5)
    return parser.parse_args(argv)


def write_single_reference_manifest(
    path: Path,
    entry: dict[str, Any],
) -> None:
    payload = {
        "schema_version": 1,
        "hash_algorithm": "sha256",
        "references": [
            {
                "path": entry["manifest_path"],
                "size_bytes": entry["size_bytes"],
                "sha256": entry["sha256"],
            }
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    references: list[Path] = args.ref
    if args.fresh.resolve() == args.out.resolve():
        raise ValueError("--fresh and --out must be different files")

    full_manifest = load_reference_manifest(args.ref_manifest, references)
    work_dir = args.work_dir or args.out.parent / f".{args.out.stem}.dedup_stages"
    work_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    current_input = args.fresh
    intermediate_outputs: list[Path] = []
    stages: list[dict[str, Any]] = []

    for index, (reference, manifest_entry) in enumerate(
        zip(references, full_manifest["references"], strict=True),
        start=1,
    ):
        is_last = index == len(references)
        stage_output = args.out if is_last else work_dir / f"stage_{index:02d}.jsonl"
        stage_manifest = work_dir / f"reference_{index:02d}.manifest.json"
        write_single_reference_manifest(stage_manifest, manifest_entry)

        stage_argv = [
            "--fresh",
            str(current_input),
            "--ref",
            str(reference),
            "--ref-manifest",
            str(stage_manifest),
            "--out",
            str(stage_output),
            "--min-chars",
            str(args.min_chars),
            "--num-perm",
            str(args.num_perm),
            "--threshold",
            str(args.threshold),
            "--shingle-size",
            str(args.shingle_size),
        ]
        if args.verify_ref_hashes:
            stage_argv.append("--verify-ref-hashes")

        dedup_main(stage_argv)
        stage_report = json.loads(report_path(stage_output).read_text(encoding="utf-8"))
        stages.append(
            {
                "stage": index,
                "reference": str(reference),
                "reference_sha256": manifest_entry["sha256"],
                "input_seen": stage_report["fresh_seen"],
                "kept": stage_report["kept"],
                "dropped_exact": stage_report["dropped_exact"],
                "dropped_near": stage_report["dropped_near"],
                "invalid_json": stage_report.get("invalid_json", 0),
            }
        )

        if not is_last:
            intermediate_outputs.append(stage_output)
        current_input = stage_output

    first_stage = stages[0]
    final_stage = stages[-1]
    duplicate_drops = sum(
        stage["dropped_exact"] + stage["dropped_near"] for stage in stages
    )
    cumulative = {
        "schema_version": 1,
        "mode": "sequential_single_reference_passes",
        "classification_note": "exact/near is attributed to the first matching reference",
        "fresh_seen": first_stage["input_seen"],
        "kept": final_stage["kept"],
        "dropped_exact": sum(stage["dropped_exact"] for stage in stages),
        "dropped_near": sum(stage["dropped_near"] for stage in stages),
        "invalid_json": sum(stage["invalid_json"] for stage in stages),
        "drop_pct": round(100 * duplicate_drops / max(1, first_stage["input_seen"]), 3),
        "reference_manifest": {
            "path": str(args.ref_manifest),
            "sha256": full_manifest["sha256"],
        },
        "config": {
            "min_chars": args.min_chars,
            "num_perm": args.num_perm,
            "threshold": args.threshold,
            "shingle_size": args.shingle_size,
            "reference_order": [str(reference) for reference in references],
        },
        "stages": stages,
    }
    report_path(args.out).write_text(
        json.dumps(cumulative, indent=2) + "\n",
        encoding="utf-8",
    )

    if not args.keep_intermediate:
        for intermediate in intermediate_outputs:
            intermediate.unlink(missing_ok=True)

    print(f"[dedup-sequential] DONE {json.dumps(cumulative)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
