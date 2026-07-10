from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.data.dedup_de_sequential import main


def write_manifest(path: Path, references: list[Path]) -> None:
    payload = {
        "schema_version": 1,
        "hash_algorithm": "sha256",
        "references": [
            {
                "path": f"/host/references/{reference.name}",
                "size_bytes": reference.stat().st_size,
                "sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
            }
            for reference in references
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_sequential_dedup_aggregates_stages_and_bounds_intermediates(tmp_path: Path) -> None:
    reference_a = tmp_path / "reference_a.txt"
    reference_b = tmp_path / "reference_b.txt"
    text_a = "Die Photosynthese wandelt Lichtenergie in chemische Energie um."
    text_b = "Das Grundgesetz schützt die Würde und die Freiheit des Menschen."
    text_kept = "Küstenwinde entstehen durch Temperaturunterschiede zwischen Land und Meer."
    reference_a.write_text(text_a + "\n", encoding="utf-8")
    reference_b.write_text(text_b + "\n", encoding="utf-8")

    manifest = tmp_path / "references.json"
    write_manifest(manifest, [reference_a, reference_b])
    fresh = tmp_path / "fresh.jsonl"
    fresh.write_text(
        "\n".join(
            json.dumps({"text": text}) for text in [text_a, text_b, text_kept]
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "fresh.dedup.jsonl"
    work_dir = tmp_path / "stages"

    assert (
        main(
            [
                "--fresh",
                str(fresh),
                "--ref",
                str(reference_a),
                str(reference_b),
                "--ref-manifest",
                str(manifest),
                "--out",
                str(output),
                "--work-dir",
                str(work_dir),
                "--min-chars",
                "1",
                "--num-perm",
                "32",
            ]
        )
        == 0
    )

    retained = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert retained == [{"text": text_kept}]
    report = json.loads(output.with_suffix(".dedup_report.json").read_text(encoding="utf-8"))
    assert report["mode"] == "sequential_single_reference_passes"
    assert report["fresh_seen"] == 3
    assert report["kept"] == 1
    assert report["dropped_exact"] == 2
    assert report["dropped_near"] == 0
    assert [stage["input_seen"] for stage in report["stages"]] == [3, 2]
    assert not (work_dir / "stage_01.jsonl").exists()


def test_sequential_dedup_rejects_in_place_output(tmp_path: Path) -> None:
    reference = tmp_path / "reference.txt"
    reference.write_text("Eine ausreichend lange Referenzzeile.\n", encoding="utf-8")
    manifest = tmp_path / "references.json"
    write_manifest(manifest, [reference])
    fresh = tmp_path / "fresh.jsonl"
    fresh.write_text('{"text": "Ein Dokument"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="must be different"):
        main(
            [
                "--fresh",
                str(fresh),
                "--ref",
                str(reference),
                "--ref-manifest",
                str(manifest),
                "--out",
                str(fresh),
            ]
        )
