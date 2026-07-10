from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.data.dedup_de_fresh import load_reference_manifest, main


def write_reference_manifest(path: Path, reference: Path) -> None:
    payload = {
        "schema_version": 1,
        "hash_algorithm": "sha256",
        "references": [
            {
                "path": f"/host/reference/{reference.name}",
                "size_bytes": reference.stat().st_size,
                "sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_reference_manifest_allows_host_container_path_drift(tmp_path: Path) -> None:
    reference = tmp_path / "german_reference.txt"
    reference.write_text("Eine Referenzzeile.\n", encoding="utf-8")
    manifest_path = tmp_path / "references.json"
    write_reference_manifest(manifest_path, reference)

    manifest = load_reference_manifest(manifest_path, [reference])

    assert manifest["references"][0]["path"] == str(reference)
    assert manifest["references"][0]["manifest_path"].endswith(reference.name)
    assert len(manifest["sha256"]) == 64


def test_load_reference_manifest_rejects_size_mismatch(tmp_path: Path) -> None:
    reference = tmp_path / "german_reference.txt"
    reference.write_text("Eine Referenzzeile.\n", encoding="utf-8")
    manifest_path = tmp_path / "references.json"
    write_reference_manifest(manifest_path, reference)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["references"][0]["size_bytes"] += 1
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="size mismatch"):
        load_reference_manifest(manifest_path, [reference])


def test_main_preserves_existing_report_fields_and_records_provenance(tmp_path: Path) -> None:
    reference_text = "Die Photosynthese wandelt Lichtenergie in chemische Energie um."
    unique_text = "Dieser eigenständige Text beschreibt die Entstehung von Küstenwinden."
    reference = tmp_path / "german_reference.txt"
    reference.write_text(reference_text + "\n", encoding="utf-8")
    manifest_path = tmp_path / "references.json"
    write_reference_manifest(manifest_path, reference)

    fresh = tmp_path / "fresh.jsonl"
    fresh.write_text(
        "\n".join(
            [
                json.dumps({"text": reference_text}),
                json.dumps({"text": unique_text}),
                "not-json",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "fresh.dedup.jsonl"

    assert (
        main(
            [
                "--fresh",
                str(fresh),
                "--ref",
                str(reference),
                "--ref-manifest",
                str(manifest_path),
                "--out",
                str(output),
                "--min-chars",
                "1",
                "--num-perm",
                "32",
            ]
        )
        == 0
    )

    retained = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert retained == [{"text": unique_text}]

    report = json.loads(
        output.with_suffix(".dedup_report.json").read_text(encoding="utf-8")
    )
    assert report["fresh_seen"] == 3
    assert report["kept"] == 1
    assert report["dropped_exact"] == 1
    assert report["dropped_near"] == 0
    assert report["invalid_json"] == 1
    assert report["config"]["threshold"] == 0.85
    assert report["config"]["cross_dataset_only"] is True
    assert report["inputs"]["reference_manifest"]["references"][0]["verified"] is False
