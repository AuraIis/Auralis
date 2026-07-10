import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "configs" / "runs" / "manifests" / "dedup_reference_sha256.json"


def test_dedup_reference_manifest_is_complete_and_ordered() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == 1
    assert manifest["hash_algorithm"] == "sha256"
    references = manifest["references"]
    assert [entry["path"] for entry in references] == [
        "cleaned/fineweb2_de.filtered.txt",
        "training/curated_40b/german.txt",
        "cleaned/german_commons.filtered.txt",
        "raw/german/german_commons.txt",
    ]
    assert sum(entry["documents"] for entry in references) == 30441496
    assert all(len(entry["sha256"]) == 64 for entry in references)
    assert all(entry["size_bytes"] > 0 for entry in references)
