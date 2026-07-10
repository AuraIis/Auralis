from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CURRENT_RUN = ROOT / "configs" / "runs" / "current_run.yaml"


def load_current_run() -> dict:
    import yaml

    return yaml.safe_load(CURRENT_RUN.read_text(encoding="utf-8"))


def test_current_run_mix_is_complete_and_german_primary() -> None:
    run = load_current_run()
    mix = run["training"]["mix_percent"]

    assert sum(mix.values()) == 100
    assert mix == {"german": 65, "english": 15, "code": 12, "math_science": 8}


def test_current_run_requires_clean_code_and_records_the_valid_data_symlink() -> None:
    run = load_current_run()

    assert run["code"]["server_checkout"]["required_mode"] == "clean_repo_checkout"
    assert run["code"]["server_checkout"]["forbidden_training_tree"] == "NEWGPT/v2data"
    curated = next(
        component for component in run["data"]["german_components"] if component["id"] == "de_curated"
    )
    assert curated["bin_aliases"] == [
        {
            "path": "/workspace/v2data/tokenized/corpus20b/de_curated.bin",
            "type": "symlink",
            "resolves_to": "/workspace/v2data/tokenized/curated_40b/german.bin",
            "status": "valid",
        }
    ]


def test_current_run_keeps_unmeasured_provenance_as_launch_blockers() -> None:
    run = load_current_run()

    assert run["status"] == "draft_waiting_for_lineage_hashes_and_resume"
    assert run["code"]["resolved_commit_sha"] is None
    assert run["base_checkpoint"]["manifest_sha256"] is None
    assert all(component["bin_sha256"] is None for component in run["data"]["german_components"])
    assert run["data"]["deduplication"]["reference_manifest"] is None
    assert run["fineweb2_hq_de_pilot"]["full_dedup_report"] is None
    assert "code.resolved_commit_sha" in run["launch_blockers"]


def test_current_run_records_required_blackwell_compatibility_override() -> None:
    run = load_current_run()

    assert run["runtime_environment"]["pytorch"] == "2.7.0+cu128"
    assert run["runtime_environment"]["required_environment"] == {
        "TRITON_OVERRIDE_ARCH": "sm89"
    }



def test_current_run_records_exact_tokenizer_and_german_inventory() -> None:
    run = load_current_run()

    assert run["tokenizer"] == {
        "immutable_for_run": True,
        "path": "tokenizer/helix_v2_tokenizer.model",
        "size_bytes": 3592329,
        "vocab_size": 200000,
        "measured_hash": "a24fbea439bc8b78",
        "measured_hash_scope": "sha256_prefix_16",
        "full_sha256": "a24fbea439bc8b78c78653b9febf708d96cf023745199d8f6e7c0b3f6285f2bc",
    }
    components = run["data"]["german_components"]
    assert sum(component["bin_tokens"] for component in components) == 7461038089
    assert run["data"]["german_measured_tokens"] == 7461038089


def test_current_run_records_probe_without_promoting_it_to_full_evidence() -> None:
    run = load_current_run()
    pilot = run["fineweb2_hq_de_pilot"]

    assert pilot["status"] == "download_approved_dedup_blocked_on_lineage"
    assert pilot["probe"]["fresh_seen"] == 127590
    assert pilot["probe"]["kept"] == 70034
    assert pilot["probe"]["drop_pct"] == 45.11
    assert pilot["probe"]["limitation"] == "legacy_report_one_shard_one_reference"
    assert pilot["full_dedup_report"] is None



def test_current_run_does_not_claim_unproven_training_sources() -> None:
    run = load_current_run()
    components = {component["id"]: component for component in run["data"]["german_components"]}

    fresh = components["german_fresh"]
    assert fresh["source_path"] is None
    assert fresh["lineage_status"] == "unresolved"
    assert fresh["bin_documents_from_index"] == 4763746
    assert fresh["lineage_evidence"]["final_tokenize_log"]["documents"] == 7412611

    commons = components["german_commons"]
    assert commons["source_path"] is None
    assert commons["lineage_status"] == "unresolved_pending_build_log"
    assert "data.german_components.source_lineage" in run["launch_blockers"]
