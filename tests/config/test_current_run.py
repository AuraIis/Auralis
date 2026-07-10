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


def test_current_run_cannot_silently_use_stale_server_tree_or_stub() -> None:
    run = load_current_run()

    assert run["code"]["server_checkout"]["required_mode"] == "clean_repo_checkout"
    assert run["code"]["server_checkout"]["forbidden_training_tree"] == "NEWGPT/v2data"
    assert run["data"]["forbidden_inputs"] == [
        {
            "path": "corpus20b/de_curated.bin",
            "observed_size_bytes": 25,
            "reason": "placeholder_stub_not_training_data",
        }
    ]


def test_current_run_keeps_unmeasured_provenance_as_launch_blockers() -> None:
    run = load_current_run()

    assert run["status"] == "draft_waiting_for_data_measurements"
    assert run["code"]["resolved_commit_sha"] is None
    assert run["tokenizer"]["full_sha256"] is None
    assert run["data"]["deduplication"]["reference_manifest"] is None
    assert run["fineweb2_hq_de_pilot"]["dedup_report"] is None
    assert "resolved_commit_sha" in run["launch_blockers"]


def test_current_run_records_required_blackwell_compatibility_override() -> None:
    run = load_current_run()

    assert run["runtime_environment"]["pytorch"] == "2.7.0+cu128"
    assert run["runtime_environment"]["required_environment"] == {
        "TRITON_OVERRIDE_ARCH": "sm89"
    }
