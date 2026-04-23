"""Automatic run-report writer.

Emits ``MANIFEST.yaml`` at run-start (config + metadata + backend summary)
and updates it at run-end with final metrics, exit reason, and alert list.
This is the single file you hand over when asking "what was this run?".
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml


def write_start_manifest(
    *,
    path: Path,
    config: dict[str, Any],
    metadata: Any,                               # RunMetadata dataclass
    backend_summary: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "status": "started",
        "metadata": asdict(metadata),
        "config": config,
        "backends": backend_summary or {},
    }
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
                    encoding="utf-8")


def write_end_manifest(
    *,
    path: Path,
    state: Any,                                  # TrainerState
    exit_reason: str,
    health_summary: dict[str, Any] | None = None,
) -> None:
    """Merge end-of-run fields into an existing start manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {}
    if path.is_file():
        try:
            existing = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            existing = {}
    existing.update({
        "status": "finished",
        "exit_reason": exit_reason,
        "final_state": asdict(state),
        "health": health_summary or {},
    })
    path.write_text(yaml.safe_dump(existing, sort_keys=False, allow_unicode=True),
                    encoding="utf-8")


__all__ = ["write_start_manifest", "write_end_manifest"]
