"""Push-only file sync from local repo to a remote training host.

Watches a whitelist of source directories and pushes any change via rsync
over SSH on every tick. Optionally also copies into a running container
(docker cp) for setups where the repo is *not* bind-mounted.

This solves the recurring "I edited a config locally but it's not on the
trainings-host" problem without requiring a Bind-Mount restart of the
container. See configs/dev/auto_sync.yaml for the runtime knobs.

Usage:
    python scripts/dev/auto_sync.py
    python scripts/dev/auto_sync.py --once         # one full sync, then exit
    python scripts/dev/auto_sync.py --dry-run      # show what would sync
    python scripts/dev/auto_sync.py --config path  # use alt config

Exit codes:
    0  clean shutdown (Ctrl+C)
    1  configuration error or fatal sync failure
    2  forbidden path detected in sync set (safety abort)
"""

from __future__ import annotations

import argparse
import logging
import logging.handlers
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO / "configs" / "dev" / "auto_sync.yaml"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class Config:
    ssh_host: str
    ssh_user: str
    ssh_port: int
    remote_path: str
    sync_paths: list[str]
    sync_files: list[str]
    excludes: list[str]
    forbidden: set[str]
    container_mode: str
    container_name: str
    container_workdir: str
    poll_interval: float
    debounce_ms: int
    log_file: Path
    log_max_mb: int
    log_backups: int
    console_level: str

    @classmethod
    def load(cls, path: Path) -> Config:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        ssh = raw["ssh"]
        cont = raw["container"]
        watcher = raw["watcher"]
        log = raw["logging"]
        return cls(
            ssh_host=str(ssh["host"]).strip(),
            ssh_user=str(ssh["user"]).strip(),
            ssh_port=int(ssh.get("port", 22)),
            remote_path=str(ssh["remote_path"]).rstrip("/"),
            sync_paths=list(raw.get("sync_paths", [])),
            sync_files=list(raw.get("sync_files", [])),
            excludes=list(raw.get("excludes", [])),
            forbidden=set(raw.get("forbidden", [])),
            container_mode=str(cont.get("mode", "off")).strip().lower(),
            container_name=str(cont.get("name", "")).strip(),
            container_workdir=str(cont.get("workdir", "")).rstrip("/"),
            poll_interval=float(watcher.get("poll_interval_sec", 2.0)),
            debounce_ms=int(watcher.get("debounce_ms", 500)),
            log_file=REPO / str(log.get("log_file", "logs/auto_sync.log")),
            log_max_mb=int(log.get("max_mb", 10)),
            log_backups=int(log.get("backups", 3)),
            console_level=str(log.get("console_level", "INFO")).upper(),
        )

    def validate(self, log: logging.Logger) -> None:
        if not self.ssh_host:
            raise SystemExit("config error: ssh.host is empty")
        if not self.remote_path.startswith("/"):
            raise SystemExit("config error: ssh.remote_path must be absolute")
        if self.container_mode not in {"docker_cp", "bind_mount", "off"}:
            raise SystemExit(f"config error: container.mode={self.container_mode!r} invalid")
        if self.container_mode == "docker_cp" and not self.container_name:
            raise SystemExit("config error: container.name required for docker_cp mode")

        # Forbidden-path check: refuse to start if a sync path is in the
        # forbidden list. Cheap insurance against an accidental commit that
        # adds e.g. "data" to sync_paths.
        for p in self.sync_paths:
            top = Path(p).parts[0] if Path(p).parts else p
            if top in self.forbidden:
                log.error("FATAL: sync_paths contains forbidden top-level dir %r", top)
                raise SystemExit(2)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logging(cfg: Config) -> logging.Logger:
    cfg.log_file.parent.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("auto_sync")
    log.setLevel(logging.DEBUG)
    log.handlers.clear()

    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S")
    file_handler = logging.handlers.RotatingFileHandler(
        cfg.log_file,
        maxBytes=cfg.log_max_mb * 1024 * 1024,
        backupCount=cfg.log_backups,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    log.addHandler(file_handler)

    console = logging.StreamHandler()
    console.setLevel(getattr(logging, cfg.console_level, logging.INFO))
    console.setFormatter(fmt)
    log.addHandler(console)
    return log


# ---------------------------------------------------------------------------
# Filesystem snapshot
# ---------------------------------------------------------------------------


def _matches_exclude(rel: str, patterns: list[str]) -> bool:
    """Lightweight pattern matcher that handles dir/ and *.ext globs."""
    from fnmatch import fnmatch

    rel_norm = rel.replace("\\", "/")
    for pat in patterns:
        if pat.endswith("/"):
            needle = pat.rstrip("/")
            if needle in rel_norm.split("/"):
                return True
        elif fnmatch(rel_norm, pat) or fnmatch(Path(rel_norm).name, pat):
            return True
    return False


def snapshot(cfg: Config) -> dict[str, float]:
    """Walk all sync_paths + sync_files and return {rel_path: mtime}."""
    out: dict[str, float] = {}
    for top in cfg.sync_paths:
        root = REPO / top
        if not root.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            # Prune excluded dirs in-place so os.walk doesn't descend into them.
            keep = []
            for d in dirnames:
                rel = str(Path(dirpath, d).relative_to(REPO))
                if _matches_exclude(rel, cfg.excludes):
                    continue
                keep.append(d)
            dirnames[:] = keep
            for fname in filenames:
                full = Path(dirpath) / fname
                rel = str(full.relative_to(REPO)).replace("\\", "/")
                if _matches_exclude(rel, cfg.excludes):
                    continue
                try:
                    out[rel] = full.stat().st_mtime
                except OSError:
                    pass
    for f in cfg.sync_files:
        full = REPO / f
        if full.is_file():
            out[f.replace("\\", "/")] = full.stat().st_mtime
    return out


def diff_snapshots(prev: dict[str, float], curr: dict[str, float]) -> tuple[set[str], set[str]]:
    """Return (changed_or_new, deleted) between two snapshots."""
    changed: set[str] = set()
    for path, mtime in curr.items():
        if path not in prev or prev[path] != mtime:
            changed.add(path)
    deleted = set(prev) - set(curr)
    return changed, deleted


# ---------------------------------------------------------------------------
# Sync operations
# ---------------------------------------------------------------------------


def _ssh_dest(cfg: Config) -> str:
    return f"{cfg.ssh_user}@{cfg.ssh_host}:{cfg.remote_path}/"


def _ssh_cmd(cfg: Config) -> list[str]:
    return ["ssh", "-p", str(cfg.ssh_port), "-o", "BatchMode=yes"]


def ensure_remote_root(cfg: Config, log: logging.Logger) -> None:
    """Make sure the remote target directory exists."""
    cmd = [
        *_ssh_cmd(cfg),
        f"{cfg.ssh_user}@{cfg.ssh_host}",
        f"mkdir -p {cfg.remote_path!s}",
    ]
    log.debug("ensure remote root: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def rsync_files(
    cfg: Config,
    files: list[str],
    log: logging.Logger,
    *,
    dry_run: bool = False,
) -> None:
    """Push the given relative paths to the remote host in one rsync call."""
    if not files:
        return
    files_from = REPO / "logs" / ".auto_sync_files_from.txt"
    files_from.parent.mkdir(parents=True, exist_ok=True)
    files_from.write_text("\n".join(sorted(files)) + "\n", encoding="utf-8")

    rsh = f"ssh -p {cfg.ssh_port} -o BatchMode=yes"
    cmd = [
        "rsync",
        "-az",
        "--relative",
        "--files-from",
        str(files_from),
        "-e",
        rsh,
    ]
    if dry_run:
        cmd.append("--dry-run")
    cmd.extend([str(REPO) + "/", _ssh_dest(cfg)])

    log.info("rsync push: %d file(s)%s", len(files), " [dry-run]" if dry_run else "")
    log.debug("cmd: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        log.error("rsync failed (exit=%d): %s", proc.returncode, proc.stderr.strip())
        raise RuntimeError("rsync failure")
    if proc.stdout.strip():
        log.debug("rsync stdout:\n%s", proc.stdout.strip())


def rsync_deletes(
    cfg: Config,
    deleted: list[str],
    log: logging.Logger,
    *,
    dry_run: bool = False,
) -> None:
    """Mirror local deletions onto the remote host."""
    if not deleted:
        return
    log.info("delete remote: %d file(s)%s", len(deleted), " [dry-run]" if dry_run else "")
    quoted = " ".join(f"{cfg.remote_path}/{d}" for d in deleted)
    rm_cmd = f"rm -f {quoted}"
    cmd = [*_ssh_cmd(cfg), f"{cfg.ssh_user}@{cfg.ssh_host}", rm_cmd]
    if dry_run:
        log.debug("would run: %s", rm_cmd)
        return
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        log.warning("remote rm exit=%d stderr=%s", proc.returncode, proc.stderr.strip())


def docker_cp_files(
    cfg: Config,
    files: list[str],
    log: logging.Logger,
    *,
    dry_run: bool = False,
) -> None:
    """After pushing to host, push the same files into the container.

    Uses one tar stream per call: we tar the requested paths on the remote
    host and pipe straight into `docker exec ... tar -x`. That way one ssh
    roundtrip handles arbitrarily many files.
    """
    if cfg.container_mode != "docker_cp" or not files:
        return
    log.info(
        "container sync: %d file(s) → %s%s",
        len(files),
        cfg.container_name,
        " [dry-run]" if dry_run else "",
    )
    paths = " ".join(files)
    remote_script = (
        f"cd {cfg.remote_path} && "
        f"tar -cf - {paths} | "
        f"docker exec -i {cfg.container_name} tar -C {cfg.container_workdir} -xf -"
    )
    cmd = [*_ssh_cmd(cfg), f"{cfg.ssh_user}@{cfg.ssh_host}", remote_script]
    if dry_run:
        log.debug("would run on remote: %s", remote_script)
        return
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        log.error("docker cp failed (exit=%d): %s", proc.returncode, proc.stderr.strip())
        raise RuntimeError("docker cp failure")


def docker_rm_files(
    cfg: Config,
    deleted: list[str],
    log: logging.Logger,
    *,
    dry_run: bool = False,
) -> None:
    """Mirror deletions inside the container."""
    if cfg.container_mode != "docker_cp" or not deleted:
        return
    quoted = " ".join(f"{cfg.container_workdir}/{d}" for d in deleted)
    rm_cmd = f"docker exec {cfg.container_name} rm -f {quoted}"
    cmd = [*_ssh_cmd(cfg), f"{cfg.ssh_user}@{cfg.ssh_host}", rm_cmd]
    if dry_run:
        log.debug("would run on remote: %s", rm_cmd)
        return
    subprocess.run(cmd, capture_output=True, text=True)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


_stop_requested = False


def _install_signal_handlers(log: logging.Logger) -> None:
    def _handler(signum: int, _frame: Any) -> None:
        global _stop_requested
        log.info("signal %s — shutting down", signum)
        _stop_requested = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):
            # SIGTERM may be unavailable on Windows in some shells.
            pass


def _check_prerequisites(log: logging.Logger) -> None:
    missing = [tool for tool in ("rsync", "ssh") if shutil.which(tool) is None]
    if missing:
        log.error("missing required tool(s): %s", ", ".join(missing))
        log.error("on Windows: install via Git Bash, MSYS2, or cwRsync and ensure they're on PATH")
        raise SystemExit(1)


def run_once(cfg: Config, log: logging.Logger, *, dry_run: bool) -> None:
    """Take a snapshot and push every file in it (full initial sync)."""
    snap = snapshot(cfg)
    files = sorted(snap.keys())
    log.info("initial sync: %d files", len(files))
    rsync_files(cfg, files, log, dry_run=dry_run)
    docker_cp_files(cfg, files, log, dry_run=dry_run)


def run_loop(cfg: Config, log: logging.Logger, *, dry_run: bool) -> None:
    """Continuous polling loop with debounced batched sync."""
    log.info("starting watcher (poll=%.1fs, debounce=%dms)", cfg.poll_interval, cfg.debounce_ms)
    prev = snapshot(cfg)
    log.info("initial snapshot: %d files tracked", len(prev))
    pending_changed: set[str] = set()
    pending_deleted: set[str] = set()
    last_change_at: float | None = None

    while not _stop_requested:
        time.sleep(cfg.poll_interval)
        curr = snapshot(cfg)
        changed, deleted = diff_snapshots(prev, curr)
        if changed or deleted:
            for c in changed:
                log.debug("changed: %s", c)
            for d in deleted:
                log.debug("deleted: %s", d)
            pending_changed |= changed
            pending_deleted |= deleted
            last_change_at = time.monotonic()
            prev = curr

        # Debounce: only flush when no new changes for `debounce_ms`.
        if last_change_at is not None:
            quiet_for = (time.monotonic() - last_change_at) * 1000
            if quiet_for >= cfg.debounce_ms:
                try:
                    rsync_files(cfg, sorted(pending_changed), log, dry_run=dry_run)
                    docker_cp_files(cfg, sorted(pending_changed), log, dry_run=dry_run)
                    rsync_deletes(cfg, sorted(pending_deleted), log, dry_run=dry_run)
                    docker_rm_files(cfg, sorted(pending_deleted), log, dry_run=dry_run)
                except Exception as exc:
                    log.exception("sync batch failed: %s", exc)
                pending_changed.clear()
                pending_deleted.clear()
                last_change_at = None

    log.info("clean shutdown")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument(
        "--once",
        action="store_true",
        help="Do a single full sync of everything in the whitelist, then exit.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be synced/deleted without touching the remote.",
    )
    args = ap.parse_args()

    if not args.config.is_file():
        print(f"config not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    cfg = Config.load(args.config)
    log = setup_logging(cfg)
    cfg.validate(log)
    log.info("config loaded: %s", args.config)
    log.info(
        "target: %s@%s:%s (container=%s)",
        cfg.ssh_user,
        cfg.ssh_host,
        cfg.remote_path,
        cfg.container_mode,
    )

    _check_prerequisites(log)
    _install_signal_handlers(log)

    try:
        ensure_remote_root(cfg, log)
    except subprocess.CalledProcessError as exc:
        log.error("cannot reach remote: %s", exc.stderr or exc)
        sys.exit(1)

    if args.once:
        run_once(cfg, log, dry_run=args.dry_run)
    else:
        # Initial full push so server state matches local at startup.
        run_once(cfg, log, dry_run=args.dry_run)
        run_loop(cfg, log, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
