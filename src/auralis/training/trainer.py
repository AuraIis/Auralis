"""PretrainTrainer: minimal but complete pretraining loop.

Scope: gradient accumulation, cosine LR schedule, grad clipping, checkpoints
with rotation, NaN detection, periodic eval. No distributed-training logic —
that is layered on via FSDP / DeepSpeed in ``scripts/pretrain/train_phase1.py``
which is what actually runs on RunPod.

All stateful configuration passes through a plain dict (from YAML) rather than
a dataclass, so future additions (e.g. MoE loss aux, MTP loss) don't need
signature changes here — only dict-key lookups.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from auralis.training.health import (
    AlertLevel,
    HealthConfig,
    HealthMonitor,
    HealthStop,
)


_AMP_DTYPES: dict[str, torch.dtype] = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
}


def _safe_log(logger: Callable[[dict[str, float], int], None]):
    """Wrap a metrics logger so a logging error never kills training."""
    def inner(metrics: dict[str, float], step: int) -> None:
        try:
            logger(metrics, step)
        except Exception as e:                                 # noqa: BLE001
            # Very noisy to print every failure; emit the type once per call.
            print(f"  warn: metrics logger failed: {type(e).__name__}: {e}", flush=True)
    return inner


def _git_sha(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _sha256_short(path: Path) -> str:
    if not path or not Path(path).is_file():
        return ""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


@dataclass
class TrainerState:
    """Pure-data training state — persisted in every checkpoint."""

    step: int = 0
    best_val_loss: float = float("inf")
    consecutive_val_increases: int = 0
    tokens_seen: int = 0
    wall_clock_seconds: float = 0.0
    alerts: list[str] = field(default_factory=list)
    # Cheap counters for post-hoc analysis of backups + logging reliability
    external_backups_ok: int = 0
    external_backups_failed: int = 0


@dataclass
class RunMetadata:
    """Run-level provenance captured once at Trainer construction.

    Written into every checkpoint alongside TrainerState so a reloaded
    checkpoint is self-explanatory ("which git rev, which config, which
    tokenizer file, which machine produced this").
    """

    git_sha: str = "unknown"
    config_sha16: str = ""
    config_path: str = ""
    tokenizer_sha16: str = ""
    tokenizer_path: str = ""
    hostname: str = ""
    python_version: str = ""
    torch_version: str = ""
    cuda_version: str | None = None
    gpu_name: str | None = None
    dtype: str = "fp32"


class PretrainTrainer:
    def __init__(
        self,
        *,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LambdaLR,
        dataloader: Iterator[dict[str, torch.Tensor]],
        config: dict[str, Any],
        state: TrainerState | None = None,
        device: str | torch.device = "cpu",
        val_dataloader: Iterator[dict[str, torch.Tensor]] | None = None,
        wandb_logger: Callable[[dict[str, float], int], None] | None = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.state = state or TrainerState()
        self.device = torch.device(device)
        self.log = _safe_log(wandb_logger or (lambda _metrics, _step: None))

        # Unpack knobs that get hit every step.
        tcfg = config["training"]
        self._total_steps = int(tcfg["total_steps"])
        self._grad_accum = int(tcfg.get("gradient_accumulation", 1))
        self._clip_norm = float(tcfg.get("gradient_clip_norm", 1.0))
        self._batch_tokens = int(
            tcfg["batch_size_per_device"] * self._grad_accum * config["data"]["seq_length"]
        )

        # AMP dtype — config.training.dtype drives the forward-pass autocast.
        # We do NOT silently fall back: picking fp16 on CPU is almost always a
        # config mistake that would later manifest as NaNs at unknown cost.
        dtype_str = str(tcfg.get("dtype", "fp32")).lower()
        if dtype_str not in _AMP_DTYPES:
            raise ValueError(f"unknown dtype {dtype_str!r}; one of {sorted(_AMP_DTYPES)}")
        self._amp_dtype = _AMP_DTYPES[dtype_str]

        # fp16 requires a GradScaler to stay numerically stable; bf16 and fp32
        # do not. Create the scaler only when fp16 on CUDA is actually active.
        self._use_amp = self._amp_dtype != torch.float32
        self._scaler: torch.cuda.amp.GradScaler | None = None
        if self._amp_dtype == torch.float16:
            if self.device.type != "cuda":
                raise ValueError("fp16 training requires a CUDA device.")
            self._scaler = torch.cuda.amp.GradScaler()

        lcfg = config["logging"]
        self._log_every = int(lcfg.get("log_every", 10))
        self._eval_every = int(lcfg.get("eval_every", 1000))
        self._save_every = int(lcfg.get("save_every", 2500))

        ccfg = config["checkpointing"]
        self._ckpt_dir = Path(ccfg["output_dir"])
        self._ckpt_dir.mkdir(parents=True, exist_ok=True)
        self._keep_last = int(ccfg.get("save_last_n", 3))
        self._external_backup = ccfg.get("external_backup") or {}

        # ---- Health monitor (auto-stop guards) ----
        mon_cfg = (config.get("monitoring") or {}).get("health") or {}
        self.health = HealthMonitor(HealthConfig(
            **{k: v for k, v in mon_cfg.items() if k in HealthConfig.__dataclass_fields__}
        ))

        # ---- Cost tracker ----
        cost_cfg = config.get("cost") or {}
        self._cost_per_gpu_hour = float(cost_cfg.get("gpu_hourly_usd", 0.0))
        self._cost_budget = float(cost_cfg.get("budget_usd", 0.0))

        # ---- Run metadata (captured once, persisted with every ckpt) ----
        repo_root = Path(__file__).resolve().parents[3]
        self.metadata = RunMetadata(
            git_sha=str(_git_sha(repo_root)),
            config_sha16=hashlib.sha256(json.dumps(config, sort_keys=True, default=str)
                                        .encode("utf-8")).hexdigest()[:16],
            config_path=str(config.get("_source_path", "")),
            tokenizer_sha16=str(_sha256_short(Path(
                config.get("data", {}).get("tokenizer_path")
                or repo_root / "tokenizer" / "helix_v2_tokenizer.model"
            ))),
            tokenizer_path=str(config.get("data", {}).get("tokenizer_path")
                               or repo_root / "tokenizer" / "helix_v2_tokenizer.model"),
            hostname=str(socket.gethostname()),
            python_version=str(platform.python_version()),
            # torch.__version__ is a TorchVersion object; str() makes it yaml-safe.
            torch_version=str(torch.__version__),
            cuda_version=(str(torch.version.cuda) if torch.cuda.is_available() else None),
            gpu_name=(str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else None),
            dtype=str(dtype_str),
        )

    def _autocast(self):
        """Context manager for mixed-precision forward/backward."""
        if not self._use_amp:
            return nullcontext()
        # torch.autocast is device-type specific.
        return torch.autocast(device_type=self.device.type, dtype=self._amp_dtype)

    # ------------------------------------------------------------------
    def train(self) -> TrainerState:
        self.model.train()
        data_iter = iter(self.dataloader)
        window_loss_sum = 0.0
        window_t0 = time.time()
        window_data_time = 0.0
        window_compute_time = 0.0

        while self.state.step < self._total_steps:
            loss_acc = 0.0
            self.optimizer.zero_grad(set_to_none=True)

            for _ in range(self._grad_accum):
                t_data_0 = time.time()
                batch = next(data_iter)
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                window_data_time += time.time() - t_data_0

                t_compute_0 = time.time()
                with self._autocast():
                    out = self.model(input_ids=batch["input_ids"], labels=batch["labels"])
                    loss = out["loss"] / self._grad_accum
                if not torch.isfinite(loss):
                    msg = f"non-finite loss at step {self.state.step}: {loss.item()}"
                    self.state.alerts.append(msg)
                    raise RuntimeError(msg)
                if self._scaler is not None:
                    self._scaler.scale(loss).backward()
                else:
                    loss.backward()
                loss_acc += loss.item()
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                window_compute_time += time.time() - t_compute_0

            # Clip + step (unscale first if fp16 + GradScaler)
            if self._scaler is not None:
                self._scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self._clip_norm
            )
            if self._scaler is not None:
                self._scaler.step(self.optimizer)
                self._scaler.update()
            else:
                self.optimizer.step()
            self.scheduler.step()

            self.state.step += 1
            self.state.tokens_seen += self._batch_tokens
            window_loss_sum += loss_acc

            if self.state.step % self._log_every == 0:
                elapsed = time.time() - window_t0
                self.state.wall_clock_seconds += elapsed
                avg_loss = window_loss_sum / self._log_every
                tps = (self._batch_tokens * self._log_every) / max(elapsed, 1e-9)
                lr = self.scheduler.get_last_lr()[0]
                metrics: dict[str, float] = {
                    "train/loss": avg_loss,
                    "train/grad_norm": float(grad_norm),
                    "train/lr": lr,
                    "train/tokens_per_second": tps,
                    "train/tokens_seen": self.state.tokens_seen,
                    "train/data_frac": window_data_time / max(elapsed, 1e-9),
                    "train/compute_frac": window_compute_time / max(elapsed, 1e-9),
                }
                metrics["system/step_time_ms"] = (elapsed / max(self._log_every, 1)) * 1000.0
                if self.device.type == "cuda":
                    metrics["train/vram_alloc_gb"] = torch.cuda.memory_allocated() / 1e9
                    metrics["train/vram_reserved_gb"] = torch.cuda.memory_reserved() / 1e9
                    metrics["train/vram_peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
                    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                    metrics["train/vram_total_gb"] = total_vram_gb
                    # VRAM-pressure alert (may request STOP)
                    for level, msg in self.health.observe_vram(
                        metrics["train/vram_alloc_gb"], total_vram_gb, self.state.step,
                    ):
                        print(f"  health[{level.value}]: {msg}", flush=True)

                # Cost tracking ($/step, projected total, ETA vs budget)
                if self._cost_per_gpu_hour > 0:
                    hours = self.state.wall_clock_seconds / 3600.0
                    spent = hours * self._cost_per_gpu_hour
                    steps_per_hour = self.state.step / max(hours, 1e-9)
                    eta_hours = max(0, (self._total_steps - self.state.step)) / max(steps_per_hour, 1e-9)
                    metrics["cost/usd_spent"] = spent
                    metrics["cost/usd_projected_total"] = spent + eta_hours * self._cost_per_gpu_hour
                    metrics["cost/usd_per_1k_steps"] = (spent / max(self.state.step, 1)) * 1000
                    metrics["cost/usd_per_1b_tokens"] = (spent / max(self.state.tokens_seen, 1)) * 1e9
                    metrics["cost/eta_hours"] = eta_hours
                    if self._cost_budget > 0 and metrics["cost/usd_projected_total"] > self._cost_budget:
                        # Not a hard stop by default — alert level. Raise to STOP
                        # by setting monitoring.health.grad_explosion_threshold=0 etc.
                        self.state.alerts.append(
                            f"projected cost ${metrics['cost/usd_projected_total']:.0f} "
                            f"> budget ${self._cost_budget:.0f}"
                        )

                self.log(metrics, self.state.step)

                # Health check — may set self.health.stop_requested
                for level, msg in self.health.observe(metrics, self.state.step):
                    print(f"  health[{level.value}]: {msg}", flush=True)
                extra = ""
                if self.device.type == "cuda":
                    extra = f" | vram {metrics['train/vram_alloc_gb']:.1f}/{metrics['train/vram_peak_gb']:.1f}GB"
                print(
                    f"step {self.state.step:6d} | loss {avg_loss:6.4f} | "
                    f"lr {lr:.2e} | grad_norm {float(grad_norm):5.2f} | "
                    f"tok/s {tps/1e3:6.1f}k | "
                    f"data {metrics['train/data_frac']*100:4.1f}%"
                    f"{extra}",
                    flush=True,
                )
                window_loss_sum = 0.0
                window_t0 = time.time()
                window_data_time = 0.0
                window_compute_time = 0.0

            if self.val_dataloader is not None and self.state.step % self._eval_every == 0:
                val_loss = self._evaluate()
                self._track_val(val_loss)
                # Health may also STOP on sustained val regression.
                for level, msg in self.health.observe_val(
                    val_loss,
                    self.state.best_val_loss,
                    self.state.consecutive_val_increases,
                    self.state.step,
                ):
                    print(f"  health[{level.value}]: {msg}", flush=True)

            if self.state.step % self._save_every == 0:
                self.save_checkpoint(f"step_{self.state.step}")

            if self.health.should_stop():
                reason = self.health.state.stop_reason
                print(f"  AUTO-STOP: {reason}. Saving emergency ckpt and exiting.", flush=True)
                self.save_checkpoint(f"step_{self.state.step}_emergency")
                raise HealthStop(reason)

        # Final checkpoint
        self.save_checkpoint(f"step_{self.state.step}")
        return self.state

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _evaluate(self) -> float:
        """Compute val_loss overall + per-language (pretrain-mix diagnostic).

        Per-language slices are drawn via ``val_dataloader.sample_language``
        if that method exists (our MixedDataLoader). Falls back to only the
        mixed loss for generic iterables.
        """
        self.model.eval()
        assert self.val_dataloader is not None
        eval_cfg = self.config.get("evaluation", {})
        max_batches = int(eval_cfg.get("max_val_batches", 50))
        per_lang_batches = int(eval_cfg.get("per_language_batches", 8))

        metrics: dict[str, float] = {}

        # Overall mixed-batch val loss
        losses: list[float] = []
        it = iter(self.val_dataloader)
        for _ in range(max_batches):
            try:
                batch = next(it)
            except StopIteration:
                break
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            with self._autocast():
                out = self.model(input_ids=batch["input_ids"], labels=batch["labels"])
            losses.append(out["loss"].item())
        val = sum(losses) / max(1, len(losses))
        metrics["eval/val_loss"] = val

        # Per-language slices
        sample_lang = getattr(self.val_dataloader, "sample_language", None)
        batch_size = int(self.config.get("training", {}).get("batch_size_per_device", 4))
        if callable(sample_lang) and per_lang_batches > 0:
            lang_losses: dict[str, list[float]] = {}
            for lang in getattr(self.val_dataloader, "mix_ratios", {}):
                if self.val_dataloader.mix_ratios.get(lang, 0) <= 0:
                    continue  # skip languages that are weighted out
                lang_losses[lang] = []
                for _ in range(per_lang_batches):
                    try:
                        batch = sample_lang(lang, batch_size)
                    except (KeyError, ValueError):
                        break
                    batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                    with self._autocast():
                        out = self.model(input_ids=batch["input_ids"], labels=batch["labels"])
                    lang_losses[lang].append(out["loss"].item())
            for lang, ls in lang_losses.items():
                if ls:
                    metrics[f"eval/val_loss/{lang}"] = sum(ls) / len(ls)

        self.model.train()
        self.log(metrics, self.state.step)
        pretty = " ".join(f"{k.split('/')[-1]}={v:.3f}" for k, v in metrics.items())
        print(f"  eval @ step {self.state.step}: {pretty}", flush=True)
        return val

    def _track_val(self, val_loss: float) -> None:
        if val_loss < self.state.best_val_loss:
            self.state.best_val_loss = val_loss
            self.state.consecutive_val_increases = 0
            self.save_checkpoint("best")
        else:
            self.state.consecutive_val_increases += 1
            if self.state.consecutive_val_increases >= 3:
                msg = (
                    f"val_loss rose {self.state.consecutive_val_increases} evals in a row; "
                    f"best={self.state.best_val_loss:.4f} now={val_loss:.4f}"
                )
                self.state.alerts.append(msg)
                print(f"  ALERT: {msg}", flush=True)

    # ------------------------------------------------------------------
    def save_checkpoint(self, name: str) -> Path:
        path = self._ckpt_dir / f"{name}.pt"
        tmp = path.with_suffix(".pt.tmp")
        payload = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "state": asdict(self.state),
            "metadata": asdict(self.metadata),
            "scaler": self._scaler.state_dict() if self._scaler is not None else None,
        }
        t_write_0 = time.time()
        torch.save(payload, tmp)
        tmp.replace(path)
        write_seconds = time.time() - t_write_0

        # Sidecar JSON for quick introspection without loading the tensors.
        sidecar = path.with_suffix(".json")
        sidecar.write_text(
            json.dumps(
                {"state": asdict(self.state), "metadata": asdict(self.metadata)},
                indent=2,
            ),
            encoding="utf-8",
        )

        # External backup (e.g. NAS) every N steps.
        backup_cfg = self._external_backup
        backup_ran = False
        if backup_cfg.get("enabled"):
            interval = int(backup_cfg.get("interval_steps", 0))
            if interval > 0 and self.state.step % interval == 0:
                backup_root = Path(backup_cfg["path"])
                try:
                    backup_root.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, backup_root / path.name)
                    shutil.copy2(sidecar, backup_root / sidecar.name)
                    self.state.external_backups_ok += 1
                    backup_ran = True
                except OSError as e:
                    # Non-fatal: a failed backup should not interrupt training.
                    self.state.external_backups_failed += 1
                    self.state.alerts.append(f"backup to {backup_root} failed: {e}")

        self.log(
            {
                "ckpt/write_seconds": float(write_seconds),
                "ckpt/bytes": float(path.stat().st_size),
                "ckpt/external_backups_ok": float(self.state.external_backups_ok),
                "ckpt/external_backups_failed": float(self.state.external_backups_failed),
            },
            self.state.step,
        )
        # Ckpt-write anomaly detection (over rolling median)
        for level, msg in self.health.observe_checkpoint_write(write_seconds, self.state.step):
            print(f"  health[{level.value}]: {msg}", flush=True)
        note = f"  ckpt {name} written in {write_seconds:.1f}s" \
               f" ({path.stat().st_size/1e9:.2f} GB)"
        if backup_ran:
            note += " + backup OK"
        print(note, flush=True)

        self._rotate_checkpoints()
        return path

    def _rotate_checkpoints(self) -> None:
        step_ckpts = sorted(
            (p for p in self._ckpt_dir.glob("step_*.pt") if not p.name.endswith(".tmp")),
            key=lambda p: int(p.stem.split("_", 1)[1]),
            reverse=True,
        )
        for p in step_ckpts[self._keep_last :]:
            p.unlink(missing_ok=True)
            p.with_suffix(".json").unlink(missing_ok=True)

    def load_checkpoint(self, path: str | Path) -> None:
        """Restore full training state (use for resume).

        Warns if the checkpoint's git_sha or config_sha16 don't match the
        current run — that's usually what you want, but silent drift is
        what turns "resume" into "Frankenrun".
        """
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scheduler.load_state_dict(payload["scheduler"])
        self.state = TrainerState(**payload["state"])
        if self._scaler is not None and payload.get("scaler") is not None:
            self._scaler.load_state_dict(payload["scaler"])

        old_meta = payload.get("metadata") or {}
        mismatches = []
        for key in ("git_sha", "config_sha16", "tokenizer_sha16"):
            old = old_meta.get(key)
            new = getattr(self.metadata, key, None)
            if old and new and old != new:
                mismatches.append(f"{key}: {old[:16]} → {new[:16]}")
        if mismatches:
            msg = "resume provenance mismatch — " + "; ".join(mismatches)
            self.state.alerts.append(msg)
            print(f"  warn: {msg}", flush=True)


__all__ = ["PretrainTrainer", "TrainerState"]
