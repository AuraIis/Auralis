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

import json
import shutil
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


_AMP_DTYPES: dict[str, torch.dtype] = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
}


@dataclass
class TrainerState:
    """Pure-data training state — persisted in every checkpoint."""

    step: int = 0
    best_val_loss: float = float("inf")
    consecutive_val_increases: int = 0
    tokens_seen: int = 0
    wall_clock_seconds: float = 0.0
    alerts: list[str] = field(default_factory=list)


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
        self.log = wandb_logger or (lambda _metrics, _step: None)

        # Unpack knobs that get hit every step.
        tcfg = config["training"]
        self._total_steps = int(tcfg["total_steps"])
        self._grad_accum = int(tcfg.get("gradient_accumulation", 1))
        self._clip_norm = float(tcfg.get("gradient_clip_norm", 1.0))
        self._batch_tokens = int(
            tcfg["batch_size_per_device"] * self._grad_accum * config["data"]["seq_length"]
        )

        # AMP dtype — config.training.dtype drives the forward-pass autocast.
        # CPU only supports bf16 autocast (no fp16); we fall back to fp32
        # automatically if the caller picks fp16 on CPU.
        dtype_str = str(tcfg.get("dtype", "fp32")).lower()
        self._amp_dtype = _AMP_DTYPES.get(dtype_str, torch.float32)
        if self.device.type == "cpu" and self._amp_dtype == torch.float16:
            self._amp_dtype = torch.float32
        self._use_amp = self._amp_dtype != torch.float32

        lcfg = config["logging"]
        self._log_every = int(lcfg.get("log_every", 10))
        self._eval_every = int(lcfg.get("eval_every", 1000))
        self._save_every = int(lcfg.get("save_every", 2500))

        ccfg = config["checkpointing"]
        self._ckpt_dir = Path(ccfg["output_dir"])
        self._ckpt_dir.mkdir(parents=True, exist_ok=True)
        self._keep_last = int(ccfg.get("save_last_n", 3))
        self._external_backup = ccfg.get("external_backup") or {}

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

        while self.state.step < self._total_steps:
            loss_acc = 0.0
            self.optimizer.zero_grad(set_to_none=True)

            for _ in range(self._grad_accum):
                batch = next(data_iter)
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                with self._autocast():
                    out = self.model(input_ids=batch["input_ids"], labels=batch["labels"])
                    loss = out["loss"] / self._grad_accum
                if not torch.isfinite(loss):
                    msg = f"non-finite loss at step {self.state.step}: {loss.item()}"
                    self.state.alerts.append(msg)
                    raise RuntimeError(msg)
                loss.backward()
                loss_acc += loss.item()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self._clip_norm
            )
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
                self.log(
                    {
                        "train/loss": avg_loss,
                        "train/grad_norm": float(grad_norm),
                        "train/lr": lr,
                        "train/tokens_per_second": tps,
                        "train/tokens_seen": self.state.tokens_seen,
                    },
                    self.state.step,
                )
                print(
                    f"step {self.state.step:6d} | loss {avg_loss:6.4f} | "
                    f"lr {lr:.2e} | grad_norm {float(grad_norm):5.2f} | "
                    f"tok/s {tps/1e3:6.1f}k",
                    flush=True,
                )
                window_loss_sum = 0.0
                window_t0 = time.time()

            if self.val_dataloader is not None and self.state.step % self._eval_every == 0:
                val_loss = self._evaluate()
                self._track_val(val_loss)

            if self.state.step % self._save_every == 0:
                self.save_checkpoint(f"step_{self.state.step}")

        # Final checkpoint
        self.save_checkpoint(f"step_{self.state.step}")
        return self.state

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _evaluate(self) -> float:
        self.model.eval()
        assert self.val_dataloader is not None
        max_batches = int(self.config.get("evaluation", {}).get("max_val_batches", 50))
        losses: list[float] = []
        it = iter(self.val_dataloader)
        for _ in range(max_batches):
            batch = next(it)
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            with self._autocast():
                out = self.model(input_ids=batch["input_ids"], labels=batch["labels"])
            losses.append(out["loss"].item())
        self.model.train()
        val = sum(losses) / max(1, len(losses))
        self.log({"eval/val_loss": val}, self.state.step)
        print(f"  val_loss @ step {self.state.step}: {val:.4f}", flush=True)
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
        }
        torch.save(payload, tmp)
        tmp.replace(path)

        # Sidecar JSON for quick introspection without loading the tensors.
        sidecar = path.with_suffix(".json")
        sidecar.write_text(json.dumps(asdict(self.state), indent=2), encoding="utf-8")

        # External backup (e.g. NAS) every N steps.
        backup_cfg = self._external_backup
        if backup_cfg.get("enabled"):
            interval = int(backup_cfg.get("interval_steps", 0))
            if interval > 0 and self.state.step % interval == 0:
                backup_root = Path(backup_cfg["path"])
                try:
                    backup_root.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, backup_root / path.name)
                    shutil.copy2(sidecar, backup_root / sidecar.name)
                except OSError as e:
                    # Non-fatal: a failed backup should not interrupt training.
                    self.state.alerts.append(f"backup to {backup_root} failed: {e}")

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
        """Restore full training state (use for resume)."""
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scheduler.load_state_dict(payload["scheduler"])
        self.state = TrainerState(**payload["state"])


__all__ = ["PretrainTrainer", "TrainerState"]
