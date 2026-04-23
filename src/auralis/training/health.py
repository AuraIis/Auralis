"""Training-health guards.

Detects the four classes of trouble that turn a 3-week pretraining run into a
3-week bill:

- **Gradient explosion**: ``grad_norm`` above threshold for K consecutive steps
- **Gradient collapse**: ``grad_norm`` below ~0 for K steps (dead model)
- **Loss spike**: current loss > ``factor × running_average`` — sudden jump
- **Throughput drop**: ``tokens_per_second`` below ``min_ratio × peak`` sustained

Each guard is configured independently (see ``HealthConfig``). On breach, the
monitor either just flags it (``AlertLevel.WARN``) or requests an immediate
abort (``AlertLevel.STOP``). The trainer checks for STOP after each logged
step and raises ``HealthStop`` if set.

Design note: guards are observational-only — they never mutate model state.
They only read the metrics dict the trainer already produces, so they can be
tested without needing a full training loop.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AlertLevel(str, Enum):
    INFO = "info"
    WARN = "warn"
    STOP = "stop"


@dataclass
class HealthConfig:
    # Gradient explosion: grad_norm above this for K consecutive logged windows.
    grad_explosion_threshold: float = 100.0
    grad_explosion_k: int = 3
    # Gradient collapse: grad_norm below this for K consecutive windows.
    grad_collapse_threshold: float = 1e-6
    grad_collapse_k: int = 5
    # Loss spike: current > factor * running_avg, window of last N.
    loss_spike_factor: float = 3.0
    loss_spike_avg_window: int = 20
    # Throughput floor: tps below min_ratio of peak (after warmup).
    tps_min_ratio: float = 0.3
    tps_warmup_logs: int = 5
    tps_k: int = 3
    # Val-loss regression: stop if val rose K consecutive evals (Trainer already
    # tracks a 3-increase warning; this is the hard cutoff).
    val_regression_stop_k: int = 5


@dataclass
class HealthState:
    grad_explosion_count: int = 0
    grad_collapse_count: int = 0
    tps_below_count: int = 0
    tps_peak: float = 0.0
    loss_window: deque[float] = field(default_factory=lambda: deque(maxlen=32))
    alerts: list[tuple[int, AlertLevel, str]] = field(default_factory=list)
    stop_requested: bool = False
    stop_reason: str = ""


class HealthStop(RuntimeError):
    """Raised by the trainer when a health guard demands an immediate stop."""


class HealthMonitor:
    """Stateful observer over training metrics. Consumes the same dicts the
    trainer logs (``train/loss``, ``train/grad_norm``, ``train/tokens_per_second``)
    and an optional ``eval/val_loss`` on eval steps.
    """

    def __init__(self, config: HealthConfig | None = None):
        self.config = config or HealthConfig()
        self.state = HealthState()
        self.state.loss_window = deque(maxlen=max(8, self.config.loss_spike_avg_window))

    # ------------------------------------------------------------------
    def observe(self, metrics: dict[str, float], step: int) -> list[tuple[AlertLevel, str]]:
        """Ingest a metrics dict. Returns any NEW alerts produced by this call."""
        fresh: list[tuple[AlertLevel, str]] = []
        c = self.config

        grad = metrics.get("train/grad_norm")
        if grad is not None:
            if grad > c.grad_explosion_threshold:
                self.state.grad_explosion_count += 1
                if self.state.grad_explosion_count >= c.grad_explosion_k:
                    fresh.append((AlertLevel.STOP,
                                  f"grad_norm={grad:.2f} > {c.grad_explosion_threshold} "
                                  f"for {self.state.grad_explosion_count} windows"))
                    self._request_stop("grad_explosion")
            else:
                self.state.grad_explosion_count = 0

            if grad < c.grad_collapse_threshold:
                self.state.grad_collapse_count += 1
                if self.state.grad_collapse_count >= c.grad_collapse_k:
                    fresh.append((AlertLevel.WARN,
                                  f"grad_norm={grad:.2e} near zero for "
                                  f"{self.state.grad_collapse_count} windows"))
            else:
                self.state.grad_collapse_count = 0

        loss = metrics.get("train/loss")
        if loss is not None:
            if len(self.state.loss_window) >= 4:
                avg = sum(self.state.loss_window) / len(self.state.loss_window)
                if avg > 0 and loss > avg * c.loss_spike_factor:
                    fresh.append((AlertLevel.WARN,
                                  f"loss spike: {loss:.3f} > "
                                  f"{c.loss_spike_factor}× running_avg {avg:.3f}"))
            self.state.loss_window.append(loss)

        tps = metrics.get("train/tokens_per_second")
        if tps is not None:
            self.state.tps_peak = max(self.state.tps_peak, tps)
            if len(self.state.loss_window) >= c.tps_warmup_logs:
                if self.state.tps_peak > 0 and tps < self.state.tps_peak * c.tps_min_ratio:
                    self.state.tps_below_count += 1
                    if self.state.tps_below_count >= c.tps_k:
                        fresh.append((AlertLevel.WARN,
                                      f"tok/s collapsed: {tps:.0f} < "
                                      f"{c.tps_min_ratio:.0%} of peak "
                                      f"{self.state.tps_peak:.0f}"))
                else:
                    self.state.tps_below_count = 0

        for level, msg in fresh:
            self.state.alerts.append((step, level, msg))
        return fresh

    # ------------------------------------------------------------------
    def observe_val(self, val_loss: float, best_val_loss: float, consecutive_increases: int, step: int) -> list[tuple[AlertLevel, str]]:
        """Called by the trainer after each val. Separate path because the
        trainer already owns ``consecutive_val_increases`` — we just decide
        whether it's long enough to stop."""
        fresh: list[tuple[AlertLevel, str]] = []
        if consecutive_increases >= self.config.val_regression_stop_k:
            fresh.append((AlertLevel.STOP,
                          f"val_loss rose {consecutive_increases} evals "
                          f"(current {val_loss:.4f}, best {best_val_loss:.4f})"))
            self._request_stop("val_regression")
        for level, msg in fresh:
            self.state.alerts.append((step, level, msg))
        return fresh

    # ------------------------------------------------------------------
    def observe_backup(self, ok: bool, fail_count: int, consecutive_fail_threshold: int = 3) -> list[tuple[AlertLevel, str]]:
        """Repeated backup failures ⇒ stop before the checkpoint graveyard."""
        fresh: list[tuple[AlertLevel, str]] = []
        if not ok and fail_count >= consecutive_fail_threshold:
            fresh.append((AlertLevel.STOP,
                          f"{fail_count} consecutive backup failures"))
            self._request_stop("backup_failures")
        return fresh

    # ------------------------------------------------------------------
    def _request_stop(self, reason: str) -> None:
        self.state.stop_requested = True
        if not self.state.stop_reason:
            self.state.stop_reason = reason

    def should_stop(self) -> bool:
        return self.state.stop_requested

    def summary(self) -> dict[str, Any]:
        return {
            "stop_requested": self.state.stop_requested,
            "stop_reason": self.state.stop_reason,
            "n_alerts": len(self.state.alerts),
            "grad_explosion_count": self.state.grad_explosion_count,
            "grad_collapse_count": self.state.grad_collapse_count,
            "tps_peak": self.state.tps_peak,
            "alerts": [
                {"step": s, "level": lvl.value, "msg": m}
                for (s, lvl, m) in self.state.alerts[-20:]
            ],
        }


__all__ = ["AlertLevel", "HealthConfig", "HealthMonitor", "HealthState", "HealthStop"]
