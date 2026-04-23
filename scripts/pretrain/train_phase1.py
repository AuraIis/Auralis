"""CLI entry point for Phase-1 pretraining.

Runs a single-device training loop backed by :class:`PretrainTrainer`. For
multi-GPU runs on RunPod, invoke via ``torchrun`` and wrap the model with
FSDP or DeepSpeed outside this script — the module deliberately stays
single-process so the smoke-test path (CPU) and the production path share
the same code.

What this wires up from ``configs/training/phase1_pretrain.yaml``:

- ``training.dtype`` → forward-pass autocast (bf16 / fp16 on GPU)
- ``training.gradient_checkpointing`` → torch checkpoint per block
- ``data.val_split_bytes`` → reserve N bytes at tail of each .bin for a
  held-out validation loader (never overlapping with train samples)
- ``logging.wandb.enabled`` → init W&B run, pass its logger to the trainer
- ``checkpointing.external_backup`` → ``Trainer`` copies ckpt to NAS every N steps

Typical invocation (GPU host)::

    python scripts/pretrain/train_phase1.py \
        --config configs/training/phase1_pretrain.yaml

Resume from the last checkpoint::

    python scripts/pretrain/train_phase1.py --resume checkpoints/phase1_pretrain/step_25000.pt

Dry-run (preflight only, no weights loaded)::

    python scripts/pretrain/train_phase1.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from auralis.model import build_model
from auralis.model.backend_info import describe_model_backends, format_backend_summary
from auralis.training.dataset import MixedDataLoader
from auralis.training.health import HealthStop
from auralis.training.optimizer import build_optimizer, build_scheduler
from auralis.training.run_report import write_end_manifest, write_start_manifest
from auralis.training.trainer import PretrainTrainer
from auralis.training.utils import load_yaml, preflight_check, set_seed


def _resolve_device(cfg_device: str, override: str | None) -> torch.device:
    wanted = override or cfg_device
    if wanted == "cuda" and not torch.cuda.is_available():
        print("warn: cuda requested but not available, falling back to cpu", file=sys.stderr)
        wanted = "cpu"
    return torch.device(wanted)


def _maybe_init_wandb(config: dict, args: argparse.Namespace):
    """Return (logger_callable, run) — logger is a no-op when W&B is off."""
    wandb_cfg = (config.get("logging") or {}).get("wandb") or {}
    if not wandb_cfg.get("enabled") or args.no_wandb:
        return (lambda _m, _s: None), None
    try:
        import wandb  # type: ignore
    except ImportError:
        print("warn: wandb not installed, skipping W&B logger", file=sys.stderr)
        return (lambda _m, _s: None), None

    run = wandb.init(
        project=wandb_cfg.get("project", "auralis-v2"),
        name=config.get("experiment", {}).get("name"),
        tags=list(wandb_cfg.get("tags", []) or []),
        config=config,
    )
    return (lambda metrics, step: wandb.log(metrics, step=step)), run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=REPO / "configs" / "training" / "phase1_pretrain.yaml")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default=None, help="Override config.training.device")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Skip wandb.init even if config.logging.wandb.enabled.")
    args = parser.parse_args()

    config = load_yaml(args.config)
    set_seed(42)

    device = _resolve_device(config["training"]["device"], args.device)

    preflight_check(
        data_dir=Path(config["data"]["data_dir"]),
        required_data_files=[f"{lang}.bin" for lang in config["data"]["mix_ratios"]],
        checkpoint_dir=Path(config["checkpointing"]["output_dir"]),
        required_free_gb=50.0,
        require_cuda=(device.type == "cuda"),
    )

    if args.dry_run:
        print("preflight ok — exiting (--dry-run)")
        return

    # ---- Model ----
    print(f"building model from {config['model']['config_path']}")
    model = build_model(REPO / config["model"]["config_path"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {n_params/1e9:.2f} B ({n_params/1e6:.1f} M)")

    # Gradient checkpointing — takes the flag from config.advanced already;
    # also honour training.gradient_checkpointing as an override.
    gc_flag = bool(
        config.get("training", {}).get("gradient_checkpointing")
        or getattr(getattr(model, "config", None), "advanced", None)
        and model.config.advanced.gradient_checkpointing
    )
    if gc_flag:
        model.gradient_checkpointing_enable()
        print("  gradient_checkpointing: ENABLED")
    else:
        print("  gradient_checkpointing: disabled")

    # torch.compile gate
    if config["training"].get("torch_compile") and not args.no_compile and device.type == "cuda":
        print("  torch.compile: compiling model…")
        model = torch.compile(model)

    # dtype announce
    dtype_str = str(config["training"].get("dtype", "fp32"))
    print(f"  autocast dtype: {dtype_str}")

    # ---- Data ----
    dcfg = config["data"]
    val_split_bytes = int(dcfg.get("val_split_bytes", 0))
    train_loader = MixedDataLoader(
        data_dir=dcfg["data_dir"],
        mix_ratios=dcfg["mix_ratios"],
        batch_size=config["training"]["batch_size_per_device"],
        seq_length=dcfg["seq_length"],
        seed=int(dcfg.get("dataloader_seed", 42)),
        split="train",
        val_split_bytes=val_split_bytes,
    )
    print(f"  train rows/batch per language: {train_loader.rows_per_language}")

    val_loader = None
    if val_split_bytes > 0:
        val_loader = MixedDataLoader(
            data_dir=dcfg["data_dir"],
            mix_ratios=dcfg["mix_ratios"],
            batch_size=config["training"]["batch_size_per_device"],
            seq_length=dcfg["seq_length"],
            seed=int(dcfg.get("dataloader_seed", 42)),
            split="val",
            val_split_bytes=val_split_bytes,
        )
        print(f"  val enabled: hold-out {val_split_bytes/1e6:.1f} MB per language")
    else:
        print("  val disabled: set data.val_split_bytes > 0 to enable")

    # ---- Optim + sched ----
    optimizer = build_optimizer(model, config["training"]["optimizer"])
    scheduler = build_scheduler(
        optimizer, config["training"]["scheduler"], total_steps=int(config["training"]["total_steps"])
    )

    # ---- W&B ----
    wandb_logger, wandb_run = _maybe_init_wandb(config, args)

    # ---- Trainer ----
    trainer = PretrainTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        dataloader=train_loader,
        val_dataloader=val_loader,
        wandb_logger=wandb_logger,
        config=config,
        device=device,
    )
    if args.resume:
        trainer.load_checkpoint(args.resume)
        print(f"  resumed from {args.resume} at step {trainer.state.step}")

    # Kernel/back-end summary
    backends = describe_model_backends(model.inner if hasattr(model, "inner") else model)
    print(format_backend_summary(backends))

    # Run-report: start
    manifest_path = Path(config["checkpointing"]["output_dir"]) / "MANIFEST.yaml"
    write_start_manifest(
        path=manifest_path,
        config=config,
        metadata=trainer.metadata,
        backend_summary=backends,
    )
    print(f"  manifest: {manifest_path}")

    exit_reason = "completed"
    try:
        trainer.train()
    except HealthStop as e:
        exit_reason = f"health_stop:{e}"
        print(f"  exit: {exit_reason}")
    except KeyboardInterrupt:
        exit_reason = "keyboard_interrupt"
    except Exception as e:                                     # noqa: BLE001
        exit_reason = f"{type(e).__name__}: {e}"
        raise
    finally:
        # Run-report: end (always written, even on exception)
        write_end_manifest(
            path=manifest_path,
            state=trainer.state,
            exit_reason=exit_reason,
            health_summary=trainer.health.summary(),
        )
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
