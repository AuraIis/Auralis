"""CLI entry point for Phase-1 pretraining.

Runs a single-device training loop backed by :class:`PretrainTrainer`. For
multi-GPU runs on RunPod, invoke via ``torchrun`` and wrap the model with
FSDP or DeepSpeed outside this script — the module deliberately stays
single-process so the smoke-test path (CPU) and the production path share
the same code.

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
from auralis.training.dataset import MixedDataLoader
from auralis.training.optimizer import build_optimizer, build_scheduler
from auralis.training.trainer import PretrainTrainer
from auralis.training.utils import load_yaml, preflight_check, set_seed


def _resolve_device(cfg_device: str, override: str | None) -> torch.device:
    wanted = override or cfg_device
    if wanted == "cuda" and not torch.cuda.is_available():
        print("warn: cuda requested but not available, falling back to cpu", file=sys.stderr)
        wanted = "cpu"
    return torch.device(wanted)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=REPO / "configs" / "training" / "phase1_pretrain.yaml")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default=None, help="Override config.training.device")
    parser.add_argument("--no-compile", action="store_true")
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

    # Model
    print(f"building model from {config['model']['config_path']}")
    model = build_model(REPO / config["model"]["config_path"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {n_params/1e9:.2f} B ({n_params/1e6:.1f} M)")

    if config["training"].get("torch_compile") and not args.no_compile and device.type == "cuda":
        model = torch.compile(model)

    # Data
    dataloader = MixedDataLoader(
        data_dir=config["data"]["data_dir"],
        mix_ratios=config["data"]["mix_ratios"],
        batch_size=config["training"]["batch_size_per_device"],
        seq_length=config["data"]["seq_length"],
        seed=int(config["data"].get("dataloader_seed", 42)),
    )
    print(f"  rows/batch per language: {dataloader.rows_per_language}")

    # Optim + sched
    optimizer = build_optimizer(model, config["training"]["optimizer"])
    scheduler = build_scheduler(
        optimizer, config["training"]["scheduler"], total_steps=int(config["training"]["total_steps"])
    )

    # Trainer
    trainer = PretrainTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        dataloader=dataloader,
        config=config,
        device=device,
    )
    if args.resume:
        trainer.load_checkpoint(args.resume)
        print(f"  resumed from {args.resume} at step {trainer.state.step}")

    trainer.train()


if __name__ == "__main__":
    main()
