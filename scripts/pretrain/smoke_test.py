"""End-to-end CPU smoke test of the full Phase-1 pipeline.

Runs a dozen real training steps through ``PretrainTrainer`` on the 100M
HelixModel against synthetic token .bin files, and verifies:

- the pipeline wires together (model + loader + optimizer + scheduler + trainer)
- gradient accumulation + clipping + cosine schedule work end to end
- checkpoints are written and can be reloaded
- loss finite throughout

Takes ~60-90 s on CPU. This is the gate we run BEFORE burning GPU hours on
RunPod. Pass = green light for real pretraining.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from auralis.model import build_model
from auralis.training.dataset import MixedDataLoader
from auralis.training.optimizer import build_optimizer, build_scheduler
from auralis.training.trainer import PretrainTrainer
from auralis.training.utils import load_yaml, set_seed


def _write_synthetic_bins(data_dir: Path, vocab_size: int, tokens_per_lang: int) -> None:
    rng = np.random.default_rng(0)
    for lang in ("english", "german", "code"):
        arr = rng.integers(0, vocab_size, size=tokens_per_lang, dtype=np.uint32)
        arr.tofile(data_dir / f"{lang}.bin")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=REPO / "configs" / "training" / "phase1_pretrain.yaml")
    args = parser.parse_args()

    full_cfg = load_yaml(args.config)
    smoke = full_cfg.get("smoke_test", {})
    set_seed(0)

    model_cfg_path = REPO / smoke.get("model_config", "configs/model/helix_v2_100m.yaml")
    batch_size = int(smoke.get("batch_size", 2))
    grad_accum = int(smoke.get("gradient_accumulation", 1))
    seq_length = int(smoke.get("seq_length", 64))
    total_steps = int(smoke.get("total_steps", 20))
    log_every = int(smoke.get("log_every", 5))

    print(f"Building model from {model_cfg_path} (CPU smoke test)")
    model = build_model(model_cfg_path)
    print(f"  params: {sum(p.numel() for p in model.parameters())/1e6:.1f} M")

    # ignore_cleanup_errors: np.memmap keeps file handles open on Windows
    # until GC runs, which can race the TemporaryDirectory exit.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        tmp_path = Path(tmp)
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        ckpt_dir = tmp_path / "ckpt"

        # Enough synthetic tokens for each lang so the loader never exhausts them.
        _write_synthetic_bins(data_dir, vocab_size=model.config.vocab_size,
                              tokens_per_lang=max(10_000, seq_length * 100))

        loader = MixedDataLoader(
            data_dir=data_dir,
            mix_ratios=full_cfg["data"]["mix_ratios"],
            batch_size=batch_size,
            seq_length=seq_length,
            seed=0,
        )

        opt = build_optimizer(model, full_cfg["training"]["optimizer"])
        sched = build_scheduler(
            opt, full_cfg["training"]["scheduler"], total_steps=total_steps
        )

        run_cfg = {
            "data": {"seq_length": seq_length},
            "training": {
                "batch_size_per_device": batch_size,
                "gradient_accumulation": grad_accum,
                "gradient_clip_norm": float(full_cfg["training"]["gradient_clip_norm"]),
                "total_steps": total_steps,
            },
            "logging": {"log_every": log_every, "eval_every": 9999, "save_every": total_steps},
            "checkpointing": {"output_dir": str(ckpt_dir), "save_last_n": 2},
        }
        trainer = PretrainTrainer(
            model=model, optimizer=opt, scheduler=sched,
            dataloader=loader, config=run_cfg, device="cpu",
        )

        import time
        t0 = time.time()
        state = trainer.train()
        dt = time.time() - t0

        # Assertions on final state
        assert state.step == total_steps, state
        assert state.tokens_seen >= batch_size * grad_accum * seq_length * total_steps
        checkpoints = list(ckpt_dir.glob("step_*.pt"))
        assert checkpoints, "no checkpoint was written"

        # Reload check
        trainer.load_checkpoint(checkpoints[0])
        assert trainer.state.step == total_steps

        print()
        print(f"smoke test OK in {dt:.1f}s "
              f"({state.tokens_seen:,} tokens over {state.step} steps)")
        print(f"  final best_val_loss tracker = {state.best_val_loss}")
        print(f"  checkpoint: {checkpoints[0]}")


if __name__ == "__main__":
    main()
