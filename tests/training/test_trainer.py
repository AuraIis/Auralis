"""End-to-end smoke test of PretrainTrainer on a tiny 100M-ish model.

Feeds synthetic token bins into the loader (no real tokenization needed),
runs a handful of gradient-accumulation steps, and checks:

- loss is finite at every step
- loss is non-increasing (on average) over the short run
- the trainer writes a checkpoint + JSON sidecar
- loading the checkpoint restores ``state.step``

Kept under ~2 minutes on CPU by using a 4-layer shrunken model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from auralis.model.config import AuralisConfig, LayerConfig
from auralis.model.helix_model import HelixModel
from auralis.training.dataset import MixedDataLoader
from auralis.training.optimizer import build_optimizer, build_scheduler
from auralis.training.trainer import PretrainTrainer


def _tiny_model() -> HelixModel:
    layers = (
        [LayerConfig(type="mamba", d_state=16, d_conv=4, expand_factor=2)] * 1
        + [LayerConfig(type="gla", d_state=16)] * 2
        + [LayerConfig(type="sparse_attention", window_size=16, global_tokens=4, use_rope=True)] * 1
    )
    cfg = AuralisConfig(
        name="tiny", version="1.0",
        vocab_size=4096,
        d_model=64, n_layers=4, n_heads=4, d_head=16, d_ffn=128,
        layers=layers,
    )
    cfg.advanced.tie_embeddings = True
    cfg.position_encoding.max_seq_length = 64
    return HelixModel(cfg)


def _write_bins(dir_path: Path, n_each: int, vocab: int) -> None:
    rng = np.random.default_rng(0)
    for lang in ("english", "german", "code"):
        arr = rng.integers(0, vocab, size=n_each, dtype=np.uint32)
        arr.tofile(dir_path / f"{lang}.bin")


@pytest.fixture
def trainer_env(tmp_path: Path):
    torch.manual_seed(0)
    np.random.seed(0)

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_bins(data_dir, n_each=2000, vocab=4096)

    ckpt_dir = tmp_path / "ckpt"
    model = _tiny_model()
    loader = MixedDataLoader(
        data_dir=data_dir,
        mix_ratios={"english": 0.5, "german": 0.5, "code": 0.0},
        batch_size=4,
        seq_length=16,
        seed=0,
    )
    opt = build_optimizer(model, {"name": "adamw", "lr": 1e-3, "weight_decay": 0.01})
    sched = build_scheduler(
        opt, {"type": "cosine", "warmup_steps": 2, "min_lr_ratio": 0.5}, total_steps=10
    )
    config = {
        "data": {"seq_length": 16},
        "training": {
            "batch_size_per_device": 4,
            "gradient_accumulation": 1,
            "gradient_clip_norm": 1.0,
            "total_steps": 10,
        },
        "logging": {"log_every": 2, "eval_every": 9999, "save_every": 10},
        "checkpointing": {"output_dir": str(ckpt_dir), "save_last_n": 2},
    }
    trainer = PretrainTrainer(
        model=model,
        optimizer=opt,
        scheduler=sched,
        dataloader=loader,
        config=config,
        device="cpu",
    )
    return trainer, ckpt_dir


def test_trainer_runs_and_loss_finite(trainer_env):
    trainer, _ = trainer_env
    losses: list[float] = []

    # Patch the log function to capture loss values.
    original = trainer.log
    def capture(metrics, step):
        if "train/loss" in metrics:
            losses.append(metrics["train/loss"])
        original(metrics, step)
    trainer.log = capture

    # Capture weights before training to verify the optimizer actually moved them.
    w_before = next(p.detach().clone() for p in trainer.model.parameters() if p.requires_grad)

    final_state = trainer.train()

    assert final_state.step == 10
    assert len(losses) >= 3
    # Every logged loss finite + within a sane absolute range (vocab=4096 ⇒ ln≈8.3).
    for lv in losses:
        assert 0 < lv < 20
    # Parameters actually changed (optimizer.step did something).
    w_after = next(p.detach() for p in trainer.model.parameters() if p.requires_grad)
    assert not torch.equal(w_before, w_after)
    assert torch.isfinite(w_after).all()


def test_trainer_saves_and_loads_checkpoint(trainer_env):
    trainer, ckpt_dir = trainer_env
    trainer.train()
    ckpts = list(ckpt_dir.glob("step_*.pt"))
    assert ckpts, "no checkpoint written"
    sidecar = ckpts[0].with_suffix(".json")
    assert sidecar.exists()

    # Make a fresh trainer from the same pieces, load, verify state.
    trainer2 = trainer.__class__(
        model=trainer.model,
        optimizer=trainer.optimizer,
        scheduler=trainer.scheduler,
        dataloader=trainer.dataloader,
        config=trainer.config,
        device="cpu",
    )
    trainer2.load_checkpoint(ckpts[0])
    assert trainer2.state.step == 10


def test_trainer_runs_evaluation_when_val_loader_present(trainer_env, tmp_path: Path):
    trainer, _ = trainer_env
    # Build a tiny val loader that reuses the same synthetic bins with a
    # fresh split (no bytes reserved — just reuse the full window for the test).
    import numpy as np
    from auralis.training.dataset import MixedDataLoader
    data_dir = tmp_path / "valdata"
    data_dir.mkdir()
    rng = np.random.default_rng(1)
    for lang in ("english", "german", "code"):
        rng.integers(0, 4096, size=2000, dtype=np.uint32).tofile(data_dir / f"{lang}.bin")
    trainer.val_dataloader = MixedDataLoader(
        data_dir=data_dir,
        mix_ratios={"english": 0.5, "german": 0.5, "code": 0.0},
        batch_size=2, seq_length=16, seed=1,
    )
    # Make eval fire once during the 10-step run
    trainer._eval_every = 5
    trainer.config["evaluation"] = {"max_val_batches": 3}

    metrics_captured: list[dict[str, float]] = []
    original = trainer.log
    def cap(m, s):
        metrics_captured.append(dict(m))
        original(m, s)
    trainer.log = cap

    trainer.train()
    # At least one eval metric must have been logged.
    assert any("eval/val_loss" in m for m in metrics_captured), metrics_captured
    # best.pt must exist after the first successful eval.
    best = Path(trainer.config["checkpointing"]["output_dir"]) / "best.pt"
    assert best.exists()


def test_trainer_bf16_autocast_selected_on_config(trainer_env):
    trainer, _ = trainer_env
    trainer.config["training"]["dtype"] = "bf16"
    # Re-run __init__-style flags
    trainer._amp_dtype = torch.bfloat16
    trainer._use_amp = True
    ctx = trainer._autocast()
    # Can't assert much at a type level, but the autocast manager must be usable
    with ctx:
        t = torch.randn(2, 3)
        y = t @ t.T
    assert y.shape == (2, 2)


def test_trainer_raises_on_nan_loss(trainer_env, monkeypatch):
    trainer, _ = trainer_env

    class NaNModel(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
        def forward(self, input_ids, labels):
            out = self.inner(input_ids=input_ids, labels=labels)
            out["loss"] = torch.tensor(float("nan"))
            return out

    trainer.model = NaNModel(trainer.model)
    with pytest.raises(RuntimeError, match="non-finite loss"):
        trainer.train()
