"""Tests for MixedDataLoader + PretrainDataset (memmap-based)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from auralis.training.dataset import MixedDataLoader, PretrainDataset


def _write_bin(path: Path, n_tokens: int, rng: np.random.Generator) -> None:
    arr = rng.integers(0, 200_000, size=n_tokens, dtype=np.uint32)
    arr.tofile(path)


@pytest.fixture
def three_bins(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    for lang, n in [("english", 20_000), ("german", 10_000), ("code", 5_000)]:
        _write_bin(tmp_path / f"{lang}.bin", n, rng)
    return tmp_path


def test_pretrain_dataset_sample_shape(three_bins: Path):
    ds = PretrainDataset(
        bin_path=three_bins / "english.bin",
        seq_length=128,
        rng=np.random.default_rng(1),
    )
    sample = ds.sample()
    assert sample.shape == (129,)
    assert sample.dtype == torch.int64


def test_pretrain_dataset_raises_on_short_bin(tmp_path: Path):
    p = tmp_path / "tiny.bin"
    np.zeros(10, dtype=np.uint32).tofile(p)
    with pytest.raises(ValueError):
        PretrainDataset(bin_path=p, seq_length=64, rng=np.random.default_rng(0))


def test_mixed_dataloader_partitions_rows_correctly(three_bins: Path):
    loader = MixedDataLoader(
        data_dir=three_bins,
        mix_ratios={"english": 0.75, "german": 0.20, "code": 0.05},
        batch_size=16,
        seq_length=64,
    )
    rows = loader.rows_per_language
    assert sum(rows.values()) == 16
    assert rows["english"] == 12  # 16 * 0.75
    assert rows["german"] == 3    # 16 * 0.20 = 3.2 → 3
    assert rows["code"] == 1      # 16 * 0.05 = 0.8 → 1 (largest-remainder)


def test_mixed_dataloader_batch_shape_and_types(three_bins: Path):
    loader = MixedDataLoader(
        data_dir=three_bins,
        mix_ratios={"english": 0.5, "german": 0.5, "code": 0.0},
        batch_size=4,
        seq_length=32,
    )
    batch = next(loader)
    assert batch["input_ids"].shape == (4, 32)
    assert batch["labels"].shape == (4, 32)
    assert batch["input_ids"].dtype == torch.int64


def test_mixed_dataloader_labels_equal_input_ids(three_bins: Path):
    """MixedDataLoader passes unshifted labels; HelixModel shifts internally."""
    loader = MixedDataLoader(
        data_dir=three_bins,
        mix_ratios={"english": 1.0, "german": 0.0, "code": 0.0},
        batch_size=2,
        seq_length=16,
    )
    batch = next(loader)
    assert torch.equal(batch["input_ids"], batch["labels"])


def test_mixed_dataloader_mix_ratios_must_sum_to_one(three_bins: Path):
    with pytest.raises(ValueError):
        MixedDataLoader(
            data_dir=three_bins,
            mix_ratios={"english": 0.5, "german": 0.2, "code": 0.1},  # = 0.8
            batch_size=4,
            seq_length=16,
        )


def test_mixed_dataloader_missing_bin_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        MixedDataLoader(
            data_dir=tmp_path,
            mix_ratios={"english": 1.0},
            batch_size=2,
            seq_length=8,
        )


def test_mixed_dataloader_train_val_split_disjoint(three_bins: Path):
    """Train and val should sample from disjoint regions of the same .bin."""
    # val_split_bytes: reserve last 4000 tokens (=16000 bytes, uint32) per lang.
    # english.bin has 20k tokens → train window [0, 16000), val [16000, 20000).
    val_bytes = 16_000
    train = MixedDataLoader(
        data_dir=three_bins,
        mix_ratios={"english": 0.5, "german": 0.5, "code": 0.0},
        batch_size=8, seq_length=32, seed=0,
        split="train", val_split_bytes=val_bytes,
    )
    val = MixedDataLoader(
        data_dir=three_bins,
        mix_ratios={"english": 0.5, "german": 0.5, "code": 0.0},
        batch_size=8, seq_length=32, seed=0,
        split="val", val_split_bytes=val_bytes,
    )
    en_train = train.datasets["english"]
    en_val = val.datasets["english"]
    # Val starts where train ends (approximately — train_end is clamped down
    # so there's always room for one seq+1 even if val is huge).
    assert en_train.train_end <= en_val.train_start
    assert en_val.train_end > en_val.train_start
    # Train and val have different RNG streams → first samples must differ.
    a = train.datasets["english"].sample()
    b = val.datasets["english"].sample()
    assert not torch.equal(a, b)


def test_mixed_dataloader_val_too_small_raises(three_bins: Path):
    with pytest.raises(ValueError, match="val split for"):
        MixedDataLoader(
            data_dir=three_bins,
            mix_ratios={"english": 1.0, "german": 0.0, "code": 0.0},
            batch_size=2, seq_length=64, seed=0,
            split="val", val_split_bytes=100,    # far too small
        )


def test_mixed_dataloader_split_name_validated(three_bins: Path):
    with pytest.raises(ValueError, match="split must be"):
        MixedDataLoader(
            data_dir=three_bins,
            mix_ratios={"english": 1.0, "german": 0.0, "code": 0.0},
            batch_size=2, seq_length=16, split="holdout",
        )
