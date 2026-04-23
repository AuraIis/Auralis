"""Pretraining dataset: memmap .bin streams with language-mix batching.

The tokenized corpus from ``tokenize_for_pretraining.py`` lives as flat
``uint32`` files per language (``english.bin``, ``german.bin``, ``code.bin``)
on the NAS. Random-access via ``numpy.memmap`` is cheap even at 80+ GB —
we never load the file, only slice into it.

Two classes:

- :class:`PretrainDataset` — infinite sampler over a single .bin file. Each
  ``__getitem__`` returns ``seq_length+1`` tokens (the extra one is shifted
  inside the model to form labels).
- :class:`MixedDataLoader` — composes multiple PretrainDatasets (one per
  language) and yields batches whose rows are drawn according to
  ``mix_ratios``. Uses a deterministic-per-seed RNG so runs are reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


def _mmap_bin(path: Path) -> np.memmap:
    """Open a flat uint32 .bin file as a read-only memmap."""
    size_bytes = path.stat().st_size
    if size_bytes == 0:
        raise ValueError(f"empty bin: {path}")
    n_tokens = size_bytes // 4
    return np.memmap(path, dtype=np.uint32, mode="r", shape=(n_tokens,))


@dataclass
class PretrainDataset:
    """Infinite token-stream sampler over one .bin file.

    Each call to :meth:`sample` draws a random start offset inside the
    ``[train_start, train_end)`` window and returns a contiguous block of
    ``seq_length + 1`` tokens as an ``int64`` tensor.

    The ``train_end`` / ``val_start`` split supports a disjoint validation
    holdout: the last ``val_split_tokens`` tokens of the .bin file are
    reserved for validation and are never yielded by a train-mode sampler.
    """

    bin_path: Path
    seq_length: int
    rng: np.random.Generator
    train_start: int = 0
    train_end: int | None = None          # exclusive; None → all tokens

    def __post_init__(self) -> None:
        self._mmap = _mmap_bin(Path(self.bin_path))
        self._n_tokens = self._mmap.shape[0]
        if self.train_end is None:
            self.train_end = self._n_tokens
        if self.train_end - self.train_start <= self.seq_length + 1:
            raise ValueError(
                f"{self.bin_path} window [{self.train_start}, {self.train_end}) has "
                f"{self.train_end - self.train_start} tokens, need > seq_length+1"
            )

    @property
    def num_tokens(self) -> int:
        return int(self._n_tokens)

    def sample(self) -> torch.Tensor:
        lo = self.train_start
        hi = self.train_end - self.seq_length - 1
        start = int(self.rng.integers(lo, hi))
        block = self._mmap[start : start + self.seq_length + 1].astype(np.int64, copy=True)
        return torch.from_numpy(block)


class MixedDataLoader:
    """Batches tokens from multiple language streams according to mix ratios.

    Yields dicts with ``input_ids`` and ``labels`` of shape
    ``[batch_size, seq_length]``. ``labels`` is just ``input_ids`` shifted
    inside the model's own loss computation, so we simply emit the same
    sequence for both.

    The number of rows per language per batch is proportional to
    ``mix_ratios[lang]``, rounded so totals match ``batch_size``.
    """

    def __init__(
        self,
        data_dir: str | Path,
        mix_ratios: dict[str, float],
        batch_size: int,
        seq_length: int,
        seed: int = 42,
        split: str = "train",                    # "train" | "val"
        val_split_bytes: int = 0,                # last N BYTES of each .bin reserved for val
    ):
        self.data_dir = Path(data_dir)
        self.mix_ratios = dict(mix_ratios)
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.split = split
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")

        total = sum(self.mix_ratios.values())
        if not 0.99 <= total <= 1.01:
            raise ValueError(f"mix_ratios must sum to 1, got {total}")

        # uint32 = 4 bytes per token
        val_split_tokens = int(val_split_bytes) // 4

        # Per-language RNGs (distinct seeds so draws do not correlate).
        # Val gets its own offset in the seed so train and val do not align.
        self.datasets: dict[str, PretrainDataset] = {}
        for i, lang in enumerate(sorted(self.mix_ratios)):
            bin_path = self.data_dir / f"{lang}.bin"
            if not bin_path.exists():
                raise FileNotFoundError(bin_path)
            n_tokens_total = bin_path.stat().st_size // 4

            if split == "train":
                train_start = 0
                train_end = max(seq_length + 2, n_tokens_total - val_split_tokens)
                rng = np.random.default_rng(seed + i * 7919)
            else:
                # Val window: last val_split_tokens of the file. Must be > seq_length+1.
                train_start = max(0, n_tokens_total - val_split_tokens)
                train_end = n_tokens_total
                if train_end - train_start <= seq_length + 1:
                    raise ValueError(
                        f"val split for {lang} too small: {train_end - train_start} tokens "
                        f"(need > seq_length+1={seq_length+1}). Increase val_split_bytes."
                    )
                rng = np.random.default_rng(seed + i * 7919 + 1_000_003)

            self.datasets[lang] = PretrainDataset(
                bin_path=bin_path, seq_length=seq_length, rng=rng,
                train_start=train_start, train_end=train_end,
            )

        # Rows-per-batch per language.
        self._rows_per_lang = self._partition_rows()

    def _partition_rows(self) -> dict[str, int]:
        """Largest-remainder allocation: each lang gets floor(B*p); remainder
        goes to the largest fractional parts."""
        exact = {lang: self.batch_size * p for lang, p in self.mix_ratios.items()}
        floor = {lang: int(v) for lang, v in exact.items()}
        remaining = self.batch_size - sum(floor.values())
        if remaining:
            frac = sorted(exact.items(), key=lambda kv: -(kv[1] - floor[kv[0]]))
            for lang, _ in frac[:remaining]:
                floor[lang] += 1
        return floor

    @property
    def rows_per_language(self) -> dict[str, int]:
        return dict(self._rows_per_lang)

    def __iter__(self):
        return self

    def __next__(self) -> dict[str, torch.Tensor]:
        rows: list[torch.Tensor] = []
        for lang, n in self._rows_per_lang.items():
            ds = self.datasets[lang]
            for _ in range(n):
                rows.append(ds.sample())
        # Shuffle so a batch does not begin with all of one language.
        np.random.default_rng(int(torch.randint(0, 1 << 30, (1,)).item())).shuffle(rows)
        batch = torch.stack(rows, dim=0)
        # Drop the extra sampled token — HelixModel._shift_loss shifts labels
        # internally, so both tensors are the same length and the same content.
        # Passing ``labels = input_ids`` (not a pre-shifted copy) avoids the
        # classic "off-by-two" bug where loss accidentally predicts t+2 from t.
        input_ids = batch[:, :-1].contiguous()
        labels = input_ids.clone()
        return {"input_ids": input_ids, "labels": labels}


__all__ = ["MixedDataLoader", "PretrainDataset"]
