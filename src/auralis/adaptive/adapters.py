"""Integration shims between the adaptive framework and the existing Helix code.

Everything torch- or tokenizer-specific lives here so the controller/signals stay
pure. Three adapters:

- :class:`TokenizerAdapter` — SentencePiece encode/decode + the chat template +
  a robust prompt/continuation tokenization split for margin probes.
- :class:`ModelAdapter` — wraps a ``HelixModel`` for a training step under bf16
  autocast, and exposes the raw module for scoring.
- stage data sources — :func:`build_stage_loader` turns a stage's ``data`` spec
  into an iterator of ``{"input_ids", "labels"}`` batches (raw mixed text, or
  formatted SFT JSONL with optional assistant-only loss masking).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import torch

from .probes import MarginProbe

SYSTEM_DE = (
    "Du bist Auralis, ein hilfreicher deutscher KI-Assistent. "
    "Antworte korrekt, knapp und ehrlich. Wenn etwas unsicher oder erfunden ist, sage das deutlich."
)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
class TokenizerAdapter:
    def __init__(self, model_path: str | Path):
        import sentencepiece as spm

        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(str(model_path))
        self.eos_id = self.sp.eos_id()
        self.pad_id = self.sp.pad_id() if self.sp.pad_id() >= 0 else (
            self.eos_id if self.eos_id >= 0 else 0
        )

    def encode(self, text: str) -> list[int]:
        return self.sp.encode(text, out_type=int)

    def decode(self, ids: list[int]) -> str:
        return self.sp.decode(ids)

    def chat_prompt(self, user: str, system: str = SYSTEM_DE) -> str:
        """User turn up to (and including) the assistant marker — no answer.

        Delegates to the canonical builder in ``tokenizer.chat_template`` (the
        single source of truth for the chat format — see LESSONS.md L-001) so
        this probe path can never drift from the training/inference rendering.
        The probe-specific ``SYSTEM_DE`` default is passed through unchanged as
        the system turn, so probe calibration stays byte-for-byte identical to
        the previous hand-built string — only the RENDERING is now shared.
        """
        from auralis.tokenizer.chat_template import build_inference_prompt

        return build_inference_prompt(
            [{"role": "user", "content": user.strip()}],
            default_system=system,
        )

    def split_continuation(self, prompt: str, continuation: str) -> tuple[list[int], list[int]]:
        """Tokenize prompt and continuation robustly against boundary merges.

        Encodes the joined string as ground truth and uses the longest shared
        prefix with the prompt's own encoding as the prompt portion, so the
        continuation ids are exactly the tokens that follow the prompt in the
        real tokenization.
        """
        prompt_ids = self.encode(prompt)
        full_ids = self.encode(prompt + continuation)
        n = 0
        for a, b in zip(prompt_ids, full_ids):
            if a != b:
                break
            n += 1
        cont = full_ids[n:]
        if not cont:                       # degenerate: continuation merged away
            cont = full_ids[len(prompt_ids):] or full_ids[-1:]
            return full_ids[: len(full_ids) - len(cont)], cont
        return full_ids[:n], cont

    def probe_ids(self, probe: MarginProbe) -> tuple[list[int], list[int], list[int]]:
        """Return (prompt_ids, correct_ids, wrong_ids) for a margin probe."""
        prompt_text = (
            self.chat_prompt(probe.prompt) if probe.prompt_style == "chat" else probe.prompt
        )
        p1, correct = self.split_continuation(prompt_text, probe.correct)
        _, wrong = self.split_continuation(prompt_text, probe.wrong)
        return p1, correct, wrong


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class ModelAdapter:
    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cuda",
        autocast_dtype: torch.dtype = torch.bfloat16,
    ):
        self.model = model
        self.device = device
        self.autocast_dtype = autocast_dtype

    def train_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        labels = batch["labels"].to(self.device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
            out = self.model(input_ids=input_ids, labels=labels)
        return out["loss"]


# ---------------------------------------------------------------------------
# Stage data sources
# ---------------------------------------------------------------------------
def build_stage_loader(
    data_spec: dict[str, Any],
    *,
    tokenizer: TokenizerAdapter,
    seq_length: int,
    batch_size: int,
    seed: int = 42,
) -> Iterator[dict[str, torch.Tensor]]:
    """Build an infinite batch iterator for a stage from its ``data`` spec.

    Supported ``kind`` values:

    - ``raw_text`` / ``mixed``: wraps the existing ``MixedDataLoader`` over the
      tokenized .bin corpora. Spec keys: ``data_dir``, ``mix_ratios``.
    - ``sft_jsonl``: formatted chat examples from a JSONL with a ``text`` field.
      Spec keys: ``paths`` (list), ``assistant_only`` (bool, default True).
    """
    kind = str(data_spec.get("kind", "raw_text"))
    if kind in ("raw_text", "mixed"):
        from auralis.training.dataset import MixedDataLoader

        loader = MixedDataLoader(
            data_dir=data_spec["data_dir"],
            mix_ratios=data_spec["mix_ratios"],
            batch_size=batch_size,
            seq_length=seq_length,
            seed=seed,
        )
        return iter(loader)
    if kind == "sft_jsonl":
        return _sft_jsonl_iterator(
            paths=[Path(p) for p in data_spec["paths"]],
            tokenizer=tokenizer,
            seq_length=seq_length,
            batch_size=batch_size,
            assistant_only=bool(data_spec.get("assistant_only", True)),
            seed=seed,
        )
    raise ValueError(f"unknown stage data kind: {kind}")


_ASSISTANT_MARKER = "<|assistant|>\n"


def _sft_jsonl_iterator(
    *,
    paths: list[Path],
    tokenizer: TokenizerAdapter,
    seq_length: int,
    batch_size: int,
    assistant_only: bool,
    seed: int,
) -> Iterator[dict[str, torch.Tensor]]:
    import random

    rng = random.Random(seed)
    examples: list[str] = []
    for p in paths:
        with p.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                txt = obj.get("text") or obj.get("prompt")
                if txt:
                    examples.append(txt)
    if not examples:
        raise ValueError(f"no examples found in {paths}")

    def encode_one(text: str) -> tuple[list[int], list[int]]:
        ids = tokenizer.encode(text)[:seq_length]
        labels = list(ids)
        if assistant_only and _ASSISTANT_MARKER in text:
            # Train ONLY on assistant responses; mask everything else. Each
            # assistant turn's target runs from just after "<|assistant|>\n" to
            # the end of that turn — the next "<|end|>" (inclusive, so the model
            # learns to stop) or end-of-text. This is multi-turn safe: the earlier
            # version masked only up to the FIRST marker, so in a
            # user->assistant->user->assistant example every later <|user|> turn
            # was trained as a generation target. Char spans map to token indices
            # via cumulative-prefix encode lengths (robust to boundary merges
            # within a single token; clamped to len(ids)).
            end_marker = "<|end|>"
            labels = [-100] * len(ids)
            search = 0
            while True:
                a = text.find(_ASSISTANT_MARKER, search)
                if a == -1:
                    break
                cs = a + len(_ASSISTANT_MARKER)
                e = text.find(end_marker, cs)
                ce = (e + len(end_marker)) if e != -1 else len(text)
                t_start = len(tokenizer.encode(text[:cs]))
                t_end = len(tokenizer.encode(text[:ce]))
                for i in range(t_start, min(t_end, len(ids))):
                    labels[i] = ids[i]
                search = ce
        return ids, labels

    pad = tokenizer.pad_id
    while True:
        rng.shuffle(examples)
        for i in range(0, len(examples) - batch_size + 1, batch_size):
            chunk = examples[i : i + batch_size]
            enc = [encode_one(t) for t in chunk]
            maxlen = max(len(ids) for ids, _ in enc)
            inp = torch.full((len(enc), maxlen), pad, dtype=torch.long)
            lab = torch.full((len(enc), maxlen), -100, dtype=torch.long)
            for r, (ids, labels) in enumerate(enc):
                inp[r, : len(ids)] = torch.tensor(ids, dtype=torch.long)
                lab[r, : len(labels)] = torch.tensor(labels, dtype=torch.long)
            yield {"input_ids": inp, "labels": lab}


__all__ = ["TokenizerAdapter", "ModelAdapter", "build_stage_loader", "SYSTEM_DE"]
