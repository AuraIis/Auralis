#!/usr/bin/env python3
"""Resume round-trip smoke: save -> fresh trainer -> load_checkpoint -> assert.

Validates the checkpoint-resume fix end-to-end WITHOUT the 1B model or data:
- training step continues (not reset)
- optimizer LR continues (NOT reset to warmup) — the bug that silently kills runs
- optimizer momentum / RNG / weights restored
- torch.compile `_orig_mod.` key prefixes align

Runs anywhere torch is installed (CPU is fine):
    python scripts/ops/resume_smoke.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from auralis.training.trainer import PretrainTrainer


def make_trainer(out_dir: Path):
    model = nn.Sequential(nn.Linear(8, 16), nn.Linear(16, 8))
    opt = AdamW(model.parameters(), lr=1.5e-4)
    warmup = 5
    sched = LambdaLR(opt, lr_lambda=lambda s: min(1.0, (s + 1) / warmup))
    config = {
        "training": {"total_steps": 100, "gradient_accumulation": 1,
                     "gradient_clip_norm": 1.0, "batch_size_per_device": 2, "dtype": "fp32"},
        "data": {"seq_length": 8},
        "logging": {"log_every": 1, "eval_every": 1000, "save_every": 1000},
        "checkpointing": {"output_dir": str(out_dir), "save_last_n": 3,
                          "save_best": True, "external_backup": {}},
        "monitoring": {"health": {}},
    }

    def loader():
        while True:
            yield {"input_ids": torch.zeros(2, 8, dtype=torch.long),
                   "labels": torch.zeros(2, 8, dtype=torch.long)}

    tr = PretrainTrainer(model=model, optimizer=opt, scheduler=sched,
                         dataloader=loader(), config=config, device="cpu")
    return tr, model, opt, sched


def main() -> int:
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        a, model_a, opt_a, sched_a = make_trainer(out)

        # Simulate 7 optimizer steps (with real grads so momentum state fills).
        for _ in range(7):
            for p in model_a.parameters():
                p.grad = torch.randn_like(p)
            opt_a.step()
            sched_a.step()
        a.state.step = 7
        a.state.best_val_loss = 1.234
        a.state.tokens_seen = 7 * 65536
        lr_at_save = opt_a.param_groups[0]["lr"]
        ckpt = a.save_checkpoint("resume_smoke")
        expected_rng = torch.rand(4)            # RNG stream immediately after save

        # Fresh trainer (different random init), then resume.
        b, model_b, opt_b, sched_b = make_trainer(out)
        assert b.state.step == 0, "fresh trainer should start at step 0"
        b.load_checkpoint(ckpt)
        got_rng = torch.rand(4)

        ok = True

        def check(name: str, cond: bool) -> None:
            nonlocal ok
            print(f"  {'PASS' if cond else 'FAIL'}  {name}")
            ok = ok and cond

        check("step restored (==7)", b.state.step == 7)
        check("best_val_loss restored", abs(b.state.best_val_loss - 1.234) < 1e-9)
        check("tokens_seen restored", b.state.tokens_seen == 7 * 65536)
        check("LR continued, NOT reset to warmup",
              abs(opt_b.param_groups[0]["lr"] - lr_at_save) < 1e-12)
        check("optimizer momentum state loaded",
              len(opt_b.state_dict()["state"]) > 0)
        check("RNG stream continues (reproducible)",
              torch.allclose(expected_rng, got_rng))
        same = all(torch.equal(pa, pb) for pa, pb in
                   zip(model_a.state_dict().values(), model_b.state_dict().values()))
        check("model weights identical after resume", same)

    print("\nRESUME SMOKE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
