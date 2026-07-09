"""Parity of fused_linear_cross_entropy vs the plain materialised loss.

The fused path forms the [N, V] logits one chunk at a time (forward + backward)
so the full logits tensor never exists. It must match
``F.cross_entropy(F.linear(hidden, weight), labels, ignore_index=-100)`` on both
the loss scalar AND the gradients w.r.t. hidden and weight — up to chunk
reduction order (fp32: ~1e-5) and bf16 rounding (compared against the fp32
ground truth). Also covers ignore_index, the all-ignored clamp, a frozen weight
(no grad requested), chunk-size invariance, and 3-D [B, L, D] inputs.
"""

import sys

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, "/workspace/v2data/src")
from auralis.model.fused_cross_entropy import fused_linear_cross_entropy  # noqa: E402


def _ref(h, w, lbl, ignore=-100):
    return F.cross_entropy(
        F.linear(h, w).reshape(-1, w.shape[0]), lbl.reshape(-1), ignore_index=ignore
    )


def _rel(a, b):
    return (a - b).norm().item() / (b.norm().item() + 1e-12)


def _make(seed=0, N=300, V=4000, D=96):
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(N, D, generator=g)
    w = torch.randn(V, D, generator=g) * 0.02
    lbl = torch.randint(0, V, (N,), generator=g)
    lbl[::7] = -100  # sprinkle ignore_index
    return h, w, lbl


@pytest.mark.parametrize("chunk", [1, 32, 128, 100_000])
def test_fp32_loss_and_grads_match(chunk):
    h, w, lbl = _make()
    hr, wr = h.clone().requires_grad_(True), w.clone().requires_grad_(True)
    Lr = _ref(hr, wr, lbl)
    Lr.backward()

    hf, wf = h.clone().requires_grad_(True), w.clone().requires_grad_(True)
    Lf = fused_linear_cross_entropy(hf, wf, lbl, ignore_index=-100, chunk_size=chunk)
    Lf.backward()

    assert torch.allclose(Lf, Lr, rtol=1e-4, atol=1e-5), (Lf.item(), Lr.item())
    assert _rel(hf.grad, hr.grad) < 1e-4
    assert _rel(wf.grad, wr.grad) < 1e-4


def test_3d_input_matches():
    B, L, V, D = 4, 50, 4000, 96
    g = torch.Generator().manual_seed(1)
    h = torch.randn(B, L, D, generator=g)
    w = torch.randn(V, D, generator=g) * 0.02
    lbl = torch.randint(0, V, (B, L), generator=g)
    lbl[:, ::5] = -100

    hr, wr = h.clone().requires_grad_(True), w.clone().requires_grad_(True)
    Lr = _ref(hr, wr, lbl)
    Lr.backward()
    hf, wf = h.clone().requires_grad_(True), w.clone().requires_grad_(True)
    Lf = fused_linear_cross_entropy(hf, wf, lbl, chunk_size=64)
    Lf.backward()

    assert torch.allclose(Lf, Lr, rtol=1e-4, atol=1e-5)
    assert _rel(hf.grad, hr.grad) < 1e-4
    assert _rel(wf.grad, wr.grad) < 1e-4


def test_frozen_weight_returns_no_weight_grad():
    h, w, lbl = _make()
    hr, wr = h.clone().requires_grad_(True), w.clone().requires_grad_(True)
    _ref(hr, wr, lbl).backward()

    hf = h.clone().requires_grad_(True)
    wf = w.clone()  # frozen: requires_grad=False
    Lf = fused_linear_cross_entropy(hf, wf, lbl, chunk_size=64)
    Lf.backward()  # must not raise despite weight needing no grad
    assert wf.grad is None
    assert _rel(hf.grad, hr.grad) < 1e-4


def test_all_ignored_is_finite_zero():
    h, w, _ = _make()
    lbl = torch.full((h.shape[0],), -100)
    hf, wf = h.clone().requires_grad_(True), w.clone().requires_grad_(True)
    Lf = fused_linear_cross_entropy(hf, wf, lbl, chunk_size=64)
    Lf.backward()
    # clamp(n_valid, min=1) keeps it finite (0.0) instead of nan
    assert torch.isfinite(Lf).all() and abs(Lf.item()) < 1e-6
    assert torch.isfinite(hf.grad).all() and hf.grad.abs().max().item() == 0.0
    assert torch.isfinite(wf.grad).all() and wf.grad.abs().max().item() == 0.0


def test_bf16_matches_fp32_ground_truth():
    h, w, lbl = _make()
    hgt, wgt = h.clone().requires_grad_(True), w.clone().requires_grad_(True)
    Lgt = _ref(hgt, wgt, lbl)
    Lgt.backward()

    hb = h.bfloat16().clone().requires_grad_(True)
    wb = w.bfloat16().clone().requires_grad_(True)
    Lb = fused_linear_cross_entropy(hb, wb, lbl, chunk_size=64)
    Lb.backward()

    assert torch.isfinite(Lb).all()
    assert torch.allclose(Lb.float(), Lgt.float(), rtol=0.02, atol=0.03), (Lb.item(), Lgt.item())
    assert torch.isfinite(hb.grad).all() and torch.isfinite(wb.grad).all()
    assert _rel(hb.grad.float(), hgt.grad) < 0.05
    assert _rel(wb.grad.float(), wgt.grad) < 0.05


# ---------------------------------------------------------------------------
# Model-level integration: HelixModel.forward with fused CE ON vs OFF must
# agree on loss / loss_main / loss_mtp AND on the parameter gradients. Exercises
# the real wiring: next-token shift, tied-embedding weight selection, MTP heads,
# and the flag branch. (The fused branch returns logits=None by design — only
# losses/grads are compared.)
# ---------------------------------------------------------------------------
from pathlib import Path  # noqa: E402

from auralis.model import AuralisConfig, HelixModel, build_model  # noqa: E402

_REPO = Path(__file__).resolve().parents[2]
_CFG_100M = _REPO / "configs" / "model" / "helix_v2_100m.yaml"


def _loss_and_grads(model, x, y):
    model.zero_grad(set_to_none=True)
    out = model(input_ids=x, labels=y)
    out["loss"].backward()
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
    return out, grads


def _compare_fused_vs_plain(model):
    model.train()
    torch.manual_seed(3)
    x = torch.randint(0, model.config.vocab_size, (2, 24))
    y = torch.randint(0, model.config.vocab_size, (2, 24))
    y[:, ::6] = -100

    model.fused_cross_entropy_disable()
    out_p, g_p = _loss_and_grads(model, x, y)
    model.fused_cross_entropy_enable(chunk_size=8)
    out_f, g_f = _loss_and_grads(model, x, y)

    assert out_f["logits"] is None  # loss-only path skips the [N, V] logits
    assert torch.allclose(out_f["loss"], out_p["loss"], rtol=1e-4, atol=1e-5), (
        out_f["loss"].item(), out_p["loss"].item(),
    )
    assert torch.allclose(out_f["loss_main"], out_p["loss_main"], rtol=1e-4, atol=1e-5)
    if out_p["loss_mtp"] is not None:
        assert torch.allclose(out_f["loss_mtp"], out_p["loss_mtp"], rtol=1e-4, atol=1e-5)
    # Gradients must match on every trained parameter (incl. tied embedding).
    assert g_p.keys() == g_f.keys()
    worst = max(_rel(g_f[n], g_p[n]) for n in g_p)
    assert worst < 1e-4, f"max grad rel-err {worst:.2e}"


def test_model_forward_fused_matches_plain_no_mtp():
    torch.manual_seed(0)
    model = build_model(_CFG_100M)  # production shape: tied embeddings, mtp off
    assert model.mtp_heads is not None and len(model.mtp_heads) == 0
    _compare_fused_vs_plain(model)


def test_model_forward_fused_matches_plain_with_mtp():
    cfg = AuralisConfig.from_yaml(_CFG_100M)
    cfg.mtp.enabled = True
    cfg.mtp.n_heads = 2
    cfg.mtp.loss_weight = 0.5
    torch.manual_seed(0)
    model = HelixModel(cfg)
    assert len(model.mtp_heads) == 2
    _compare_fused_vs_plain(model)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
