"""End-to-end HelixModel tests: build, forward, backward, parameter counts."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from auralis.model import AuralisConfig, HelixModel, build_model

REPO = Path(__file__).resolve().parents[2]
CFG_100M = REPO / "configs" / "model" / "helix_v2_100m.yaml"


@pytest.fixture(scope="module")
def small_model() -> HelixModel:
    torch.manual_seed(0)
    return build_model(CFG_100M)


# ---------- Build ----------

def test_model_builds(small_model: HelixModel):
    assert isinstance(small_model, HelixModel)
    assert len(small_model.blocks) == small_model.config.n_layers


def test_parameter_count_in_100m_range(small_model: HelixModel):
    n = small_model.count_parameters()
    # Actual count should be within ~10% of the a-priori estimate.
    est = small_model.config.estimate_parameters()["total"]
    assert 0.9 * est <= n <= 1.1 * est


def test_tied_embeddings_means_no_lm_head(small_model: HelixModel):
    assert small_model.config.advanced.tie_embeddings is True
    assert small_model.lm_head is None


# ---------- Forward ----------

def test_forward_output_shape(small_model: HelixModel):
    small_model.eval()
    x = torch.randint(0, small_model.config.vocab_size, (2, 16))
    with torch.no_grad():
        out = small_model(x)
    assert out["logits"].shape == (2, 16, small_model.config.vocab_size)
    assert out["loss"] is None


def test_forward_loss_when_labels_given(small_model: HelixModel):
    small_model.eval()
    x = torch.randint(0, small_model.config.vocab_size, (2, 16))
    y = torch.randint(0, small_model.config.vocab_size, (2, 16))
    with torch.no_grad():
        out = small_model(x, labels=y)
    assert out["loss"] is not None
    assert torch.isfinite(out["loss"])
    # Cross-entropy against random targets with a 200k vocab: ln(200000) ≈ 12.2
    assert 5.0 < out["loss"].item() < 20.0


def test_forward_no_nan(small_model: HelixModel):
    small_model.eval()
    x = torch.randint(0, small_model.config.vocab_size, (1, 8))
    with torch.no_grad():
        out = small_model(x)
    assert torch.isfinite(out["logits"]).all()


# ---------- Backward ----------

def test_backward_creates_gradients(small_model: HelixModel):
    small_model.train()
    x = torch.randint(0, small_model.config.vocab_size, (2, 8))
    y = torch.randint(0, small_model.config.vocab_size, (2, 8))
    out = small_model(x, labels=y)
    out["loss"].backward()
    grads = [p.grad for p in small_model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert any(g.abs().sum() > 0 for g in grads)


def test_backward_no_inf_nan_gradients(small_model: HelixModel):
    small_model.zero_grad()
    x = torch.randint(0, small_model.config.vocab_size, (2, 8))
    y = torch.randint(0, small_model.config.vocab_size, (2, 8))
    out = small_model(x, labels=y)
    out["loss"].backward()
    for name, p in small_model.named_parameters():
        if p.grad is None:
            continue
        assert torch.isfinite(p.grad).all(), f"Non-finite gradient in {name}"


# ---------- Layer type positions ----------

def test_layer_types_match_config(small_model: HelixModel):
    """The block's wired layer type must match the config's spec."""
    for idx, block in enumerate(small_model.blocks):
        assert block.layer_config.type == small_model.config.layers[idx].type


def test_mamba_early_gla_mid_sparse_late(small_model: HelixModel):
    types = [b.layer_config.type for b in small_model.blocks]
    assert types[0] == "mamba"
    assert types[-1] == "sparse_attention"
    assert "gla" in types
