"""LoRA / DoRA adapters for Helix v2 — modular skills on a FROZEN base.

Wraps the Linear projections of the hybrid stack (Mamba in/out, GLA q/k/v/g/out,
Attn q/k/v/out, FFN gate/up/down). The SSM-internal projections (x_proj, dt_proj,
alpha_proj) are deliberately NOT targeted — they are the state-space dynamics.

Why: full fine-tuning on the 0.9B catastrophically forgets prior skills (tool-use,
facts) within ~50 steps. A frozen base + small adapter cannot overwrite the base,
and the adapter strength is tunable at inference.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "g_proj", "out_proj",
                   "in_proj", "gate_proj", "up_proj", "down_proj")
ADAPTER_KEYS = ("lora_A", "lora_B", "magnitude")


class LoRALinear(nn.Module):
    """y = W0 x + (alpha/r) * (B A) x   ·   W0 frozen, B init 0 -> starts as identity."""
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.r = r
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.zeros(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.scale = 1.0  # inference-time strength dial (alpha-sweep); scale=0 -> exact base

    def forward(self, x):
        delta = (self.dropout(x) @ self.lora_A.t()) @ self.lora_B.t()
        return self.base(x) + delta * (self.scaling * self.scale)


class DoRALinear(nn.Module):
    """Weight-decomposed LoRA. W = W0 + (alpha/r) B A ; y = m * (W / ||W||_row) x + b.
    m (per output row) inits to ||W0||_row -> starts as exact identity."""
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        W0 = base.weight.detach()
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.r = r
        self.scaling = alpha / r
        out_f, in_f = W0.shape
        self.lora_A = nn.Parameter(torch.zeros(r, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.magnitude = nn.Parameter(W0.norm(dim=1))  # (out,)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.scale = 1.0  # inference dial (note: DoRA scale=0 != exact base, magnitude is trained)

    def forward(self, x):
        W = self.base.weight + (self.lora_B @ self.lora_A) * (self.scaling * self.scale)
        norm = W.norm(dim=1, keepdim=True) + 1e-8
        W_eff = (self.magnitude.unsqueeze(1) / norm) * W
        return F.linear(self.dropout(x), W_eff.to(x.dtype), self.base.bias)


def _parent(model: nn.Module, dotted: str):
    parts = dotted.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def inject_adapters(model: nn.Module, targets=DEFAULT_TARGETS, r: int = 16,
                    alpha: float = 32.0, dropout: float = 0.0, kind: str = "dora",
                    exclude=("inner",)) -> list[str]:
    """Replace matching nn.Linear modules with LoRA/DoRA wrappers in place.

    `exclude`: skip any module whose full path contains one of these substrings.
    Default excludes '.inner.' = the Mamba `mamba_ssm` fused kernel, which reads
    `out_proj.weight` DIRECTLY (bypassing module forward) — wrapping it both crashes
    and would have no effect. So adapters live on GLA/Attn/FFN (the mixing + MLP
    surface); the SSM state path stays un-adapted (acceptable + standard for PEFT)."""
    Cls = DoRALinear if kind == "dora" else LoRALinear
    injected = []
    for name, module in list(model.named_modules()):
        if any(ex in name for ex in exclude):
            continue
        if isinstance(module, nn.Linear) and name.split(".")[-1] in targets:
            parent, attr = _parent(model, name)
            setattr(parent, attr, Cls(module, r, alpha, dropout))
            injected.append(name)
    return injected


def freeze_base(model: nn.Module) -> tuple[int, int]:
    """requires_grad only on adapter params. Returns (trainable, total) param counts."""
    train = total = 0
    for n, p in model.named_parameters():
        is_adapter = any(k in n for k in ADAPTER_KEYS)
        p.requires_grad_(is_adapter)
        total += p.numel()
        if is_adapter:
            train += p.numel()
    return train, total


def adapter_state_dict(model: nn.Module) -> dict:
    return {n: p.detach().cpu() for n, p in model.named_parameters()
            if any(k in n for k in ADAPTER_KEYS)}


def load_adapter_state_dict(model: nn.Module, sd: dict):
    own = dict(model.named_parameters())
    missing = [n for n in sd if n not in own]
    if missing:
        raise KeyError(f"adapter keys not in model: {missing[:3]}...")
    for n, v in sd.items():
        own[n].data.copy_(v.to(own[n].device, own[n].dtype))


def set_adapter_scale(model: nn.Module, scale: float) -> int:
    """Set the inference-time strength dial on all adapters (alpha-sweep).
    scale=0 -> exact base (LoRA), scale=1 -> trained strength. Returns count set."""
    n = 0
    for m in model.modules():
        if isinstance(m, (LoRALinear, DoRALinear)):
            m.scale = scale
            n += 1
    return n


def enable_input_require_grads(model: nn.Module):
    """Make the input-embedding output require grad, so gradient checkpointing works
    with a FROZEN base. Without this, no checkpoint input requires grad -> activations
    are not freed -> OOM (the standard PEFT trick). Returns the hook handle."""
    emb = getattr(model, "embedding", None) or getattr(model, "embed_tokens", None)
    if emb is None:
        return None

    def _hook(module, inp, out):
        out.requires_grad_(True)

    return emb.register_forward_hook(_hook)


def _selftest():
    """Lightweight plumbing test on a mock model (no big model needed)."""
    class Mock(nn.Module):
        def __init__(s):
            super().__init__()
            s.q_proj = nn.Linear(32, 32, bias=False)
            s.out_proj = nn.Linear(32, 32, bias=False)
            s.keep = nn.Linear(32, 32)  # not a target -> stays frozen-able

        def forward(s, x):
            return s.out_proj(s.q_proj(x))

    for kind in ("lora", "dora"):
        m = Mock()
        x = torch.randn(4, 32)
        y0 = m(x)
        inj = inject_adapters(m, r=4, alpha=8, kind=kind)
        assert set(inj) == {"q_proj", "out_proj"}, inj
        y1 = m(x)
        # B init 0 -> adapter starts as IDENTITY (output unchanged)
        assert torch.allclose(y0, y1, atol=1e-5), f"{kind}: not identity at init"
        train, total = freeze_base(m)
        assert 0 < train < total
        sd = adapter_state_dict(m)
        assert all(any(k in n for k in ADAPTER_KEYS) for n in sd)
        load_adapter_state_dict(m, sd)
        print(f"  [{kind}] OK  injected={inj}  trainable={train}/{total} ({100*train/total:.1f}%)  adapter-keys={len(sd)}")
    print("=== LoRA/DoRA SELFTEST PASS ===")


if __name__ == "__main__":
    _selftest()
