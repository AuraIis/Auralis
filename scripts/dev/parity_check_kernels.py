"""Kernel-vs-native parity check for GLA and Sparse-Attn layers.

Builds one layer, runs the same inputs with kernel env on/off, compares.
Run inside the auralis-blackwell container:
    AURALIS_PARITY=1 python scripts/dev/parity_check_kernels.py
"""
import os, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import torch

torch.manual_seed(0)
dev = "cuda"
dt = torch.float32  # use fp32 for the math comparison; kernel may cast internally

from auralis.model.layers.gla_layer import GLALayer
from auralis.model.layers.sparse_attn_layer import SparseAttentionLayer
from auralis.model.utils.rotary import RotaryEmbedding, apply_rotary_pos_emb

def rel(a, b):
    return ((a - b).abs().max().item(), (a - b).norm().item() / (b.norm().item() + 1e-9))

# ---------------- GLA -----------------
B, L, H, D = 2, 256, 4, 64
layer = GLALayer(d_model=H * D, n_heads=H, d_head=D).to(dev).to(dt)
x = torch.randn(B, L, H * D, device=dev, dtype=dt) * 0.5

os.environ["AURALIS_USE_GLA_KERNEL"] = "0"
os.environ["AURALIS_USE_CUDA_KERNELS"] = "0"
with torch.no_grad():
    out_native, _ = layer(x)

os.environ["AURALIS_USE_GLA_KERNEL"] = "1"
with torch.no_grad():
    out_fla, _ = layer(x)

m, r = rel(out_native, out_fla)
print(f"GLA   native vs fla : max_abs={m:.6f} rel={r:.6f}")

# also: native with decay broadcast on KEY dim instead of value dim
import torch.nn.functional as F
def native_keydecay(layer, x):
    B, L, _ = x.shape
    H, D = layer.n_heads, layer.d_head
    q = layer.q_proj(x).view(B, L, H, D) * (D ** -0.5)
    k = layer.k_proj(x).view(B, L, H, D)
    v = layer.v_proj(x).view(B, L, H, D)
    g_out = torch.sigmoid(layer.g_proj(x).view(B, L, H, D))
    log_alpha = -F.softplus(-layer.alpha_proj(x).view(B, L, H, D))
    alpha = torch.exp(log_alpha)
    S = torch.zeros(B, H, D, D, device=x.device, dtype=x.dtype)
    outs = []
    for t in range(L):
        a_t = alpha[:, t].unsqueeze(-1)          # [B,H,D,1] -> decay per KEY channel
        S = a_t * S + torch.einsum("bhd,bhe->bhde", k[:, t], v[:, t])
        outs.append(torch.einsum("bhd,bhde->bhe", q[:, t], S))
    out = torch.stack(outs, dim=1) * g_out
    return layer.out_proj(out.reshape(B, L, H * D))

with torch.no_grad():
    out_fix = native_keydecay(layer, x)
m2, r2 = rel(out_fix, out_fla)
print(f"GLA   keydecay vs fla: max_abs={m2:.6f} rel={r2:.6f}")

# ---------------- Sparse attn ----------------
B, L, H, D, W = 2, 512, 4, 64, 128
sa = SparseAttentionLayer(d_model=H * D, n_heads=H, d_head=D, window_size=W,
                          global_tokens=0, use_rope=True).to(dev).to(torch.bfloat16)
rope = RotaryEmbedding(dim=D, max_seq_len=L).to(dev)
cos, sin = rope(L, device=torch.device(dev), dtype=torch.float32)
xb = torch.randn(B, L, H * D, device=dev, dtype=torch.bfloat16)

os.environ["AURALIS_USE_FLASH_ATTN"] = "0"
with torch.no_grad():
    o_nat, _ = sa(xb, rope=(cos.bfloat16(), sin.bfloat16()))
os.environ["AURALIS_USE_FLASH_ATTN"] = "1"
with torch.no_grad():
    o_fla, _ = sa(xb, rope=(cos.bfloat16(), sin.bfloat16()))
m3, r3 = rel(o_nat.float(), o_fla.float())
print(f"SPARSE native vs flash: max_abs={m3:.6f} rel={r3:.6f}  (bf16 noise ~1e-2 max_abs ok)")
