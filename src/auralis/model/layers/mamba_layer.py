"""Mamba-2 layer with optional CUDA kernel from ``mamba_ssm``.

Two back-ends, same ``forward(x) -> (out, state)`` contract:

- **native** (default on CPU / when ``AURALIS_USE_CUDA_KERNELS != "1"``):
  pure-PyTorch selective-scan reference. Slow, correct, portable.
- **mamba_ssm** (auto-selected when env flag is on and CUDA is available):
  wraps ``mamba_ssm.Mamba2`` which uses the official CUDA kernel
  (parallel selective-scan) and optional ``causal-conv1d``.

The parameter layout differs between the two back-ends — swap is only safe
**before training** (fresh init). If you trained with one back-end, reload
with the same one.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional CUDA back-end
try:
    from mamba_ssm import Mamba2 as _Mamba2SSM
    _MAMBA_SSM_AVAILABLE = True
except Exception:
    _Mamba2SSM = None
    _MAMBA_SSM_AVAILABLE = False


def _cuda_kernels_enabled() -> bool:
    """Per-layer opt-in. AURALIS_USE_MAMBA_KERNEL=1 enables mamba_ssm.
    AURALIS_USE_CUDA_KERNELS=1 enables ALL kernel back-ends at once.

    Note: on Blackwell (sm_120) Triton inside mamba_ssm sometimes fails to
    compile with cu128. Leave AURALIS_USE_MAMBA_KERNEL unset on Blackwell
    until upstream catches up.
    """
    if os.environ.get("AURALIS_USE_MAMBA_KERNEL", "") == "1":
        return True
    return os.environ.get("AURALIS_USE_CUDA_KERNELS", "0") == "1"


def _native_ssd_enabled() -> bool:
    """Opt-in for the pure-PyTorch Mamba-2 SSD back-end.

    This back-end mirrors ``mamba_ssm.Mamba2``'s parameter tree
    (``inner.in_proj / conv1d / A_log / D / dt_bias / norm.weight / out_proj``)
    and reproduces its forward math WITHOUT any CUDA kernel. It exists so a
    checkpoint trained with the mamba_ssm CUDA back-end (param layout under
    ``_impl.inner.*``) can run on a machine that has no mamba_ssm/Triton — the
    weights load with ZERO remapping because the key names already match.

    Set ``AURALIS_MAMBA_NATIVE_SSD=1`` to select it. It is the back-end the
    kernel->native converter (scripts/dev/convert_kernel_to_native.py) targets.
    """
    return os.environ.get("AURALIS_MAMBA_NATIVE_SSD", "") == "1"


# ---------------------------------------------------------------------------
# Pure-PyTorch reference implementation (original)
# ---------------------------------------------------------------------------
class _Mamba2Native(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        d_conv: int = 4,
        expand_factor: int = 2,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand_factor = expand_factor
        self.d_inner = expand_factor * d_model
        self.dt_min = float(dt_min)
        self.dt_max = float(dt_max)

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner, out_channels=self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner, bias=True,
        )
        self.x_proj = nn.Linear(self.d_inner, self.d_inner + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.reset_special_parameters()

    def reset_special_parameters(self) -> None:
        """Restore Mamba-specific parameters that generic init must not zero.

        The model-wide initializer touches every ``nn.Linear`` and zeros its
        bias. ``dt_proj.bias`` is not a normal bias: it is the inverse-softplus
        parameterisation for the SSM time step, so it must be reset after the
        generic init pass.
        """
        with torch.no_grad():
            log_min = torch.log(torch.tensor(self.dt_min, dtype=torch.float32))
            log_max = torch.log(torch.tensor(self.dt_max, dtype=torch.float32))
            dt = torch.exp(torch.rand(self.d_inner, device=self.dt_proj.bias.device) * (log_max - log_min) + log_min)
            # inverse softplus: softplus(bias) == dt
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def forward(self, x, ssm_state=None):
        B, L, _ = x.shape
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)
        x_in = x_in.transpose(1, 2)
        x_in = self.conv1d(x_in)[..., :L]
        x_in = x_in.transpose(1, 2)
        x_in = F.silu(x_in)
        x_dbl = self.x_proj(x_in)
        dt, B_ssm, C_ssm = x_dbl.split([self.d_inner, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log.float())
        y, new_state = self._selective_scan(x_in, dt, A, B_ssm, C_ssm, self.D, ssm_state)
        y = y * F.silu(z)
        return self.out_proj(y), new_state

    # ------------------------------------------------------------------
    # Incremental decoding (native reference) — O(1) per token
    # ------------------------------------------------------------------
    def allocate_cache(self, batch: int, max_seqlen: int, device, dtype):
        return {
            "conv": torch.zeros(batch, self.d_inner, self.d_conv - 1, device=device, dtype=dtype),
            "h": torch.zeros(batch, self.d_inner, self.d_state, device=device, dtype=torch.float32),
        }

    def prefill(self, x, cache):
        B, L, _ = x.shape
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)
        x_raw = x_in.transpose(1, 2)                      # [B, d_inner, L] pre-conv
        # conv tail for the step path (last d_conv-1 raw inputs, zero-padded)
        pad = self.d_conv - 1
        if pad > 0:
            cache["conv"].copy_(F.pad(x_raw, (max(pad - L, 0), 0))[..., -pad:])
        x_c = F.silu(self.conv1d(x_raw)[..., :L].transpose(1, 2))
        x_dbl = self.x_proj(x_c)
        dt, B_ssm, C_ssm = x_dbl.split([self.d_inner, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log.float())
        y, h = self._selective_scan(x_c, dt, A, B_ssm, C_ssm, self.D, None)
        cache["h"].copy_(h)
        y = y * F.silu(z)
        return self.out_proj(y)

    def step(self, x, cache):
        # x: [B, 1, d_model] → single recurrent update.
        B = x.shape[0]
        xz = self.in_proj(x[:, 0])
        x_raw, z = xz.chunk(2, dim=-1)                    # [B, d_inner]
        window = torch.cat([cache["conv"], x_raw.unsqueeze(-1)], dim=-1)  # [B, d_inner, d_conv]
        cache["conv"].copy_(window[..., 1:])
        w = self.conv1d.weight.squeeze(1)                 # [d_inner, d_conv]
        x_c = F.silu((window * w).sum(-1) + self.conv1d.bias)
        x_dbl = self.x_proj(x_c)
        dt, B_ssm, C_ssm = x_dbl.split([self.d_inner, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log.float())
        dA = torch.exp(dt.float().unsqueeze(-1) * A)      # [B, d_inner, d_state]
        dB = dt.float().unsqueeze(-1) * B_ssm.float().unsqueeze(-2)
        h = dA * cache["h"] + dB * x_c.float().unsqueeze(-1)
        cache["h"].copy_(h)
        y = (h * C_ssm.float().unsqueeze(-2)).sum(-1).to(x.dtype) + x_c * self.D
        y = y * F.silu(z)
        return self.out_proj(y).unsqueeze(1)

    def _selective_scan(self, x, dt, A, Bp, Cp, D, state):
        B, L, d_inner = x.shape
        d_state = A.shape[1]
        dtype = x.dtype
        # Recurrence runs in fp32 (h drifts under bf16 on long sequences);
        # output is cast back to the input dtype. A is already fp32.
        x_f = x.float()
        dA = torch.exp(dt.float().unsqueeze(-1) * A)
        dB = dt.float().unsqueeze(-1) * Bp.float().unsqueeze(-2)
        Cp = Cp.float()
        h = (torch.zeros(B, d_inner, d_state, device=x.device, dtype=torch.float32)
             if state is None else state.to(torch.float32))
        outputs = []
        for t in range(L):
            h = dA[:, t] * h + dB[:, t] * x_f[:, t].unsqueeze(-1)
            outputs.append((h * Cp[:, t].unsqueeze(-2)).sum(dim=-1))
        y = torch.stack(outputs, dim=1).to(dtype)
        y = y + x * D
        return y, h


# ---------------------------------------------------------------------------
# mamba_ssm.Mamba2 wrapper (CUDA)
# ---------------------------------------------------------------------------
class _Mamba2CUDA(nn.Module):
    """Thin wrapper over ``mamba_ssm.Mamba2`` exposing our ``(out, state)`` API."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        d_conv: int = 4,
        expand_factor: int = 2,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
    ):
        super().__init__()
        assert _Mamba2SSM is not None
        # mamba_ssm has strict shape constraints. d_model*expand must be
        # divisible by its internal "headdim" (default 64). For the
        # configs we ship (d_model in {512, 768, 1280}, expand=2) that is
        # always satisfied.
        self.inner = _Mamba2SSM(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand_factor,
            # Forward dt bounds — Mamba2 defaults happen to match ours
            # (0.001 / 0.1), but a config tuner must not be silently ignored.
            dt_min=dt_min,
            dt_max=dt_max,
        )

    def forward(self, x, ssm_state=None):
        # mamba_ssm Mamba2 returns just the output tensor; no explicit state
        # is exposed through the simple forward path we use for training.
        # NOTE: we deliberately return None (not the input state) — the input
        # state is NOT the updated state, echoing it back is a recurrence bug.
        out = self.inner(x)
        return out, None

    # ------------------------------------------------------------------
    # Incremental decoding (cached prefill + O(1) single-token step)
    # ------------------------------------------------------------------
    def allocate_cache(self, batch: int, max_seqlen: int, device, dtype):
        from mamba_ssm.utils.generation import InferenceParams
        if self.inner.layer_idx is None:
            self.inner.layer_idx = 0  # per-layer cache → single key
        ip = InferenceParams(max_seqlen=max_seqlen, max_batch_size=batch)
        conv_state, ssm_state = self.inner.allocate_inference_cache(batch, max_seqlen, dtype=dtype)
        ip.key_value_memory_dict[self.inner.layer_idx] = (conv_state, ssm_state)
        return ip

    def prefill(self, x, cache):
        # seqlen_offset == 0 → full chunked scan; final conv/ssm states are
        # written into the cache by mamba_ssm.
        out = self.inner(x, inference_params=cache)
        cache.seqlen_offset += x.shape[1]
        return out

    def step(self, x, cache):
        # seqlen_offset > 0 → mamba_ssm single-token step (states in-place).
        out = self.inner(x, inference_params=cache)
        cache.seqlen_offset += 1
        return out


# ---------------------------------------------------------------------------
# Pure-PyTorch Mamba-2 SSD back-end (kernel-compatible parameter layout)
# ---------------------------------------------------------------------------
class _GatedRMSNorm(nn.Module):
    """RMSNorm with optional gate, matching mamba_ssm's RMSNormGated.

    norm_before_gate=False (the layout used by Mamba2 default): out =
    rmsnorm(x * silu(z)). Reduction is grouped: each group of ``group_size``
    channels is normalised independently (ngroups groups). Computed in fp32.
    Parameter name ``weight`` matches mamba_ssm so checkpoints load 1:1.
    """

    def __init__(self, dim: int, group_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.group_size = group_size
        self.eps = eps

    def forward(self, x: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        if z is not None:
            xf = xf * F.silu(z.float())
        gs = self.group_size
        xg = xf.reshape(*xf.shape[:-1], xf.shape[-1] // gs, gs)
        rstd = torch.rsqrt(xg.square().mean(-1, keepdim=True) + self.eps)
        out = (xg * rstd).reshape_as(xf) * self.weight.float()
        return out.to(dtype)


class _Mamba2SSDInner(nn.Module):
    """Parameter container whose tree matches ``mamba_ssm.Mamba2`` exactly.

    Holds: in_proj (fused [z, x, B, C, dt]), conv1d (on [x, B, C]), A_log / D /
    dt_bias (per-head scalars), norm (gated RMSNorm), out_proj. No CUDA. The
    forward reproduces the SSD recurrence verified bit-for-bit against
    mamba_ssm's own ``ssd_chunk_scan_combined_ref`` (max abs diff ~1e-7).
    """

    def __init__(self, d_model, d_state, d_conv, expand_factor, headdim=64, ngroups=1,
                 dt_min=0.001, dt_max=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = expand_factor * d_model
        self.headdim = headdim
        self.ngroups = ngroups
        self.dt_min = dt_min
        self.dt_max = dt_max
        assert self.d_inner % headdim == 0, (
            f"d_inner={self.d_inner} not divisible by headdim={headdim}")
        self.nheads = self.d_inner // headdim
        self.d_ssm = self.d_inner  # no gated-MLP split in our configs

        d_in_proj = 2 * self.d_inner + 2 * ngroups * d_state + self.nheads
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=False)
        conv_dim = self.d_ssm + 2 * ngroups * d_state
        self.conv1d = nn.Conv1d(conv_dim, conv_dim, kernel_size=d_conv,
                                groups=conv_dim, padding=d_conv - 1, bias=True)
        self.dt_bias = nn.Parameter(torch.zeros(self.nheads))
        self.A_log = nn.Parameter(torch.zeros(self.nheads))
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.norm = _GatedRMSNorm(self.d_ssm, group_size=self.d_ssm // ngroups, eps=1e-5)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        # A_log / dt_bias are bare Parameters the model-wide scaled-normal init
        # does NOT touch (it only matches Embedding/Linear/Conv1d). Left at zero
        # they give A=-1 and dt_bias=0 for every head — degenerate SSM decay that
        # diverges from the mamba_ssm / _Mamba2Native backends this class must
        # match. Warm-start them here (mamba_ssm-style); checkpoint loads simply
        # overwrite them afterward, so this only affects the from-scratch path.
        self.reset_special_parameters()

    def reset_special_parameters(self) -> None:
        with torch.no_grad():
            # A: deterministic init A = 1..nheads -> A_log = log(A). (Stock
            # mamba_ssm samples A ~ Uniform[1,16]; any reasonable A-init is fine
            # here — this runs only from-scratch and a ckpt-load overwrites it.)
            A = torch.arange(1, self.nheads + 1, dtype=torch.float32,
                             device=self.A_log.device)
            self.A_log.copy_(torch.log(A))
            # dt_bias: inverse-softplus of dt sampled log-uniform in [dt_min, dt_max]
            # (same parameterisation as _Mamba2Native.dt_proj.bias).
            log_min = torch.log(torch.tensor(self.dt_min, dtype=torch.float32))
            log_max = torch.log(torch.tensor(self.dt_max, dtype=torch.float32))
            dt = torch.exp(torch.rand(self.nheads, device=self.dt_bias.device)
                           * (log_max - log_min) + log_min)
            self.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def _split(self, u):
        zxbcdt = self.in_proj(u)
        z, xBC, dt = torch.split(
            zxbcdt,
            [self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1,
        )
        return z, xBC, dt

    def _conv(self, xBC, L):
        # causal depthwise conv → silu, matching mamba_ssm activation="silu"
        xc = self.conv1d(xBC.transpose(1, 2))[..., :L].transpose(1, 2)
        return F.silu(xc)

    def _scan(self, x, dt_raw, B, C):
        """Sequential SSD recurrence (fp32). Returns y [B, L, d_ssm] incl. D skip."""
        Bsz, L, _ = x.shape
        H, P, N, G = self.nheads, self.headdim, self.d_state, self.ngroups
        A = -torch.exp(self.A_log.float())                       # [H]
        dt = F.softplus(dt_raw.float() + self.dt_bias.float())   # [B, L, H]
        xs = x.reshape(Bsz, L, H, P).float()
        Bf = B.reshape(Bsz, L, G, N).float()
        Cf = C.reshape(Bsz, L, G, N).float()
        # ngroups=1 → broadcast the single group across all heads
        rep = H // G
        h = torch.zeros(Bsz, H, P, N, device=x.device, dtype=torch.float32)
        outs = []
        for t in range(L):
            dA = torch.exp(dt[:, t] * A)                         # [B, H]
            Bt = Bf[:, t].repeat_interleave(rep, dim=1)          # [B, H, N]
            Ct = Cf[:, t].repeat_interleave(rep, dim=1)          # [B, H, N]
            dBx = torch.einsum("bh,bhn,bhp->bhpn", dt[:, t], Bt, xs[:, t])
            h = h * dA[:, :, None, None] + dBx
            yt = torch.einsum("bhpn,bhn->bhp", h, Ct) + self.D.float()[None, :, None] * xs[:, t]
            outs.append(yt)
        y = torch.stack(outs, dim=1).reshape(Bsz, L, self.d_ssm)
        return y, h

    def forward(self, u, ssm_state=None):
        L = u.shape[1]
        z, xBC, dt = self._split(u)
        xc = self._conv(xBC, L)
        x, B, C = torch.split(
            xc, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        y, _ = self._scan(x, dt, B, C)
        y = self.norm(y, z)                                      # gated RMSNorm
        return self.out_proj(y.to(u.dtype))

    # ---- incremental decode ----
    def allocate_cache(self, batch, max_seqlen, device, dtype):
        conv_dim = self.d_ssm + 2 * self.ngroups * self.d_state
        return {
            "conv": torch.zeros(batch, conv_dim, self.d_conv, device=device, dtype=dtype),
            "h": torch.zeros(batch, self.nheads, self.headdim, self.d_state,
                             device=device, dtype=torch.float32),
        }

    def prefill(self, u, cache):
        L = u.shape[1]
        z, xBC, dt = self._split(u)
        # seed conv state with the last d_conv raw inputs (pre-activation)
        xBC_t = xBC.transpose(1, 2)
        cache["conv"].copy_(F.pad(xBC_t, (self.d_conv - xBC_t.shape[-1], 0))[..., -self.d_conv:]
                            if xBC_t.shape[-1] < self.d_conv else xBC_t[..., -self.d_conv:])
        xc = F.silu(self.conv1d(xBC_t)[..., :L].transpose(1, 2))
        x, B, C = torch.split(
            xc, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        y, h = self._scan(x, dt, B, C)
        cache["h"].copy_(h)
        y = self.norm(y, z)
        return self.out_proj(y.to(u.dtype))

    def step(self, u, cache):
        z, xBC, dt = self._split(u)                              # u: [B,1,d_model]
        z = z[:, 0]; xBC = xBC[:, 0]; dt = dt[:, 0]
        # conv step: roll cache, append, depthwise dot
        conv = cache["conv"]
        conv.copy_(torch.roll(conv, shifts=-1, dims=-1))
        conv[:, :, -1] = xBC
        w = self.conv1d.weight.squeeze(1)                        # [conv_dim, d_conv]
        xc = (conv * w).sum(-1) + self.conv1d.bias
        xc = F.silu(xc)
        x, B, C = torch.split(
            xc, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        H, P, N, G = self.nheads, self.headdim, self.d_state, self.ngroups
        A = -torch.exp(self.A_log.float())
        dtf = F.softplus(dt.float() + self.dt_bias.float())      # [B, H]
        xs = x.reshape(-1, H, P).float()
        rep = H // G
        Bt = B.reshape(-1, G, N).float().repeat_interleave(rep, dim=1)
        Ct = C.reshape(-1, G, N).float().repeat_interleave(rep, dim=1)
        dA = torch.exp(dtf * A)
        dBx = torch.einsum("bh,bhn,bhp->bhpn", dtf, Bt, xs)
        h = cache["h"] * dA[:, :, None, None] + dBx
        cache["h"].copy_(h)
        y = torch.einsum("bhpn,bhn->bhp", h, Ct) + self.D.float()[None, :, None] * xs
        y = y.reshape(u.shape[0], 1, self.d_ssm)
        y = self.norm(y, z.unsqueeze(1))
        return self.out_proj(y.to(u.dtype))


class _Mamba2NativeSSD(nn.Module):
    """Pure-PyTorch Mamba-2 (SSD) back-end with kernel-matching param tree.

    Wraps :class:`_Mamba2SSDInner` as ``self.inner`` so the parameter keys
    are ``inner.*`` — byte-identical to the mamba_ssm CUDA wrapper
    (:class:`_Mamba2CUDA`). A checkpoint trained with the kernel back-end
    therefore loads with NO remapping (only the ``_orig_mod.`` compile-prefix
    strip the loader already does).
    """

    def __init__(self, d_model, d_state=128, d_conv=4, expand_factor=2,
                 dt_min=0.001, dt_max=0.1, headdim=64):
        super().__init__()
        self.inner = _Mamba2SSDInner(d_model, d_state, d_conv, expand_factor,
                                     headdim=headdim, dt_min=dt_min, dt_max=dt_max)

    def reset_special_parameters(self) -> None:
        # Delegate so Mamba2Layer.reset_special_parameters() (called from
        # HelixModel._init_weights) reaches the inner SSM params for this backend.
        self.inner.reset_special_parameters()

    @property
    def out_proj(self):
        return self.inner.out_proj

    def forward(self, x, ssm_state=None):
        return self.inner(x, ssm_state), None

    def allocate_cache(self, batch, max_seqlen, device, dtype):
        return self.inner.allocate_cache(batch, max_seqlen, device, dtype)

    def prefill(self, x, cache):
        return self.inner.prefill(x, cache)

    def step(self, x, cache):
        return self.inner.step(x, cache)


# ---------------------------------------------------------------------------
# Public Mamba2Layer: picks back-end at construction time
# ---------------------------------------------------------------------------
class Mamba2Layer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        d_conv: int = 4,
        expand_factor: int = 2,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
    ):
        super().__init__()
        use_cuda = _cuda_kernels_enabled() and _MAMBA_SSM_AVAILABLE and torch.cuda.is_available()
        if use_cuda:
            self.backend = "mamba_ssm"
            impl = _Mamba2CUDA
        elif _native_ssd_enabled():
            # Pure-PyTorch SSD with the kernel's parameter layout — load a
            # kernel-trained checkpoint with no remap on a CPU/non-kernel box.
            self.backend = "native_ssd"
            impl = _Mamba2NativeSSD
        else:
            self.backend = "native"
            impl = _Mamba2Native
        self._impl = impl(
            d_model=d_model, d_state=d_state, d_conv=d_conv,
            expand_factor=expand_factor, dt_min=dt_min, dt_max=dt_max,
        )

    @property
    def out_proj(self) -> nn.Module:
        """Expose out_proj so the scaled-output init finds it in both back-ends."""
        inner = self._impl
        # Native has a direct out_proj. mamba_ssm Mamba2 nests it at .inner.out_proj.
        if hasattr(inner, "out_proj"):
            return inner.out_proj
        if hasattr(inner, "inner") and hasattr(inner.inner, "out_proj"):
            return inner.inner.out_proj
        raise AttributeError("no out_proj")

    def forward(self, x, ssm_state=None):
        return self._impl(x, ssm_state)

    # Incremental decoding API — dispatched to the active back-end.
    def allocate_cache(self, batch: int, max_seqlen: int, device, dtype):
        return self._impl.allocate_cache(batch, max_seqlen, device, dtype)

    def prefill(self, x, cache):
        return self._impl.prefill(x, cache)

    def step(self, x, cache):
        return self._impl.step(x, cache)

    def reset_special_parameters(self) -> None:
        reset = getattr(self._impl, "reset_special_parameters", None)
        if callable(reset):
            reset()


__all__ = ["Mamba2Layer"]
