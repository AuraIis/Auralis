"""Mamba-2 (selective state space) layer — pure-PyTorch reference.

This is the portable reference implementation used for unit tests and
CPU-side development. For GPU training swap in ``mamba_ssm.Mamba2`` via a
config flag (TODO Phase 1 GPU): the interface here matches
``forward(x) -> (out, new_ssm_state)`` which is what ``HelixBlock``
expects regardless of backend.

Reference: Dao & Gu, "Transformers are SSMs: Generalized Models and
Efficient Algorithms Through Structured State Space Duality" (2024).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand_factor = expand_factor
        self.d_inner = expand_factor * d_model

        # x + z stream projections (gated)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        # Depthwise 1-D conv over the sequence
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )

        # SSM projections: dt | B | C (C and B live in d_state; dt in d_inner)
        self.x_proj = nn.Linear(self.d_inner, self.d_inner + 2 * d_state, bias=False)

        # dt projection (input-dependent discretization step)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        # Initialize bias so softplus(bias) is distributed in [dt_min, dt_max]
        with torch.no_grad():
            dt = torch.exp(
                torch.rand(self.d_inner)
                * (float(torch.log(torch.tensor(dt_max))) - float(torch.log(torch.tensor(dt_min))))
                + float(torch.log(torch.tensor(dt_min)))
            )
            inv_softplus_dt = dt + torch.log(-torch.expm1(-dt))
            self.dt_proj.bias.copy_(inv_softplus_dt)

        # Diagonal state matrix A — parameterised as log to keep it negative
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))

        # Skip/residual scale per channel
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection back to d_model
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,                                       # [B, L, d_model]
        ssm_state: torch.Tensor | None = None,                 # [B, d_inner, d_state]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, L, _ = x.shape

        # Gated input streams
        xz = self.in_proj(x)                                   # [B, L, 2*d_inner]
        x_in, z = xz.chunk(2, dim=-1)

        # Depthwise conv along sequence
        x_in = x_in.transpose(1, 2)                            # [B, d_inner, L]
        x_in = self.conv1d(x_in)[..., :L]                      # causal trim
        x_in = x_in.transpose(1, 2)                            # [B, L, d_inner]
        x_in = F.silu(x_in)

        # Selective SSM params
        x_dbl = self.x_proj(x_in)                              # [B, L, d_inner + 2*d_state]
        dt, B_ssm, C_ssm = x_dbl.split(
            [self.d_inner, self.d_state, self.d_state], dim=-1,
        )
        dt = F.softplus(self.dt_proj(dt))                      # [B, L, d_inner] >= 0
        A = -torch.exp(self.A_log.float())                     # [d_inner, d_state], negative

        y, new_state = self._selective_scan(x_in, dt, A, B_ssm, C_ssm, self.D, ssm_state)

        # Gate output with z stream
        y = y * F.silu(z)
        return self.out_proj(y), new_state

    # ------------------------------------------------------------------
    # Selective scan — pure Python sequential reference.
    # Production: replace with mamba_ssm's CUDA kernel for O(L) parallel time.
    # ------------------------------------------------------------------
    def _selective_scan(
        self,
        x: torch.Tensor,                                       # [B, L, d_inner]
        dt: torch.Tensor,                                      # [B, L, d_inner]
        A: torch.Tensor,                                       # [d_inner, d_state]
        Bp: torch.Tensor,                                      # [B, L, d_state]
        Cp: torch.Tensor,                                      # [B, L, d_state]
        D: torch.Tensor,                                       # [d_inner]
        state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, L, d_inner = x.shape
        d_state = A.shape[1]
        dtype = x.dtype

        # Discretize: dA = exp(dt * A), dB = dt * B
        dA = torch.exp(dt.unsqueeze(-1) * A)                   # [B, L, d_inner, d_state]
        dB = dt.unsqueeze(-1) * Bp.unsqueeze(-2)               # [B, L, d_inner, d_state]

        if state is None:
            h = torch.zeros(B, d_inner, d_state, device=x.device, dtype=dtype)
        else:
            h = state.to(dtype)

        outputs = []
        for t in range(L):
            h = dA[:, t] * h + dB[:, t] * x[:, t].unsqueeze(-1)  # [B, d_inner, d_state]
            y_t = (h * Cp[:, t].unsqueeze(-2)).sum(dim=-1)       # [B, d_inner]
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)                        # [B, L, d_inner]
        y = y + x * D                                          # residual skip
        return y, h


__all__ = ["Mamba2Layer"]
