"""Memory-lean fused (linear → cross-entropy) over a large vocabulary.

The stock training loss materialises the full ``[N, V]`` logits tensor (and,
during backward, its fp32 softmax gradient) just to reduce it to a scalar. At
V=200k that tensor — computed for the main head *and* every MTP head — is the
single largest activation in the step and the wall we hit when trying to raise
the batch size.

``fused_linear_cross_entropy`` computes exactly the same
``cross_entropy(hidden @ weight.T, labels, ignore_index=…)`` but forms the
logits one row-chunk at a time in both forward and backward, so the full
``[N, V]`` tensor never exists. Only a running loss scalar and the two gradient
accumulators persist.

Numerics match ``F.cross_entropy`` up to (a) chunk reduction order and (b) an
fp32 upcast of the per-chunk softmax — the latter only *improves* accuracy. It
is therefore not bit-identical; parity is asserted with a tolerance
(tests/model/test_fused_ce_parity.py: ~1e-5 in fp32, bf16-loose in bf16).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class _FusedLinearCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, labels, ignore_index, chunk_size):
        # hidden: [N, D]  weight: [V, D]  labels: [N]
        n = hidden.shape[0]
        n_valid = (labels != ignore_index).sum().clamp_(min=1)
        loss = hidden.new_zeros((), dtype=torch.float32)
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            logits_c = F.linear(hidden[start:end], weight).float()  # [c, V]
            loss = loss + F.cross_entropy(
                logits_c,
                labels[start:end],
                ignore_index=ignore_index,
                reduction="sum",
            )
        loss = loss / n_valid.to(loss.dtype)
        ctx.save_for_backward(hidden, weight, labels)
        ctx.ignore_index = ignore_index
        ctx.chunk_size = chunk_size
        ctx.n_valid = n_valid
        return loss.to(hidden.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        hidden, weight, labels = ctx.saved_tensors
        ignore_index = ctx.ignore_index
        chunk_size = ctx.chunk_size
        n_valid = ctx.n_valid
        need_h, need_w = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        n = hidden.shape[0]
        # d(mean-loss)/d(logit) = (softmax - onehot) / n_valid, times upstream grad.
        scale = grad_output.float() / n_valid.to(torch.float32)
        grad_hidden = torch.zeros_like(hidden) if need_h else None
        grad_weight = torch.zeros_like(weight, dtype=torch.float32) if need_w else None
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            h_c = hidden[start:end]
            lbl_c = labels[start:end]
            probs = torch.softmax(F.linear(h_c, weight).float(), dim=-1)  # [c, V]
            valid = lbl_c != ignore_index
            safe = torch.where(valid, lbl_c, torch.zeros_like(lbl_c))
            probs.scatter_add_(
                1,
                safe.unsqueeze(1),
                probs.new_full((probs.shape[0], 1), -1.0),
            )  # probs -> (softmax - onehot)
            probs *= valid.unsqueeze(1).to(probs.dtype)  # zero out ignore rows
            d_logits = probs * scale  # [c, V] fp32
            if need_h:
                grad_hidden[start:end] = (d_logits.to(weight.dtype) @ weight).to(hidden.dtype)
            if need_w:
                grad_weight += d_logits.t() @ h_c.float()
        return (
            grad_hidden,
            grad_weight.to(weight.dtype) if need_w else None,
            None,
            None,
            None,
        )


def fused_linear_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """Mean cross-entropy of ``F.linear(hidden, weight)`` vs ``labels``.

    ``hidden`` is ``[..., D]`` and ``labels`` ``[...]`` (flattened internally);
    ``weight`` is ``[V, D]`` (a tied embedding matrix or an ``nn.Linear`` weight,
    bias-free). Never materialises the full ``[N, V]`` logits.
    """
    h2 = hidden.reshape(-1, hidden.shape[-1])
    l1 = labels.reshape(-1)
    return _FusedLinearCrossEntropy.apply(h2, weight, l1, ignore_index, int(chunk_size))
