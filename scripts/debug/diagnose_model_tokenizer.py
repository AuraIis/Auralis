"""Diagnose whether the Helix tokenizer/model stack is structurally sane.

This is intentionally not a benchmark. It checks invariants that should hold
before we blame data quality or training length:

- tokenizer vocab size and special-token atomics
- Unicode/chat-template round-trips
- model/tokenizer vocab match
- checkpoint state_dict compatibility
- finite forward loss/logits
- no future-token leakage on prefix logits
- finite/non-zero gradients
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import sentencepiece as spm
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))


def _maybe_enable_mamba_kernel() -> bool:
    if os.environ.get("AURALIS_USE_MAMBA_KERNEL") == "1":
        return True
    if not torch.cuda.is_available():
        return False
    try:
        import mamba_ssm  # noqa: F401
    except ImportError:
        return False
    os.environ["AURALIS_USE_MAMBA_KERNEL"] = "1"
    return True


_KERNEL_ACTIVE = _maybe_enable_mamba_kernel()

from auralis.model import build_model  # noqa: E402
from auralis.tokenizer.chat_template import (  # noqa: E402
    build_inference_prompt,
    build_training_prompt,
)


def _print_check(name: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    suffix = f" | {detail}" if detail else ""
    print(f"{status:4} {name}{suffix}", flush=True)


def check_tokenizer(sp: spm.SentencePieceProcessor) -> bool:
    print("\n=== TOKENIZER ===", flush=True)
    ok_all = True
    print(f"vocab_size={sp.get_piece_size():,}", flush=True)

    reserved = [
        ("<pad>", 0),
        ("<unk>", 1),
        ("<s>", 2),
        ("</s>", 3),
    ]
    for token, expected_id in reserved:
        piece_id = sp.piece_to_id(token)
        ok = piece_id == expected_id
        ok_all = ok_all and ok
        _print_check(
            f"reserved token {token}",
            ok,
            f"piece_id={piece_id} expected={expected_id}",
        )

    specials = [
        "<|system|>",
        "<|user|>",
        "<|assistant|>",
        "<|end|>",
        "<think>",
        "</think>",
        "<|python|>",
        "<|endcode|>",
    ]
    for token in specials:
        piece_id = sp.piece_to_id(token)
        ids = sp.encode(token, out_type=int)
        # SentencePiece adds a dummy prefix marker at the beginning of a string
        # unless the model was trained with add_dummy_prefix=false. That makes
        # a raw encode("<|system|>") look like [dummy_prefix, special_id].
        # The important invariant is that the special marker itself is one
        # registered piece, not split into "<", "|", "system", ... fragments.
        ok = (
            piece_id >= 0
            and ids.count(piece_id) == 1
            and sp.decode(ids) == token
        )
        mode = "single-id" if ids == [piece_id] else "with-dummy-prefix"
        ok_all = ok_all and ok
        _print_check(
            f"special registered {token}",
            ok,
            f"piece_id={piece_id} encode={ids} mode={mode}",
        )

    examples = [
        "Äpfel, Öl, Grüße, Straße, München, größer, Fußball.",
        "„Die Photosynthese ist wichtig.“ — CO2 + H2O -> C6H12O6 + O2.",
        'def add(a, b):\n    return a + b\nprint("Hallo")',
        build_inference_prompt(
            [{"role": "user", "content": "Was ist Wasser?"}],
            default_system="Du bist Auralis.",
        ),
        build_training_prompt(
            [
                {"role": "system", "content": "Du bist Auralis."},
                {"role": "user", "content": "Sag Hallo."},
                {"role": "assistant", "content": "Hallo!"},
            ],
        ),
    ]
    for idx, text in enumerate(examples, start=1):
        ids = sp.encode(text, out_type=int)
        decoded = sp.decode(ids)
        unk = sum(1 for token_id in ids if token_id == sp.unk_id())
        ok = decoded == text and unk == 0
        ok_all = ok_all and ok
        _print_check(
            f"roundtrip sample {idx}",
            ok,
            f"chars={len(text)} tokens={len(ids)} unk={unk}",
        )
        if not ok:
            print(f"  original={text!r}", flush=True)
            print(f"  decoded ={decoded!r}", flush=True)

    return ok_all


def load_checkpoint(model: torch.nn.Module, checkpoint: Path, device: torch.device) -> bool:
    if not checkpoint:
        print("checkpoint=none", flush=True)
        return True
    if not checkpoint.exists():
        _print_check("checkpoint exists", False, str(checkpoint))
        return False

    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    state = {k.replace("_orig_mod.", ""): v for k, v in payload["model"].items()}
    missing, extra = model.load_state_dict(state, strict=False)
    ok = not missing and not extra
    step = payload.get("state", {}).get("step", payload.get("step", "?"))
    _print_check(
        "checkpoint state_dict load",
        ok,
        f"missing={len(missing)} extra={len(extra)} step={step}",
    )
    if missing:
        print(f"  first missing={missing[:5]}", flush=True)
    if extra:
        print(f"  first extra={extra[:5]}", flush=True)
    return ok


def check_model(args: argparse.Namespace, sp: spm.SentencePieceProcessor) -> bool:
    print("\n=== MODEL ===", flush=True)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} mamba_kernel={_KERNEL_ACTIVE}", flush=True)

    model = build_model(args.model_config).to(device)
    print(
        f"params={model.count_parameters()/1e6:.2f}M cfg_vocab={model.config.vocab_size:,} "
        f"tok_vocab={sp.get_piece_size():,} tied={model.config.advanced.tie_embeddings}",
        flush=True,
    )
    ok_all = model.config.vocab_size == sp.get_piece_size()
    _print_check("model/tokenizer vocab match", ok_all)

    if args.checkpoint:
        ok_all = load_checkpoint(model, args.checkpoint, device) and ok_all

    model.eval()
    with torch.no_grad():
        x = torch.randint(0, model.config.vocab_size, (1, args.seq_len), device=device)
        out = model(input_ids=x, labels=x)
        logits = out["logits"]
        loss = out["loss"]
    ok_shape = tuple(logits.shape) == (1, args.seq_len, model.config.vocab_size)
    ok_finite = bool(torch.isfinite(logits).all()) and bool(torch.isfinite(loss))
    _print_check("forward logits shape", ok_shape, str(tuple(logits.shape)))
    _print_check("forward finite logits/loss", ok_finite, f"loss={float(loss):.4f}")
    ok_all = ok_all and ok_shape and ok_finite

    prefix = args.seq_len // 2
    base = torch.randint(0, model.config.vocab_size, (1, args.seq_len), device=device)
    alt = base.clone()
    alt[:, prefix:] = torch.randint(
        0,
        model.config.vocab_size,
        (1, args.seq_len - prefix),
        device=device,
    )
    model.eval()
    with torch.no_grad():
        a = model(input_ids=base)["logits"][:, :prefix, :]
        b = model(input_ids=alt)["logits"][:, :prefix, :]
    max_diff = float((a - b).abs().max())
    mean_diff = float((a - b).abs().mean())
    ok_causal = max_diff <= args.causal_tol
    _print_check(
        "causal prefix unaffected by future suffix",
        ok_causal,
        f"max_abs_diff={max_diff:.6g} mean_abs_diff={mean_diff:.6g}",
    )
    ok_all = ok_all and ok_causal

    model.train()
    model.gradient_checkpointing_disable()
    x = torch.randint(0, model.config.vocab_size, (1, args.seq_len), device=device)
    y = x.clone()
    out = model(input_ids=x, labels=y)
    loss = out["loss"]
    loss.backward()
    checked = 0
    finite_grad = True
    nonzero_grad = False
    for _, param in model.named_parameters():
        if param.grad is None:
            continue
        checked += 1
        finite_grad = finite_grad and bool(torch.isfinite(param.grad).all())
        nonzero_grad = nonzero_grad or bool(param.grad.abs().sum() > 0)
        if checked >= args.grad_param_limit:
            break
    _print_check(
        "gradient finite/nonzero sample",
        finite_grad and nonzero_grad and checked > 0,
        f"checked={checked} loss={float(loss.detach()):.4f}",
    )
    ok_all = ok_all and finite_grad and nonzero_grad and checked > 0
    return ok_all


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=REPO / "tokenizer" / "helix_v2_tokenizer.model")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--causal-tol", type=float, default=1e-4)
    parser.add_argument("--grad-param-limit", type=int, default=20)
    args = parser.parse_args()

    sp = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    tok_ok = check_tokenizer(sp)
    model_ok = check_model(args, sp)
    print("\n=== SUMMARY ===", flush=True)
    _print_check("tokenizer structurally sane", tok_ok)
    _print_check("model/checkpoint structurally sane", model_ok)
    raise SystemExit(0 if tok_ok and model_ok else 1)


if __name__ == "__main__":
    main()
