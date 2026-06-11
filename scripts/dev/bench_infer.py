"""Autoregressive-generation benchmark + correctness gate for Helix v2.

Measures, on real checkpoint weights with the kernel back-end:

  baseline : full re-forward over the growing prefix every token  (O(L^2))
  cached   : incremental decode — Mamba/GLA recurrent state + windowed
             KV cache carried across steps                         (O(1)/token)

Correctness gate: cached greedy output must be token-IDENTICAL to the
baseline on every prompt before any speed number is trusted.

Every result is appended to a JSON file so nothing is lost mid-session.

Example (in container):
  python3 scripts/dev/bench_infer.py \
      --model-config configs/model/helix_v2_1b_flash.yaml \
      --checkpoint /workspace/v2data/checkpoints/.../step_50000.pt \
      --out /workspace/v2data/perf_lab/infer_bench/results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Kernel back-ends must be on BEFORE model import (parameter layout).
os.environ.setdefault("AURALIS_USE_CUDA_KERNELS", "1")

import sentencepiece as spm
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from auralis.model import build_model  # noqa: E402

PROMPTS = [
    # German prose
    "Die Industrialisierung veränderte das Leben in Deutschland grundlegend. "
    "Städte wuchsen rasant, Eisenbahnen verbanden entfernte Regionen, und die "
    "Fabrikarbeit ersetzte das traditionelle Handwerk. Doch der Fortschritt hatte "
    "auch Schattenseiten: Kinderarbeit, lange Arbeitszeiten und beengte Wohnungen.",
    # German Q&A
    "Frage: Erkläre den Unterschied zwischen Wetter und Klima. Antwort: Wetter "
    "beschreibt den kurzfristigen Zustand der Atmosphäre an einem Ort, etwa "
    "Temperatur, Niederschlag und Wind. Klima dagegen ist der langjährige",
    # Code
    "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n\n"
    "def main():\n    for i in range(10):\n        print(fibonacci(i))\n",
]


def encode_prompt(sp, text, n_tokens):
    ids = sp.EncodeAsIds(text)
    while len(ids) < n_tokens:
        ids = ids + ids
    return ids[:n_tokens]


@torch.no_grad()
def gen_baseline(model, ids, max_new):
    """Full re-forward over the growing prefix every token (current path)."""
    x = torch.tensor([ids], device="cuda", dtype=torch.long)
    new = []
    torch.cuda.synchronize(); t0 = time.perf_counter()
    out = model(input_ids=x)
    nid = out["logits"][0, -1].argmax().item()
    new.append(nid)
    torch.cuda.synchronize(); ttft = time.perf_counter() - t0
    t1 = time.perf_counter()
    for _ in range(max_new - 1):
        x = torch.cat([x, torch.tensor([[nid]], device="cuda")], dim=1)
        out = model(input_ids=x)
        nid = out["logits"][0, -1].argmax().item()
        new.append(nid)
    torch.cuda.synchronize()
    decode_s = time.perf_counter() - t1
    return new, ttft, (max_new - 1) / decode_s


@torch.no_grad()
def gen_cached(model, ids, max_new, cuda_graph=False):
    x = torch.tensor([ids], device="cuda", dtype=torch.long)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    # generate() emits the first token at the end of prefill
    toks = model.generate(x, max_new_tokens=1)
    torch.cuda.synchronize(); ttft = time.perf_counter() - t0
    # full run for decode throughput
    torch.cuda.synchronize(); t1 = time.perf_counter()
    toks = model.generate(x, max_new_tokens=max_new, cuda_graph=cuda_graph)
    torch.cuda.synchronize()
    total = time.perf_counter() - t1
    decode_s = total - ttft
    return toks[0].tolist(), ttft, (max_new - 1) / max(decode_s, 1e-9)


def gen_graph(model, ids, max_new):
    return gen_cached(model, ids, max_new, cuda_graph=True)


@torch.no_grad()
def gen_graph_batched(model, prompt_ids_list, max_new, batch):
    """Aggregate decode throughput: `batch` prompts decoded simultaneously."""
    base = prompt_ids_list * (batch // len(prompt_ids_list) + 1)
    x = torch.tensor(base[:batch], device="cuda", dtype=torch.long)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    toks = model.generate(x, max_new_tokens=max_new, cuda_graph=True)
    torch.cuda.synchronize()
    total = time.perf_counter() - t0
    return toks, batch * max_new / total


def vram_peak():
    return torch.cuda.max_memory_allocated() / 2**30


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, default=REPO / "tokenizer" / "helix_v2_tokenizer.model")
    p.add_argument("--prompt-tokens", type=int, default=64)
    p.add_argument("--new-tokens", type=int, default=256)
    p.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    p.add_argument("--modes", default="baseline,cached")
    p.add_argument("--batch", type=int, default=0, help="also bench graph-mode batched decode at this batch size")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    torch.manual_seed(0)
    model = build_model(args.model_config).cuda().eval()
    payload = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
    state = {k.replace("_orig_mod.", ""): v for k, v in payload["model"].items()}
    miss, extra = model.load_state_dict(state, strict=False)
    assert not miss and not extra, (miss[:3], extra[:3])
    if args.dtype == "bf16":
        model = model.to(torch.bfloat16)
    print(f"loaded step={payload.get('state', {}).get('step')} dtype={args.dtype}", flush=True)

    sp = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    prompts = [encode_prompt(sp, t, args.prompt_tokens) for t in PROMPTS]
    modes = args.modes.split(",")
    results = {"dtype": args.dtype, "prompt_tokens": args.prompt_tokens,
               "new_tokens": args.new_tokens, "runs": [], "match": None}

    tokens = {m: [] for m in modes}
    for m in modes:
        fn = {"baseline": gen_baseline, "cached": gen_cached, "graph": gen_graph}[m]
        # warmup on prompt 0 (triton autotune etc.)
        fn(model, prompts[0], 16)
        for i, ids in enumerate(prompts):
            torch.cuda.reset_peak_memory_stats()
            toks, ttft, dec = fn(model, ids, args.new_tokens)
            tokens[m].append(toks)
            r = {"mode": m, "prompt": i, "ttft_s": round(ttft, 4),
                 "decode_tok_s": round(dec, 2), "peak_vram_gib": round(vram_peak(), 2)}
            results["runs"].append(r)
            print(r, flush=True)

    ref = modes[0]
    for m in modes[1:]:
        match = all(a == b for a, b in zip(tokens[ref], tokens[m]))
        results[f"match_{m}_vs_{ref}"] = match
        print(f"TOKEN-IDENTICAL {m} vs {ref}: {match}", flush=True)
        if not match:
            for i, (a, b) in enumerate(zip(tokens[ref], tokens[m])):
                d = next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), None)
                print(f"  prompt {i}: first divergence at {d}", flush=True)

    if args.batch > 0:
        toks_b, agg = gen_graph_batched(model, prompts, args.new_tokens, args.batch)   # warmup+capture
        torch.cuda.reset_peak_memory_stats()
        toks_b, agg = gen_graph_batched(model, prompts, args.new_tokens, args.batch)
        # correctness: rows of the batch must match the single-prompt graph runs
        bmatch = None
        if "graph" in tokens and tokens["graph"]:
            bmatch = all(toks_b[i].tolist() == tokens["graph"][i % len(prompts)]
                         for i in range(min(args.batch, len(prompts))))
        r = {"mode": f"graph_b{args.batch}", "agg_tok_s": round(agg, 2),
             "peak_vram_gib": round(vram_peak(), 2), "rows_match_single": bmatch}
        results["runs"].append(r)
        print(r, flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    hist = json.loads(args.out.read_text()) if args.out.exists() else []
    hist.append(results)
    args.out.write_text(json.dumps(hist, indent=1))
    print(f"saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
