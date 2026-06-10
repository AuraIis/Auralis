#!/usr/bin/env python3
"""Perplexity on held-out sets (code + DE/EN retention) for a checkpoint.
Measure BEFORE annealing (baseline) and after (code ppl should drop, retention hold)."""
import os, sys, json, argparse, math, pathlib
REPO = pathlib.Path("/workspace/v2data"); sys.path.insert(0, str(REPO/"src")); sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO/"scripts/data"))
os.environ.setdefault("AURALIS_USE_MAMBA_KERNEL", "1")
os.environ.setdefault("AURALIS_USE_GLA_KERNEL", "1")   # fla chunk_gla: 20-30x faster than native scan
import torch, torch.nn.functional as F
import sentencepiece as spm
from auralis.model import build_model
from code_format import tab_indent  # pretrain code is tab-indented -> eval must match


def ppl_on(model, sp, path, dev, max_len=1024, max_docs=200, tab=False):
    tot_loss = 0.0; tot_tok = 0; nd = 0
    for line in open(path, encoding="utf-8"):
        if not line.strip() or nd >= max_docs:
            if nd >= max_docs: break
            continue
        try: text = json.loads(line)["text"]
        except Exception: continue
        if tab: text = tab_indent(text)
        ids = sp.EncodeAsIds(text)[:max_len]
        if len(ids) < 8: continue
        inp = torch.tensor([ids], device=dev)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(input_ids=inp)["logits"][0].float()
        loss = F.cross_entropy(logits[:-1], torch.tensor(ids[1:], device=dev), reduction="sum")
        tot_loss += loss.item(); tot_tok += len(ids) - 1; nd += 1
    avg = tot_loss / max(1, tot_tok)
    return avg, math.exp(avg), nd, tot_tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--sets", required=True, help="name=path,name=path,...")
    ap.add_argument("--model-config", default=str(REPO/"configs/model/helix_v2_1b.yaml"))
    ap.add_argument("--tokenizer", default=str(REPO/"tokenizer/helix_v2_tokenizer.model"))
    ap.add_argument("--max-docs", type=int, default=200)
    a = ap.parse_args()
    dev = torch.device("cuda")
    sp = spm.SentencePieceProcessor(model_file=a.tokenizer)
    model = build_model(a.model_config).to(dev).eval()
    p = torch.load(a.checkpoint, map_location=dev, weights_only=False)
    model.load_state_dict(p.get("model", p.get("state_dict", p)), strict=False)
    print(f"=== PPL | {pathlib.Path(a.checkpoint).name} ===", flush=True)
    for pair in a.sets.split(","):
        name, path = pair.split("=", 1)
        tab = name.startswith("code")  # CODE sets only — never prose retention sets
        avg, ppl, nd, tk = ppl_on(model, sp, path, dev, max_docs=a.max_docs, tab=tab)
        print(f"  {name:8} loss={avg:.3f} ppl={ppl:7.1f}  (docs {nd}, tokens {tk}, tab={tab})", flush=True)


if __name__ == "__main__":
    main()
