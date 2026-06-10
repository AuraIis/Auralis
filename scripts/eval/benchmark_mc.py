#!/usr/bin/env python3
"""Lightweight multiple-choice benchmark runner (lm-eval-style loglikelihood).

Scores Helix (custom) AND HF baselines on the SAME methodology for a fair compare.
Tasks: mmlu (letter-continuation, acc), arc_challenge + hellaswag (choice-text, acc_norm).

  --model helix:checkpoints/sft_v1/sft_smoke_step_2000.pt
  --model hf:Qwen/Qwen2.5-0.5B
  --tasks mmlu,arc_challenge,hellaswag  --limit 300
"""
import os, sys, argparse, math, random, pathlib
import torch

os.environ.setdefault("AURALIS_USE_MAMBA_KERNEL", "1")
os.environ.setdefault("AURALIS_USE_GLA_KERNEL", "1")   # fla chunk_gla: 20-30x faster than native scan
REPO = pathlib.Path("/workspace/v2data"); sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "src"))

# ---------------- model backends ----------------
class HelixLM:
    def __init__(self, ckpt, cfg=str(REPO/"configs/model/helix_v2_1b.yaml"), tok=str(REPO/"tokenizer/helix_v2_tokenizer.model")):
        import sentencepiece as spm
        from auralis.model import build_model
        self.dev = torch.device("cuda")
        self.sp = spm.SentencePieceProcessor(model_file=tok)
        self.model = build_model(cfg).to(self.dev).eval()
        payload = torch.load(ckpt, map_location=self.dev, weights_only=False)
        state = payload.get("model", payload.get("state_dict", payload))
        self.model.load_state_dict(state, strict=False)
    def encode(self, s): return self.sp.EncodeAsIds(s)
    @torch.no_grad()
    def loglik(self, ctx, cont):
        ci = self.encode(ctx); oi = self.encode(cont)
        if not oi: return -1e9, 1
        ids = torch.tensor([ci + oi], device=self.dev)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = self.model(input_ids=ids)["logits"][0].float()
        lp = torch.log_softmax(logits, -1)
        tot = 0.0
        for i, t in enumerate(oi):
            pos = len(ci) + i - 1
            if pos < 0: continue
            tot += lp[pos, t].item()
        return tot, len(oi)

class HFLM:
    def __init__(self, repo):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.dev = torch.device("cuda")
        self.tok = AutoTokenizer.from_pretrained(repo)
        self.model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.bfloat16).to(self.dev).eval()
    def encode(self, s): return self.tok.encode(s, add_special_tokens=False)
    @torch.no_grad()
    def loglik(self, ctx, cont):
        ci = self.encode(ctx); oi = self.encode(cont)
        if not oi: return -1e9, 1
        ids = torch.tensor([ci + oi], device=self.dev)
        logits = self.model(ids).logits[0].float()
        lp = torch.log_softmax(logits, -1)
        tot = 0.0
        for i, t in enumerate(oi):
            pos = len(ci) + i - 1
            if pos < 0: continue
            tot += lp[pos, t].item()
        return tot, len(oi)

# ---------------- tasks ----------------
def load_task(name, limit):
    from datasets import load_dataset
    rng = random.Random(0)
    if name == "mmlu":
        ds = load_dataset("cais/mmlu", "all", split="test")
        items = []
        for r in ds:
            q = r["question"]; ch = r["choices"]; ans = r["answer"]  # int 0-3
            ctx = f"{q.strip()}\n" + "".join(f"{l}. {c}\n" for l, c in zip("ABCD", ch)) + "Answer:"
            items.append({"ctx": ctx, "conts": [f" {l}" for l in "ABCD"], "gold": ans, "norm": False})
    elif name == "arc_challenge":
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
        items = []
        for r in ds:
            labels = r["choices"]["label"]; texts = r["choices"]["text"]
            if r["answerKey"] not in labels: continue
            gold = labels.index(r["answerKey"])
            ctx = f"Question: {r['question'].strip()}\nAnswer:"
            items.append({"ctx": ctx, "conts": [f" {t}" for t in texts], "gold": gold, "norm": True})
    elif name == "hellaswag":
        ds = load_dataset("Rowan/hellaswag", split="validation")
        items = []
        for r in ds:
            ctx = (r["activity_label"] + ": " + r["ctx"]).strip()
            items.append({"ctx": ctx, "conts": [" " + e.strip() for e in r["endings"]], "gold": int(r["label"]), "norm": True})
    elif name == "mmlu_de":
        ds = load_dataset("alexandrainst/m_mmlu", "de", split="test")
        items = []
        for r in ds:
            if r["answer"] not in "ABCD": continue
            opts = [r["option_a"], r["option_b"], r["option_c"], r["option_d"]]
            gold = "ABCD".index(r["answer"])
            ctx = f"{r['instruction'].strip()}\n" + "".join(f"{l}. {c}\n" for l, c in zip("ABCD", opts)) + "Antwort:"
            items.append({"ctx": ctx, "conts": [f" {l}" for l in "ABCD"], "gold": gold, "norm": False})
    elif name == "arc_de":
        ds = load_dataset("alexandrainst/m_arc", "de", split="test")
        items = []
        letters = "ABCDE"
        for r in ds:
            opts = [r.get(f"option_{l.lower()}") for l in letters]
            opts = [o for o in opts if o and str(o) != "None"]
            if r["answer"] not in letters: continue
            gold = letters.index(r["answer"])
            if gold >= len(opts): continue
            ctx = f"Frage: {r['instruction'].strip()}\nAntwort:"
            items.append({"ctx": ctx, "conts": [f" {o}" for o in opts], "gold": gold, "norm": True})
    elif name == "hellaswag_de":
        ds = load_dataset("alexandrainst/m_hellaswag", "de", split="val")
        items = []
        for r in ds:
            ctx = (r.get("activity_label", "") + ": " + r["ctx"]).strip()
            items.append({"ctx": ctx, "conts": [" " + e.strip() for e in r["endings"]], "gold": int(r["label"]), "norm": True})
    else:
        raise SystemExit("unknown task " + name)
    rng.shuffle(items)
    return items[:limit] if limit else items

def run(lm, items):
    correct = 0
    for it in items:
        scores = []
        for c in it["conts"]:
            ll, n = lm.loglik(it["ctx"], c)
            scores.append(ll / n if it["norm"] else ll)
        if max(range(len(scores)), key=lambda i: scores[i]) == it["gold"]:
            correct += 1
    return correct / max(1, len(items))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tasks", default="mmlu,arc_challenge,hellaswag")
    ap.add_argument("--limit", type=int, default=300)
    a = ap.parse_args()
    kind, ref = a.model.split(":", 1)
    print(f"== loading {a.model} ==", flush=True)
    lm = HelixLM(ref) if kind == "helix" else HFLM(ref)
    print("== model ready ==", flush=True)
    for t in a.tasks.split(","):
        items = load_task(t, a.limit)
        acc = run(lm, items)
        metric = "acc_norm" if items and items[0]["norm"] else "acc"
        print(f"RESULT {a.model} | {t} | {metric}={acc*100:.1f}% (n={len(items)})", flush=True)

if __name__ == "__main__":
    main()
