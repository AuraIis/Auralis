#!/usr/bin/env python3
"""Scored serving sweep (OFAT). Loads model once, varies ONE knob set per CONFIGS row,
scores a fixed prompt set on 8 metrics. Edit CONFIGS per stage (alpha first)."""

import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ["AURALIS_USE_CUDA_KERNELS"] = "1"
REPO = "/workspace/v2data"
sys.path.insert(0, REPO)
sys.path.insert(0, REPO + "/src")
import sentencepiece as spm
import torch

from auralis.adapters.lora import (
    freeze_base,
    inject_adapters,
    load_adapter_state_dict,
    set_adapter_scale,
)
from auralis.model import build_model

CFG = REPO + "/configs/model/helix_v2_1b_flash.yaml"
CKPT = REPO + "/checkpoints/corpus20b_codeheavy/step_60000.pt"
TOK = REPO + "/tokenizer/helix_v2_tokenizer.model"
SYSP = "Du bist Auralis, ein hilfreicher, ehrlicher KI-Assistent."
END = "<|end|>"
ASST = "<|assistant|>\n"
dev = torch.device("cuda")
sp = spm.SentencePieceProcessor(model_file=TOK)
END_ID = sp.EncodeAsIds(END)[-1]
model = build_model(CFG)
pl = torch.load(CKPT, map_location="cpu", weights_only=False)
model.load_state_dict(
    {k.replace("_orig_mod.", ""): v for k, v in pl["model"].items()}, strict=False
)
model = model.to(dev)
inject_adapters(model, r=64, alpha=128, kind="lora")
freeze_base(model)
emb = getattr(model, "embedding", None) or getattr(model, "embed_tokens", None)
model.eval()
ADP = {
    n: torch.load(REPO + f"/checkpoints/{d}/adapter_best.pt", map_location="cpu")
    for n, d in {"code": "sft_code_v3", "corrective": "sft_corrective_v3"}.items()
}
_cur = {"n": None}


def use(name):
    ck = ADP[name]
    load_adapter_state_dict(model, ck["adapter"])
    model.to(dev)
    for i, tid in enumerate(ck["emb_ids"]):
        emb.weight.data[tid] = ck["emb_rows"][i].to(emb.weight.device, emb.weight.dtype)
    _cur["n"] = name


# serving rewriter (same as shim)
ACR = {"gpu", "cpu", "ram", "vram", "ai", "ki", "llm", "api"}
QW = (
    "was",
    "wer",
    "wie",
    "wo",
    "wann",
    "warum",
    "welche",
    "nenne",
    "schreibe",
    "erklaere",
    "gib",
    "hallo",
    "hi",
    "hey",
    "danke",
    "ja",
    "nein",
    "ok",
    "bitte",
)


def bare(t):
    s = t.strip()
    if not s or any(c in s for c in "?.,\n") or any(c.isdigit() for c in s):
        return False
    w = s.split()
    return 0 < len(w) <= 3 and w[0].lower().rstrip(":") not in QW


def rw(t):
    if not bare(t):
        return t
    ws = t.strip().split()
    return (
        "Was ist "
        + " ".join(
            (w.upper() if w.lower() in ACR else (w[0].upper() + w[1:] if i == 0 else w))
            for i, w in enumerate(ws)
        )
        + "?"
    )


def gen(p, temp, rep, mn):
    ids = sp.EncodeAsIds(f"<|system|>\n{SYSP}\n{END}\n<|user|>\n{rw(p)}\n{END}\n{ASST}")
    x = torch.tensor([ids], device=dev)
    out = []
    stp = False
    torch.manual_seed(0)
    with torch.no_grad():
        for _ in range(mn):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                lg = model(input_ids=x)["logits"][0, -1].float()
            for t in set(out):
                lg[t] = lg[t] / rep if lg[t] > 0 else lg[t] * rep
            if temp > 0:
                lg = lg / temp
                v, ix = torch.topk(lg, 40)
                pr = torch.softmax(v, -1)
                nid = int(ix[torch.multinomial(pr, 1)])
            else:
                nid = int(torch.argmax(lg))
            if nid == END_ID:
                stp = True
                break
            out.append(nid)
            x = torch.cat([x, torch.tensor([[nid]], device=dev)], 1)
            if END in sp.DecodeIds(out[-4:]):
                stp = True
                break
    return sp.DecodeIds(out).split(END)[0].strip(), stp


ABST = (
    "weiß nicht",
    "weiss nicht",
    "nicht vorzukommen",
    "existiert vermutlich nicht",
    "kein eigenständiger",
    "frei erfunden",
    "nicht als eigenständig",
    "nicht bekannt",
)


def ab(t):
    tl = t.lower()
    return any(p in tl for p in ABST)


def degen(t):
    w = t.split()
    if len(w) >= 8:
        for i in range(len(w) - 7):
            if w[i : i + 4] == w[i + 4 : i + 8]:
                return True
    return len(w) > 20 and len(set(w)) / len(w) < 0.45


PROMPTS = [
    ("katze", "chat"),
    ("gpu", "chat"),
    ("hardware", "chat"),
    ("Was ist eine Katze?", "chat"),
    ("Was ist ein Hund?", "chat"),
    ("Erklaere kurz Photosynthese.", "chat"),
    ("Was ist ein Auto?", "chat"),
    ("Was ist Wasser?", "chat"),
    ("hallo", "greet"),
    ("danke", "greet"),
    ("Nenne drei Farben.", "list"),
    ("Nenne drei Tiere.", "list"),
    ("Was ist 47 mal 83?", "math"),
    ("Was sind 5 x 5?", "math"),
    ("Was sind 245 x 4?", "math"),
    ("Was ist die Hauptstadt Frankreichs?", "fact"),
    ("Was ist die Hauptstadt von Deutschland?", "fact"),
    ("Wer schrieb Faust?", "fact"),
    ("Was ist die Hauptstadt von Italien?", "fact"),
    ("Was ist ein Goblin?", "unknown"),
    ("Was ist Moxthal?", "unknown"),
    ("Wer schrieb den Roman Pombyon?", "unknown"),
    ("Was ist ein Glaztronk?", "unknown"),
]
FACT = {
    "Was ist die Hauptstadt Frankreichs?": "paris",
    "Was ist die Hauptstadt von Deutschland?": "berlin",
    "Wer schrieb Faust?": "goethe",
    "Was ist die Hauptstadt von Italien?": "rom",
}
MATH = {"Was ist 47 mal 83?": "3901", "Was sind 5 x 5?": "25", "Was sind 245 x 4?": "980"}


def m(xs):
    return round(sum(xs) / max(1, len(xs)), 2)


CONFIGS = [
    ("base t0/r1.15/120", "code", 0.6, 0.0, 1.15, 120),
    ("rep 1.05", "code", 0.6, 0.0, 1.05, 120),
    ("rep 1.10", "code", 0.6, 0.0, 1.10, 120),
    ("rep 1.20", "code", 0.6, 0.0, 1.20, 120),
    ("temp 0.2", "code", 0.6, 0.2, 1.15, 120),
    ("temp 0.4", "code", 0.6, 0.4, 1.15, 120),
    ("maxnew 80", "code", 0.6, 0.0, 1.15, 80),
    ("maxnew 200", "code", 0.6, 0.0, 1.15, 200),
]
print(
    f"{'config':12s} {'stop':5s} {'len':5s} {'degen':5s} {'fTool':5s} {'facts':5s} {'abst?':5s} {'topic':5s}"
)
for label, adp, alpha, temp, rep, mn in CONFIGS:
    use(adp)
    set_adapter_scale(model, alpha)
    R = []
    for p, cat in PROMPTS:
        t, s = gen(p, temp, rep, mn)
        R.append((p, cat, t, s))
    stop = m([s for _, _, _, s in R])
    avgl = int(m([len(t) for _, _, t, _ in R]))
    dg = m([degen(t) for _, _, t, _ in R])
    ft = m([("<tool" in t) for p, c, t, _ in R if c != "math"])
    fa = m([FACT[p] in t.lower() for p, c, t, _ in R if c == "fact"])
    au = m([ab(t) for p, c, t, _ in R if c == "unknown"])
    tp = m([(not ab(t) and not degen(t) and len(t) > 20) for p, c, t, _ in R if c == "chat"])
    print(f"{label:12s} {stop:<5} {avgl:<5} {dg:<5} {ft:<5} {fa:<5} {au:<5} {tp:<5}", flush=True)
