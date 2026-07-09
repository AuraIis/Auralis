#!/usr/bin/env python3
"""Quick broad test of an SFT checkpoint with PROPER decoding (chat template +
repetition penalty + sampling). Throws fresh questions (not in train/eval) at it."""

import argparse
import os
import pathlib
import sys

import sentencepiece as spm
import torch

os.environ.setdefault("AURALIS_USE_MAMBA_KERNEL", "1")
REPO = pathlib.Path("/workspace/v2data")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
from auralis.model import build_model
from auralis.tokenizer.chat_template import build_inference_prompt

SYS = "Du bist Auralis, ein hilfreicher, ehrlicher KI-Assistent."
QUESTIONS = [
    "Was ist die Hauptstadt von Oesterreich?",
    "Erklaere kurz, was ein Vulkan ist.",
    "Wer hat die Gluehbirne erfunden?",
    "Was ist schwerer, 1 kg Eisen oder 1 kg Federn?",
    "Ist Pluto ein Planet?",
    "Wie viele Tage hat ein Schaltjahr?",
    "What is the capital of Spain?",
    "Nenne drei Bundeslaender in Deutschland.",
    "Was ist 12 plus 15?",
    "Warum muss man Zaehne putzen?",
]


def generate(model, sp, prompt, device, max_new=90, temperature=0.0, top_k=0, rep_pen=1.3):
    end_id = sp.EncodeAsIds("<|end|>")[-1]
    ids = sp.EncodeAsIds(prompt)
    inp = torch.tensor([ids], device=device)
    gen = []
    for _ in range(max_new):
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(input_ids=inp)["logits"][0, -1].float()
        for t in set(gen):
            logits[t] = logits[t] / rep_pen if logits[t] > 0 else logits[t] * rep_pen
        if temperature and temperature > 0:
            logits = logits / temperature
            if top_k > 0:
                v, i = torch.topk(logits, min(top_k, logits.numel()))
                nxt = int(i[torch.multinomial(torch.softmax(v, -1), 1)].item())
            else:
                nxt = int(torch.multinomial(torch.softmax(logits, -1), 1).item())
        else:
            nxt = int(torch.argmax(logits).item())
        if nxt == end_id:
            break
        gen.append(nxt)
        inp = torch.cat([inp, torch.tensor([[nxt]], device=device)], 1)
    return sp.DecodeIds(gen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model-config", default=str(REPO / "configs/model/helix_v2_1b.yaml"))
    ap.add_argument("--tokenizer", default=str(REPO / "tokenizer/helix_v2_tokenizer.model"))
    a = ap.parse_args()
    torch.manual_seed(42)
    device = torch.device("cuda")
    sp = spm.SentencePieceProcessor(model_file=a.tokenizer)
    print("building model...", flush=True)
    model = build_model(a.model_config).to(device).eval()
    payload = torch.load(a.checkpoint, map_location=device, weights_only=False)
    state = payload.get("model", payload.get("state_dict", payload))
    miss, extra = model.load_state_dict(state, strict=False)
    print(f"loaded {a.checkpoint} | missing={len(miss)} extra={len(extra)}", flush=True)
    for q in QUESTIONS:
        prompt = build_inference_prompt([{"role": "user", "content": q}], default_system=SYS)
        g = generate(model, sp, prompt, device, temperature=0.0, rep_pen=1.3)
        s = generate(model, sp, prompt, device, temperature=0.7, top_k=40, rep_pen=1.3)
        print("=" * 66)
        print("Q:", q)
        print("-- greedy+rep1.3 :", g.strip()[:400])
        print("-- sample t0.7   :", s.strip()[:400])


if __name__ == "__main__":
    main()
