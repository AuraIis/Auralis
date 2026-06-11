#!/usr/bin/env python3
"""Alpha-sweep a trained adapter on a FROZEN base — the modular inference dial.
alpha=0 -> pure base ; alpha=1 -> full adapter skill. Tests that the skill turns ON
(code) while the base behaviour (honesty/chat) stays intact at low alpha."""
import os, sys, argparse, pathlib
REPO = pathlib.Path("/workspace/v2data")
sys.path.insert(0, str(REPO / "scripts/sft")); sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("AURALIS_USE_MAMBA_KERNEL", "1")
import torch
import sentencepiece as spm
from auralis.model import build_model
from auralis.adapters import inject_adapters, load_adapter_state_dict, set_adapter_scale
from auralis.tokenizer.chat_template import build_inference_prompt
from tool_harness import gen_until, SYS, END  # noqa

PROMPTS = [
    ("Schreibe eine Python-Funktion, die prueft ob eine Zahl gerade ist.", "CODE"),
    ("Schreibe eine Funktion, die die Summe einer Liste von Zahlen berechnet.", "CODE"),
    ("wer ist einstein?", "HONESTY (abstain soll bleiben)"),
    ("Was ist die Hauptstadt von Deutschland?", "CHAT (Berlin soll bleiben)"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--adapter-r", type=int, default=16)
    ap.add_argument("--adapter-alpha", type=float, default=32.0)
    ap.add_argument("--kind", default="lora")
    ap.add_argument("--alphas", default="0.0,0.5,1.0")
    ap.add_argument("--model-config", default=str(REPO / "configs/model/helix_v2_1b.yaml"))
    ap.add_argument("--tokenizer", default=str(REPO / "tokenizer/helix_v2_tokenizer.model"))
    a = ap.parse_args()
    dev = torch.device("cuda")
    sp = spm.SentencePieceProcessor(model_file=a.tokenizer)
    model = build_model(a.model_config).to(dev).eval()
    bp = torch.load(a.base, map_location=dev, weights_only=False)
    model.load_state_dict(bp.get("model", bp.get("state_dict", bp)), strict=False)
    inject_adapters(model, r=a.adapter_r, alpha=a.adapter_alpha, kind=a.kind)
    model = model.to(dev).eval()
    load_adapter_state_dict(model, torch.load(a.adapter, map_location=dev, weights_only=False)["adapter"])
    print(f"base {pathlib.Path(a.base).name} + {a.kind} r={a.adapter_r} ({pathlib.Path(a.adapter).name})", flush=True)
    for alpha in [float(x) for x in a.alphas.split(",")]:
        set_adapter_scale(model, alpha)
        print(f"\n========== alpha = {alpha:.2f} ==========", flush=True)
        for q, why in PROMPTS:
            prompt = build_inference_prompt([{"role": "user", "content": q}], default_system=SYS)
            ans, _ = gen_until(model, sp, prompt, [END], dev, max_new=160)
            print(f"[{why}] {q}\n  -> {' '.join(ans.strip().split())[:240]}", flush=True)


if __name__ == "__main__":
    main()
