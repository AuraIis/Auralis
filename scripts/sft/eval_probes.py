#!/usr/bin/env python3
"""Quick diagnostic probes for a Helix chat checkpoint — the exact failures we saw live
on the old tool_sft_v12 (stone-salad, hallo->math, Walhai-Einstein, no abstain).
Loads once, runs a fixed prompt list. tools on (so 'hallo' must NOT trigger a tool)."""
import os, sys, argparse, pathlib
REPO = pathlib.Path("/workspace/v2data")
sys.path.insert(0, str(REPO / "scripts/sft")); sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("AURALIS_USE_MAMBA_KERNEL", "1")
from tool_harness import load, run_with_tools, gen_until, SYS, END  # noqa

PROBES = [
    ("hallo", "Begruessung -> KEIN Tool/keine erfundene Mathe"),
    ("Was ist 47 mal 6?", "Mathe -> Tool, 282"),
    ("wer ist einstein?", "Wissen/Unsicher -> kohaerent statt 'Walhai-System'"),
    ("Was ist die Hauptstadt von Deutschland?", "Faktenfrage"),
    ("Wer war Moxthal Vornurr?", "erfundene Entitaet -> abstain"),
    ("Erklaere kurz, was Photosynthese ist.", "Erklaeren -> kohaerent"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model-config", default=str(REPO / "configs/model/helix_v2_1b.yaml"))
    ap.add_argument("--tokenizer", default=str(REPO / "tokenizer/helix_v2_tokenizer.model"))
    ap.add_argument("--no-tools", action="store_true")
    a = ap.parse_args()
    import torch
    from auralis.tokenizer.chat_template import build_inference_prompt
    dev = torch.device("cuda")
    print(f"=== probes on {pathlib.Path(a.checkpoint).name} (tools={'off' if a.no_tools else 'on'}) ===", flush=True)
    model, sp = load(a.checkpoint, a.model_config, a.tokenizer, dev)
    for q, why in PROBES:
        prompt = build_inference_prompt([{"role": "user", "content": q}], default_system=SYS)
        if a.no_tools:
            ans, _ = gen_until(model, sp, prompt, [END], dev, max_new=160)
        else:
            ans = run_with_tools(model, sp, prompt, dev, verbose=False)
        ans = " ".join(ans.strip().split())[:280]
        print(f"\n[{why}]\nQ: {q}\nA: {ans}", flush=True)


if __name__ == "__main__":
    main()
