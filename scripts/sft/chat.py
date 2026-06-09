#!/usr/bin/env python3
"""Interactive chat REPL for Helix v2 (bespoke Mamba-2/GLA/Sparse-Attn hybrid).

Helix can't run in Ollama/llama.cpp (no GLA kernel / GGUF converter for this arch),
so we run it natively where its CUDA kernels live (Blackwell) and chat over SSH:

  ssh bitbastion "docker exec -it auralis-blackwell \\
     python /workspace/v2data/scripts/sft/chat.py --checkpoint <ckpt.pt>"

Reuses tool_harness (load + tool-use generation loop + chat template). With --tools
(default for tool-SFT checkpoints) math questions are answered via the safe calculator.
"""
import os, sys, argparse, pathlib

REPO = pathlib.Path("/workspace/v2data")
sys.path.insert(0, str(REPO / "scripts/sft")); sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("AURALIS_USE_MAMBA_KERNEL", "1")

from tool_harness import load, run_with_tools, gen_until, SYS, END  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model-config", default=str(REPO / "configs/model/helix_v2_1b.yaml"))
    ap.add_argument("--tokenizer", default=str(REPO / "tokenizer/helix_v2_tokenizer.model"))
    ap.add_argument("--no-tools", action="store_true", help="disable the calculator tool loop")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--system", default=SYS)
    ap.add_argument("--once", default=None, help="answer one prompt and exit (for testing)")
    a = ap.parse_args()

    import torch
    from auralis.tokenizer.chat_template import build_inference_prompt
    device = torch.device("cuda")
    print(f"Lade Helix aus {a.checkpoint} ...", flush=True)
    model, sp = load(a.checkpoint, a.model_config, a.tokenizer, device)
    use_tools = not a.no_tools
    print("=" * 64)
    print(f"Helix v2 — Chat (tools={'an' if use_tools else 'aus'}). "
          f"{'Eine Frage, dann Ende.' if a.once else 'Tippe deine Nachricht. /reset leert den Verlauf, /exit beendet.'}")
    print("Hinweis: 0.9B from-scratch Modell — erwarte keinen GPT-4, aber gib ihm eine Chance.")
    print("=" * 64, flush=True)

    messages = []

    def answer(user_text):
        messages.append({"role": "user", "content": user_text})
        prompt = build_inference_prompt(messages, default_system=a.system)
        if use_tools:
            ans = run_with_tools(model, sp, prompt, device, verbose=False)
        else:
            ans, _ = gen_until(model, sp, prompt, [END], device, max_new=a.max_new)
        ans = ans.strip()
        messages.append({"role": "assistant", "content": ans})
        return ans

    if a.once is not None:
        print("Du:", a.once)
        print("Helix:", answer(a.once))
        return

    while True:
        try:
            user = input("\nDu: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nTschuess."); break
        if not user:
            continue
        if user in ("/exit", "/quit"):
            print("Tschuess."); break
        if user == "/reset":
            messages.clear(); print("(Verlauf geleert)"); continue
        print("Helix:", answer(user), flush=True)


if __name__ == "__main__":
    main()
