#!/usr/bin/env python3
"""Interaktiver Helix-Chat im Terminal. Laedt step_60000 + ein gewaehltes Adapter.
Start (interaktiv):
  ssh -t root@BITBASTION "docker exec -it auralis-blackwell python /workspace/v2data/diag/chat_helix.py"
Env:
  ADAPTER=corrective|grounded|code  (oder voller .pt-Pfad)   default: corrective (sauberster Chat)
  TEMP=0.0 (greedy, default) | z.B. 0.7 fuer Sampling ; MAXNEW=220 ; REP=1.2
Befehle im Chat:  exit / quit zum Beenden."""
import os, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ["AURALIS_USE_CUDA_KERNELS"] = "1"
REPO = "/workspace/v2data"; sys.path.insert(0, REPO); sys.path.insert(0, REPO + "/src")
import torch, sentencepiece as spm
from auralis.model import build_model
from auralis.adapters.lora import inject_adapters, freeze_base, load_adapter_state_dict, set_adapter_scale
CFG = REPO + "/configs/model/helix_v2_1b_flash.yaml"; CKPT = REPO + "/checkpoints/corpus20b_codeheavy/step_60000.pt"
TOK = REPO + "/tokenizer/helix_v2_tokenizer.model"
MAP = {"corrective": "sft_corrective_v3", "instruct": "sft_corrective_v3",
       "grounded": "sft_grounded_v4", "code": "sft_code_v3"}
sel = os.environ.get("ADAPTER", "corrective")
ADP = sel if sel.endswith(".pt") else REPO + f"/checkpoints/{MAP.get(sel, sel)}/adapter_best.pt"
TEMP = float(os.environ.get("TEMP", "0.0")); MAXNEW = int(os.environ.get("MAXNEW", "220")); REP = float(os.environ.get("REP", "1.2"))
SYS = "Du bist Auralis, ein hilfreicher, ehrlicher KI-Assistent."; ASST = "<|assistant|>\n"; END = "<|end|>"
print(f"Lade Helix (step_60000) + Adapter: {ADP}", flush=True)
if not os.path.exists(ADP):
    print(f"!! Adapter nicht gefunden. Verfuegbar:"); os.system("ls /workspace/v2data/checkpoints/ | grep -E 'sft_|code_'"); sys.exit(1)
sp = spm.SentencePieceProcessor(model_file=TOK); dev = torch.device("cuda"); END_ID = sp.EncodeAsIds(END)[-1]
model = build_model(CFG); pl = torch.load(CKPT, map_location="cpu", weights_only=False)
model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in pl["model"].items()}, strict=False)
model = model.to(dev); inject_adapters(model, r=64, alpha=128, kind="lora"); freeze_base(model)
ck = torch.load(ADP, map_location="cpu"); load_adapter_state_dict(model, ck["adapter"])
emb = getattr(model, "embedding", None) or getattr(model, "embed_tokens", None)
for i, tid in enumerate(ck["emb_ids"]): emb.weight.data[tid] = ck["emb_rows"][i].to(emb.weight.device, emb.weight.dtype)
model = model.to(dev).eval(); set_adapter_scale(model, 1.0)

def gen(q):
    ids = sp.EncodeAsIds(f"<|system|>\n{SYS}\n{END}\n<|user|>\n{q}\n{END}\n{ASST}")
    x = torch.tensor([ids], device=dev); out = []
    with torch.no_grad():
        for _ in range(MAXNEW):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16): lg = model(input_ids=x)["logits"][0, -1].float()
            for t in set(out): lg[t] = lg[t]/REP if lg[t] > 0 else lg[t]*REP
            if TEMP > 0:
                lg = lg/TEMP; v, ix = torch.topk(lg, 40); p = torch.softmax(v, -1); nid = int(ix[torch.multinomial(p, 1)])
            else:
                nid = int(torch.argmax(lg))
            if nid == END_ID: break
            out.append(nid); x = torch.cat([x, torch.tensor([[nid]], device=dev)], 1)
            if END in sp.DecodeIds(out[-4:]): break
    return sp.DecodeIds(out).split(END)[0].strip()

print(f"\n== Helix bereit ==  Adapter={sel}  TEMP={TEMP} ({'greedy' if TEMP == 0 else 'sampling'})  | 'exit' beendet\n", flush=True)
while True:
    try:
        q = input("Du> ").strip()
    except (EOFError, KeyboardInterrupt):
        print(); break
    if not q: continue
    if q.lower() in ("exit", "quit", ":q"): break
    print("Helix> " + gen(q) + "\n", flush=True)
