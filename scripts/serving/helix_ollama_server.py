#!/usr/bin/env python3
"""Helix server speaking the two Ollama endpoints the Auralis Hub uses, with MODES.
   GET  /api/tags  -> model variants (the on/off switches live here, picked in the hub dropdown)
   POST /v1/chat/completions -> OpenAI SSE streaming

Model variants (name = the toggle):
   helix-corrective              standard (form only)
   helix-corrective-tools        + Tool-Nutzung: <tool:python> wird WIRKLICH ausgefuehrt
   helix-corrective-think        + Denken: Schritt fuer Schritt
   helix-corrective-think-tools  + beides
   helix-grounded                grounded (Kontext rein, antwortet nur daraus)
   helix-code                    schreibt Python-Funktionen
Runs in auralis-blackwell:11434. Pure PyTorch (no Ollama/llama.cpp). Stdlib only."""
import os, sys, json, re, threading, subprocess
os.environ["AURALIS_USE_CUDA_KERNELS"] = "1"
REPO = "/workspace/v2data"; sys.path.insert(0, REPO); sys.path.insert(0, REPO + "/src")
import torch, sentencepiece as spm
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from auralis.model import build_model
from auralis.adapters.lora import inject_adapters, freeze_base, load_adapter_state_dict, set_adapter_scale
CFG = REPO + "/configs/model/helix_v2_1b_flash.yaml"; CKPT = REPO + "/checkpoints/corpus20b_codeheavy/step_60000.pt"
TOK = REPO + "/tokenizer/helix_v2_tokenizer.model"
ADAPTERS = {"corrective": "sft_corrective_v3", "grounded": "sft_grounded_v4", "code": "sft_code_v3"}
# variant -> (adapter_key, lora_scale, think, tools). Lower scale = adapter less dominant = natural chat.
# Measured: corrective @1.0 over-fires the abstain/encyclopedic template (Hund=Katze, word-mangling);
# @0.5 is much more sensible while "Paris." stays crisp.
VARIANT_CFG = {
    "helix-chat":               ("code", 0.5, False, False),   # OFAT: facts 1.0; trade greeting-vs-unknown-abstain at 0.5 vs 0.6
    "helix-corrective":         ("corrective", 0.5, False, False),
    "helix-corrective-precise": ("corrective", 1.0, False, False),
    "helix-corrective-tools":   ("corrective", 1.0, False, True),
    "helix-corrective-think":   ("corrective", 0.5, True,  False),
    "helix-grounded":           ("grounded",   1.0, False, False),
    "helix-code":               ("code",       1.0, False, False),
}
VARIANTS = list(VARIANT_CFG)
SYS_BASE = "Du bist Auralis, ein hilfreicher, ehrlicher KI-Assistent."
END = "<|end|>"; PORT = 11434; MAXNEW = 360; REP = 1.2
TOOL_RE = re.compile(r"<tool:[a-zA-Z0-9_]*>\s*(.*?)\s*</tool>", re.S)

dev = torch.device("cuda"); sp = spm.SentencePieceProcessor(model_file=TOK); END_ID = sp.EncodeAsIds(END)[-1]
print("[helix-srv] loading base step_60000 ...", flush=True)
model = build_model(CFG); pl = torch.load(CKPT, map_location="cpu", weights_only=False)
model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in pl["model"].items()}, strict=False)
model = model.to(dev); inject_adapters(model, r=64, alpha=128, kind="lora"); freeze_base(model)
emb = getattr(model, "embedding", None) or getattr(model, "embed_tokens", None); model.eval()
ADP_STATE = {}
for name, d in ADAPTERS.items():
    p = REPO + f"/checkpoints/{d}/adapter_best.pt"
    if os.path.exists(p): ADP_STATE[name] = torch.load(p, map_location="cpu"); print(f"[helix-srv] adapter ready: {name}", flush=True)
GPU_LOCK = threading.Lock(); _cur = {"name": None}

def apply_adapter(name):
    ck = ADP_STATE[name]; load_adapter_state_dict(model, ck["adapter"]); model.to(dev)
    for i, tid in enumerate(ck["emb_ids"]): emb.weight.data[tid] = ck["emb_rows"][i].to(emb.weight.device, emb.weight.dtype)
    set_adapter_scale(model, 1.0); _cur["name"] = name

def parse_model(name):
    return VARIANT_CFG.get(name, VARIANT_CFG["helix-corrective"])  # (adapter_key, scale, think, tools)

def run_tool(code):
    prev = None  # interpret arithmetic 'x'/'×' between numbers as Python '*' (model copies it from the question)
    while prev != code:
        prev = code; code = re.sub(r"(\d)\s*[x×]\s*(\d)", r"\1*\2", code)
    try:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=5)
        out = (r.stdout or "").strip()
        if out: return out[:400]
        err = (r.stderr or "").strip().splitlines()
        return ("Fehler: " + err[-1])[:200] if err else "(kein Output)"
    except subprocess.TimeoutExpired: return "Fehler: Zeitueberschreitung"
    except Exception as e: return f"Fehler: {e}"

def build_sys(messages, think, tools):
    s = SYS_BASE
    for m in messages:
        if m.get("role") == "system" and (m.get("content") or "").strip():
            s = m["content"].strip(); break
    if tools:
        s += (" Bei jeder Rechnung rechne NICHT im Kopf, sondern benutze das Werkzeug: schreibe genau"
              " <tool:python>print(<Ausdruck mit den Originalzahlen aus der Frage>)</tool> und warte auf das Ergebnis."
              " Benutze gueltigen Python-Code: * fuer Mal, / fuer Geteilt.")
    if think:
        s += " Denke zuerst kurz Schritt fuer Schritt nach, dann gib die endgueltige Antwort."
    return s

def render(messages, think, tools):
    sysmsg = build_sys(messages, think, tools); body = []
    for m in messages[-8:]:
        r = m.get("role"); c = (m.get("content") or "").strip()
        if r == "user": body.append(f"<|user|>\n{c}\n{END}\n")
        elif r == "assistant": body.append(f"<|assistant|>\n{c}\n{END}\n")
    return f"<|system|>\n{sysmsg}\n{END}\n" + "".join(body) + "<|assistant|>\n"

# --- query rewriter: bare fragment ("katze") -> clear question ("Was ist Katze?"). 0.9B is phrasing-fragile;
#     this is normalization, not training. Skips math, full sentences, questions/commands, grounded & code.
QWORDS = ("was", "wer", "wie", "wo", "wann", "warum", "welche", "welcher", "welches", "nenne", "schreibe",
          "erklaere", "erklre", "gib", "liste", "zeige", "kannst", "ist", "sind", "hallo", "hi", "hey",
          "moin", "danke", "ja", "nein", "ok", "bitte", "rechne")
def looks_bare(t):
    s = (t or "").strip()
    if not s or "?" in s or "." in s or "\n" in s or "," in s: return False
    if any(c.isdigit() for c in s): return False
    w = s.split()
    if len(w) == 0 or len(w) > 3: return False
    if w[0].lower().rstrip(":") in QWORDS: return False
    return True
ACRONYMS = {"gpu", "cpu", "ram", "vram", "ai", "ki", "llm", "api", "ssd", "hdd", "os", "url", "sql", "usb", "ip", "dns"}
def rewrite_term(s):  # "katze"->"Katze", "gpu"->"GPU", "beste gpu"->"Beste GPU"
    words = s.strip().split(); out = []
    for i, w in enumerate(words):
        if w.lower() in ACRONYMS: out.append(w.upper())
        elif i == 0: out.append(w[0].upper() + w[1:])
        else: out.append(w)
    return " ".join(out)
def maybe_rewrite(name, messages):
    if name in ("helix-grounded", "helix-code"): return messages
    msgs = [dict(m) for m in messages]
    for m in reversed(msgs):
        if m.get("role") == "user":
            c = m.get("content") or ""
            if looks_bare(c): m["content"] = f"Was ist {rewrite_term(c)}?"
            break
    return msgs

def gen_stream(name, messages):
    base, scale, think, tools = parse_model(name)
    messages = maybe_rewrite(name, messages)
    with GPU_LOCK:
        if _cur["name"] != base: apply_adapter(base)
        set_adapter_scale(model, scale)  # variant-specific LoRA strength (alpha)
        ids = sp.EncodeAsIds(render(messages, think, tools)); x = torch.tensor([ids], device=dev)
        out = []; prev = ""
        with torch.no_grad():
            for _ in range(MAXNEW):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16): lg = model(input_ids=x)["logits"][0, -1].float()
                for t in set(out): lg[t] = lg[t]/REP if lg[t] > 0 else lg[t]*REP
                nid = int(torch.argmax(lg))
                if nid == END_ID: break
                out.append(nid); x = torch.cat([x, torch.tensor([[nid]], device=dev)], 1)
                full = sp.DecodeIds(out)
                if END in full: full = full.split(END)[0]
                delta = full[len(prev):]; prev = full
                if delta: yield delta
                # --- tool execution: run a just-completed <tool:...></tool> and inject the REAL result ---
                if tools and full.rstrip().endswith("</tool>"):
                    m = TOOL_RE.findall(full)
                    if m:
                        result = run_tool(m[-1])
                        inj = f"\n<result>\n{result}\n</result>\n"
                        yield inj
                        inj_ids = sp.EncodeAsIds(inj)
                        x = torch.cat([x, torch.tensor([inj_ids], device=dev)], 1)
                        out.extend(inj_ids); prev = sp.DecodeIds(out)
                        if END in prev: prev = prev.split(END)[0]
                if END in sp.DecodeIds(out[-4:]): break

class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"
    def log_message(self, *a): pass
    def _json(self, obj, code=200):
        b = json.dumps(obj).encode(); self.send_response(code)
        self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b)))
        self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path.startswith("/api/tags"):
            self._json({"models": [{"name": n, "model": n, "modified_at": "2024-01-01T00:00:00Z", "size": 0, "digest": "helix-" + n} for n in VARIANTS]})
        elif self.path.startswith("/api/version"):
            self._json({"version": "helix-0.9b-facade"})
        else:
            self._json({"status": "ok", "models": VARIANTS})
    def do_POST(self):
        ln = int(self.headers.get("Content-Length", "0")); raw = self.rfile.read(ln) if ln else b"{}"
        try: req = json.loads(raw or b"{}")
        except Exception: req = {}
        if not self.path.startswith("/v1/chat/completions"):
            return self._json({"error": "not found"}, 404)
        model_name = req.get("model", "helix-corrective"); messages = req.get("messages", [])
        self.send_response(200); self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache"); self.send_header("Connection", "close"); self.end_headers()
        cid = "chatcmpl-helix"
        def send(d):
            self.wfile.write(("data: " + json.dumps(d) + "\n\n").encode()); self.wfile.flush()
        try:
            send({"id": cid, "object": "chat.completion.chunk", "model": model_name, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
            for delta in gen_stream(model_name, messages):
                send({"id": cid, "object": "chat.completion.chunk", "model": model_name, "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}]})
            send({"id": cid, "object": "chat.completion.chunk", "model": model_name, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
            self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
        except Exception as e:
            try: send({"id": cid, "choices": [{"index": 0, "delta": {"content": f"[helix error: {e}]"}, "finish_reason": "stop"}]}); self.wfile.write(b"data: [DONE]\n\n")
            except Exception: pass

print(f"[helix-srv] ready on :{PORT}  variants={VARIANTS}", flush=True)
ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
