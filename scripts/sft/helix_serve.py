#!/usr/bin/env python3
"""OpenAI-compatible HTTP server for Helix v2 (bespoke Mamba-2/GLA/Sparse-Attn hybrid).

Helix can't run in Ollama/LocalAI/vLLM (no GGUF/HF/registered-arch path for GLA), so we
serve its NATIVE inference behind the OpenAI protocol that every client understands:

  GET  /                      -> minimal built-in chat web page
  GET  /v1/models             -> model list
  POST /v1/chat/completions   -> OpenAI chat completions (stream + non-stream)

stdlib-only (http.server) -> no pip install needed in the container. One generation at a
time (model is single-instance) via a lock. Reuses tool_harness (load + tool-use loop).

Run (in the kernel container):
  docker exec -d auralis-blackwell python /workspace/v2data/scripts/sft/helix_serve.py \\
     --checkpoint <ckpt.pt> --port 8088
Then from the LAN:  http://<host>:8088/
"""
import os, sys, json, time, threading, argparse, pathlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REPO = pathlib.Path("/workspace/v2data")
sys.path.insert(0, str(REPO / "scripts/sft")); sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("AURALIS_USE_MAMBA_KERNEL", "1")
from tool_harness import load, run_with_tools, gen_until, SYS, END  # noqa

STATE = {"model": None, "sp": None, "device": None, "tools": True, "name": "helix-v2", "lock": threading.Lock()}

PAGE = """<!doctype html><html lang="de"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Helix v2 Chat</title><style>
body{font-family:system-ui,sans-serif;max-width:760px;margin:0 auto;padding:14px;background:#0d1117;color:#e6edf3}
h1{font-size:18px;color:#7ee787} #log{border:1px solid #30363d;border-radius:8px;padding:12px;height:62vh;overflow:auto;background:#010409}
.m{margin:8px 0;white-space:pre-wrap;line-height:1.4} .u{color:#79c0ff} .a{color:#e6edf3} .meta{color:#8b949e;font-size:12px}
#row{display:flex;gap:8px;margin-top:10px} #q{flex:1;padding:10px;border-radius:8px;border:1px solid #30363d;background:#0d1117;color:#e6edf3}
button{padding:10px 16px;border-radius:8px;border:0;background:#238636;color:#fff;cursor:pointer} button:disabled{opacity:.5}
</style></head><body>
<h1>Helix v2 — Chat <span class="meta" id="model"></span></h1>
<div class="meta">0.9B from-scratch (Mamba-2/GLA/Sparse-Attn). Mathe laeuft per Tool. Befehle: leeren = Seite neu laden.</div>
<div id="log"></div>
<div id="row"><input id="q" placeholder="Nachricht an Helix..." autofocus><button id="send" onclick="go()">Senden</button></div>
<script>
let msgs=[]; const log=document.getElementById('log'), q=document.getElementById('q'), btn=document.getElementById('send');
fetch('/v1/models').then(r=>r.json()).then(d=>{document.getElementById('model').textContent='('+d.data[0].id+')'});
function add(role,text){const d=document.createElement('div');d.className='m '+(role==='user'?'u':'a');d.textContent=(role==='user'?'Du: ':'Helix: ')+text;log.appendChild(d);log.scrollTop=log.scrollHeight;return d;}
q.addEventListener('keydown',e=>{if(e.key==='Enter')go();});
async function go(){const t=q.value.trim(); if(!t)return; q.value=''; add('user',t); msgs.push({role:'user',content:t});
 btn.disabled=true; const d=add('assistant','...');
 try{const r=await fetch('/v1/chat/completions',{method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify({model:'helix',messages:msgs,stream:false})});
  const j=await r.json(); const a=j.choices[0].message.content; d.textContent='Helix: '+a; msgs.push({role:'assistant',content:a});
 }catch(e){d.textContent='Helix: [Fehler] '+e;} btn.disabled=false; q.focus();}
</script></body></html>"""


def generate(messages, max_new):
    from auralis.tokenizer.chat_template import build_inference_prompt
    msgs = [{"role": m.get("role", "user"), "content": m.get("content", "")} for m in messages if m.get("content")]
    prompt = build_inference_prompt(msgs, default_system=SYS)
    with STATE["lock"]:
        if STATE["tools"]:
            ans = run_with_tools(STATE["model"], STATE["sp"], prompt, STATE["device"], verbose=False)
        else:
            ans, _ = gen_until(STATE["model"], STATE["sp"], prompt, [END], STATE["device"], max_new=max_new)
    return ans.strip()


def completion_obj(text, finish="stop"):
    return {"id": "chatcmpl-helix", "object": "chat.completion", "created": int(time.time()),
            "model": STATE["name"], "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
            "finish_reason": finish}], "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        b = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_OPTIONS(self):
        self.send_response(204); self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*"); self.send_header("Access-Control-Allow-Methods", "*")
        self.end_headers()

    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/" or p == "/index.html":
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif p == "/v1/models":
            self._send(200, json.dumps({"object": "list", "data": [
                {"id": STATE["name"], "object": "model", "created": 0, "owned_by": "auralis"}]}))
        elif p in ("/health", "/healthz"):
            self._send(200, json.dumps({"status": "ok"}))
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        p = self.path.split("?")[0]
        if p not in ("/v1/chat/completions", "/chat/completions"):
            self._send(404, json.dumps({"error": "not found"})); return
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            self._send(400, json.dumps({"error": str(e)})); return
        messages = req.get("messages", [])
        max_new = int(req.get("max_tokens") or 256)
        stream = bool(req.get("stream"))
        try:
            text = generate(messages, max_new)
        except Exception as e:
            self._send(500, json.dumps({"error": f"{type(e).__name__}: {e}"})); return
        if not stream:
            self._send(200, json.dumps(completion_obj(text))); return
        # streaming (SSE): emit the (already-computed) answer in chunks so UIs are happy
        self.send_response(200); self.send_header("Content-Type", "text/event-stream")
        self.send_header("Access-Control-Allow-Origin", "*"); self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        def sse(obj):
            self.wfile.write(("data: " + json.dumps(obj) + "\n\n").encode("utf-8")); self.wfile.flush()
        cid = "chatcmpl-helix"
        sse({"id": cid, "object": "chat.completion.chunk", "model": STATE["name"],
             "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
        step = max(1, len(text) // 24)
        for i in range(0, len(text), step):
            sse({"id": cid, "object": "chat.completion.chunk", "model": STATE["name"],
                 "choices": [{"index": 0, "delta": {"content": text[i:i + step]}, "finish_reason": None}]})
        sse({"id": cid, "object": "chat.completion.chunk", "model": STATE["name"],
             "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
        self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model-config", default=str(REPO / "configs/model/helix_v2_1b.yaml"))
    ap.add_argument("--tokenizer", default=str(REPO / "tokenizer/helix_v2_tokenizer.model"))
    ap.add_argument("--no-tools", action="store_true")
    ap.add_argument("--name", default="helix-v2")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8088)
    a = ap.parse_args()
    import torch
    STATE["device"] = torch.device("cuda")
    STATE["tools"] = not a.no_tools
    STATE["name"] = a.name
    print(f"[helix_serve] loading {a.checkpoint} ...", flush=True)
    STATE["model"], STATE["sp"] = load(a.checkpoint, a.model_config, a.tokenizer, STATE["device"])
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"[helix_serve] OpenAI-compatible API on http://{a.host}:{a.port}  (tools={'on' if STATE['tools'] else 'off'})", flush=True)
    print(f"[helix_serve] chat page: http://<host>:{a.port}/   |   model id: {a.name}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
