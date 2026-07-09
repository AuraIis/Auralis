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

import json
import os
import re
import subprocess
import sys
import threading

os.environ["AURALIS_USE_CUDA_KERNELS"] = "1"
REPO = "/workspace/v2data"
sys.path.insert(0, REPO)
sys.path.insert(0, REPO + "/src")
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

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
ADAPTERS = {"corrective": "sft_corrective_v3", "grounded": "sft_grounded_v4", "code": "sft_code_v3"}
# variant -> (adapter_key, lora_scale, think, tools). Lower scale = adapter less dominant = natural chat.
# Measured: corrective @1.0 over-fires the abstain/encyclopedic template (Hund=Katze, word-mangling);
# @0.5 is much more sensible while "Paris." stays crisp.
VARIANT_CFG = {
    "helix": (
        "code",
        0.5,
        False,
        False,
    ),  # AUTO-ROUTER: dispatches to math/rag/web/code/chat by query (default placeholder cfg)
    "helix-chat": (
        "code",
        0.5,
        False,
        False,
    ),  # OFAT: facts 1.0; trade greeting-vs-unknown-abstain at 0.5 vs 0.6
    "helix-rag": (
        "grounded",
        1.0,
        False,
        False,
    ),  # local Wikipedia (FTS5) -> grounded reader (honest: abstains vs confabulates; corrective@0.5 extracts more but invents facts on messy/fake context)
    "helix-web": (
        "grounded",
        1.0,
        False,
        False,
    ),  # LIVE web search (DuckDuckGo) -> grounded reader; honest over confident-wrong
    "helix-corrective": ("corrective", 0.5, False, False),
    "helix-corrective-precise": ("corrective", 1.0, False, False),
    "helix-corrective-tools": ("corrective", 1.0, False, True),
    "helix-corrective-think": ("corrective", 0.5, True, False),
    "helix-grounded": ("grounded", 1.0, False, False),
    "helix-code": ("code", 1.0, False, False),
}
VARIANTS = list(VARIANT_CFG)
SYS_BASE = "Du bist Auralis, ein hilfreicher, ehrlicher KI-Assistent."
END = "<|end|>"
PORT = 11434
MAXNEW = 360
REP = 1.2
TOOL_RE = re.compile(r"<tool:[a-zA-Z0-9_]*>\s*(.*?)\s*</tool>", re.S)

dev = torch.device("cuda")
sp = spm.SentencePieceProcessor(model_file=TOK)
END_ID = sp.EncodeAsIds(END)[-1]
print("[helix-srv] loading base step_60000 ...", flush=True)
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
ADP_STATE = {}
for name, d in ADAPTERS.items():
    p = REPO + f"/checkpoints/{d}/adapter_best.pt"
    if os.path.exists(p):
        ADP_STATE[name] = torch.load(p, map_location="cpu")
        print(f"[helix-srv] adapter ready: {name}", flush=True)
GPU_LOCK = threading.Lock()
_cur = {"name": None}


def apply_adapter(name):
    ck = ADP_STATE[name]
    load_adapter_state_dict(model, ck["adapter"])
    model.to(dev)
    for i, tid in enumerate(ck["emb_ids"]):
        emb.weight.data[tid] = ck["emb_rows"][i].to(emb.weight.device, emb.weight.dtype)
    set_adapter_scale(model, 1.0)
    _cur["name"] = name


def parse_model(name):
    return VARIANT_CFG.get(
        name, VARIANT_CFG["helix-corrective"]
    )  # (adapter_key, scale, think, tools)


def run_tool(code):
    prev = None  # interpret arithmetic 'x'/'×' between numbers as Python '*' (model copies it from the question)
    while prev != code:
        prev = code
        code = re.sub(r"(\d)\s*[x×]\s*(\d)", r"\1*\2", code)
    try:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=5)
        out = (r.stdout or "").strip()
        if out:
            return out[:400]
        err = (r.stderr or "").strip().splitlines()
        return ("Fehler: " + err[-1])[:200] if err else "(kein Output)"
    except subprocess.TimeoutExpired:
        return "Fehler: Zeitueberschreitung"
    except Exception as e:
        return f"Fehler: {e}"


def build_sys(messages, think, tools):
    s = SYS_BASE
    for m in messages:
        if m.get("role") == "system" and (m.get("content") or "").strip():
            s = m["content"].strip()
            break
    if tools:
        s += (
            " Bei jeder Rechnung rechne NICHT im Kopf, sondern benutze das Werkzeug: schreibe genau"
            " <tool:python>print(<Ausdruck mit den Originalzahlen aus der Frage>)</tool> und warte auf das Ergebnis."
            " Benutze gueltigen Python-Code: * fuer Mal, / fuer Geteilt."
        )
    if think:
        s += " Denke zuerst kurz Schritt fuer Schritt nach, dann gib die endgueltige Antwort."
    return s


def render(messages, think, tools):
    sysmsg = build_sys(messages, think, tools)
    body = []
    for m in messages[-8:]:
        r = m.get("role")
        c = (m.get("content") or "").strip()
        if r == "user":
            body.append(f"<|user|>\n{c}\n{END}\n")
        elif r == "assistant":
            body.append(f"<|assistant|>\n{c}\n{END}\n")
    return f"<|system|>\n{sysmsg}\n{END}\n" + "".join(body) + "<|assistant|>\n"


# --- query rewriter: bare fragment ("katze") -> clear question ("Was ist Katze?"). 0.9B is phrasing-fragile;
#     this is normalization, not training. Skips math, full sentences, questions/commands, grounded & code.
QWORDS = (
    "was",
    "wer",
    "wie",
    "wo",
    "wann",
    "warum",
    "welche",
    "welcher",
    "welches",
    "nenne",
    "schreibe",
    "erklaere",
    "erklre",
    "gib",
    "liste",
    "zeige",
    "kannst",
    "ist",
    "sind",
    "hallo",
    "hi",
    "hey",
    "moin",
    "danke",
    "ja",
    "nein",
    "ok",
    "bitte",
    "rechne",
)


def looks_bare(t):
    s = (t or "").strip()
    if not s or "?" in s or "." in s or "\n" in s or "," in s:
        return False
    if any(c.isdigit() for c in s):
        return False
    w = s.split()
    if len(w) == 0 or len(w) > 3:
        return False
    if w[0].lower().rstrip(":") in QWORDS:
        return False
    return True


ACRONYMS = {
    "gpu",
    "cpu",
    "ram",
    "vram",
    "ai",
    "ki",
    "llm",
    "api",
    "ssd",
    "hdd",
    "os",
    "url",
    "sql",
    "usb",
    "ip",
    "dns",
}


def rewrite_term(s):  # "katze"->"Katze", "gpu"->"GPU", "beste gpu"->"Beste GPU"
    words = s.strip().split()
    out = []
    for i, w in enumerate(words):
        if w.lower() in ACRONYMS:
            out.append(w.upper())
        elif i == 0:
            out.append(w[0].upper() + w[1:])
        else:
            out.append(w)
    return " ".join(out)


def maybe_rewrite(name, messages):
    if name in ("helix-grounded", "helix-code"):
        return messages
    msgs = [dict(m) for m in messages]
    for m in reversed(msgs):
        if m.get("role") == "user":
            c = m.get("content") or ""
            if looks_bare(c):
                m["content"] = f"Was ist {rewrite_term(c)}?"
            break
    return msgs


import sqlite3

RAG_DB = "/workspace/v2data/rag/dewiki.fts5.db"
_rag = {"con": None}
RAG_STOP = {
    "was",
    "ist",
    "eine",
    "ein",
    "einen",
    "der",
    "die",
    "das",
    "wer",
    "wie",
    "wo",
    "wann",
    "warum",
    "welche",
    "welcher",
    "welches",
    "mir",
    "mich",
    "du",
    "kannst",
    "ueber",
    "über",
    "sagen",
    "erklaere",
    "erkläre",
    "kurz",
    "nenne",
    "sind",
    "den",
    "dem",
    "des",
    "von",
    "im",
    "in",
    "zu",
    "und",
    "mit",
    "auf",
    "fuer",
    "für",
    "etwas",
    "mal",
    "bitte",
}


def rag_terms(q):
    ks = [w for w in re.findall(r"[\wäöüÄÖÜß]+", q.lower()) if w not in RAG_STOP and len(w) > 1]
    return " OR ".join(ks)


_BAD = (
    "steht für",
    "begriffsklärung",
    "ist der familienname",
    "ist der name mehrerer",
    "ist der name folgender",
    "ist der name von",
    "bezeichnet:",
    "ist eine ortsbezeichnung",
    "kann sich beziehen",
    "ist eine liste",
    "ist eine begriffsklärung",
    "bezeichnet mehrere",
)


def _good(ti, intro):  # drop disambiguation / name-list pages (bad context, confuse the reader)
    if "Begriffsklärung" in ti or "(Familienname)" in ti:
        return False
    h = intro[:130].lower()
    return not any(b in h for b in _BAD)


def _title_cands(q):  # capitalized / acronym-upper main terms, e.g. "Paris", "GPU", "Hardware"
    ws = [w for w in re.findall(r"[\wäöüÄÖÜß]+", q) if w.lower() not in RAG_STOP and len(w) > 1]
    c = [(w.upper() if w.lower() in ACRONYMS else w[0].upper() + w[1:]) for w in ws]
    if len(ws) > 1:
        c.append(
            " ".join((w.upper() if w.lower() in ACRONYMS else w[0].upper() + w[1:]) for w in ws)
        )
    return c


def rag_retrieve(q, k=3):
    if _rag["con"] is None:
        try:
            _rag["con"] = sqlite3.connect(
                f"file:{RAG_DB}?mode=ro", uri=True, check_same_thread=False
            )
        except Exception:
            return []
    con = _rag["con"]
    out = []
    seen = set()
    for term in _title_cands(
        q
    ):  # HARD RULE: alias-resolve (Katze->Hauskatze) then exact title beats niche/disambig
        tgt = term
        try:
            a = con.execute(
                "SELECT target FROM aliasmap WHERE a = ? LIMIT 1", (term.lower(),)
            ).fetchone()
            if a:
                tgt = a[0]
        except Exception:
            pass
        try:
            r = con.execute(
                "SELECT t, intro FROM titlemap WHERE t = ? COLLATE NOCASE LIMIT 1", (tgt,)
            ).fetchone()
        except Exception:
            r = None
        if r and _good(r[0], r[1]) and r[0].lower() not in seen:
            out.append((r[0], r[1]))
            seen.add(r[0].lower())
    if out:
        return out[:k]  # exact/alias hit -> clean single context, no BM25 dilution
    t = rag_terms(q)  # BM25 fallback only when there is no exact/alias article
    if t:
        try:
            rows = con.execute(
                "SELECT title, intro FROM docs WHERE docs MATCH ? ORDER BY bm25(docs) LIMIT ?",
                (t, k * 15),
            ).fetchall()
        except Exception:
            rows = []
        for ti, i in rows:
            if (
                _good(ti, i) and len(i) >= 80 and ti.lower() not in seen
            ):  # skip extremely short niche stubs
                out.append((ti, i))
                seen.add(ti.lower())
    return out[:k]


_ddgs = {"cls": None}


def web_search(q, k=4):  # live DuckDuckGo (no API key); snippets as grounded context
    try:
        if _ddgs["cls"] is None:
            from ddgs import DDGS

            _ddgs["cls"] = DDGS
        rows = list(_ddgs["cls"]().text(q, max_results=k, region="de-de"))
        return [
            ((x.get("title") or "").strip(), (x.get("body") or "").strip())
            for x in rows
            if (x.get("body") or "").strip()
        ]
    except Exception:
        return []


def route(q):  # AUTO-ROUTER: pick the right mode from the query
    ql = (q or "").lower().strip()
    if (
        re.search(r"\d[\d.,]*\s*(\*|x|×|·|/|:|\+|mal|geteilt|plus|minus|hoch|durch)\s*\d", ql)
        or ql.startswith(("rechne", "berechne"))
        or "wie viel ist" in ql
        or "wieviel ist" in ql
    ):
        return "math"
    if any(
        w in ql
        for w in (
            "schreibe eine funktion",
            "schreib eine funktion",
            "python-funktion",
            "funktion die",
            "programmiere",
            "schreibe code",
            "schreib code",
        )
    ):
        return "code"
    if any(
        w in ql
        for w in (
            "aktuell",
            "heute",
            "derzeit",
            "momentan",
            "neueste",
            "jüngste",
            "dieses jahr",
            "gerade ",
        )
    ) or re.search(r"\b20(1[5-9]|2[0-9])\b", ql):
        return "web"
    if re.match(
        r"^(was ist|was sind|wer ist|wer war|wer sind|wer hat|erklaer|erkläre|was bedeutet|wie funktioniert|wie viele|wo liegt|wo ist|welche|welcher|welches|definiere)\b",
        ql,
    ) or looks_bare(q):
        return "rag"
    return "chat"


def gen_stream(name, messages):
    requested = name
    q0 = next(
        (m.get("content", "").strip() for m in reversed(messages) if m.get("role") == "user"), ""
    )
    if name == "helix":
        name = {
            "math": "helix-corrective-tools",
            "code": "helix-code",
            "chat": "helix-chat",
            "rag": "helix-rag",
            "web": "helix-web",
        }[route(q0)]
    base, scale, think, tools = parse_model(name)
    if name in ("helix-rag", "helix-web"):
        q = q0
        if looks_bare(q):
            q = f"Was ist {rewrite_term(q)}?"
        hits = web_search(q, 4) if name == "helix-web" else rag_retrieve(q, 3)
        if not hits and name == "helix-rag" and requested == "helix":
            hits = web_search(q, 4)  # auto-router: fall back to web for unknown/current
        if not hits:
            yield (
                "Ich habe online nichts dazu gefunden."
                if name == "helix-web"
                else "Dazu finde ich keinen passenden Eintrag — ich weiss es nicht."
            )
            return
        ctx = "\n\n".join(f"{t}: {i[:380]}" for t, i in hits)[
            :1500
        ]  # short per-doc snippet = cleaner for the 0.9B reader
        messages = [{"role": "user", "content": f"{ctx}\n\nFrage: {q}"}]
    else:
        messages = maybe_rewrite(name, messages)
    with GPU_LOCK:
        if _cur["name"] != base:
            apply_adapter(base)
        set_adapter_scale(model, scale)  # variant-specific LoRA strength (alpha)
        ids = sp.EncodeAsIds(render(messages, think, tools))
        x = torch.tensor([ids], device=dev)
        out = []
        prev = ""
        with torch.no_grad():
            for _ in range(MAXNEW):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    lg = model(input_ids=x)["logits"][0, -1].float()
                for t in set(out):
                    lg[t] = lg[t] / REP if lg[t] > 0 else lg[t] * REP
                nid = int(torch.argmax(lg))
                if nid == END_ID:
                    break
                out.append(nid)
                x = torch.cat([x, torch.tensor([[nid]], device=dev)], 1)
                full = sp.DecodeIds(out)
                if END in full:
                    full = full.split(END)[0]
                delta = full[len(prev) :]
                prev = full
                if delta:
                    yield delta
                # --- tool execution: run a just-completed <tool:...></tool> and inject the REAL result ---
                if tools and full.rstrip().endswith("</tool>"):
                    m = TOOL_RE.findall(full)
                    if m:
                        result = run_tool(m[-1])
                        inj = f"\n<result>\n{result}\n</result>\n"
                        yield inj
                        inj_ids = sp.EncodeAsIds(inj)
                        x = torch.cat([x, torch.tensor([inj_ids], device=dev)], 1)
                        out.extend(inj_ids)
                        prev = sp.DecodeIds(out)
                        if END in prev:
                            prev = prev.split(END)[0]
                if END in sp.DecodeIds(out[-4:]):
                    break


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path.startswith("/api/tags"):
            self._json(
                {
                    "models": [
                        {
                            "name": n,
                            "model": n,
                            "modified_at": "2024-01-01T00:00:00Z",
                            "size": 0,
                            "digest": "helix-" + n,
                        }
                        for n in VARIANTS
                    ]
                }
            )
        elif self.path.startswith("/api/version"):
            self._json({"version": "helix-0.9b-facade"})
        else:
            self._json({"status": "ok", "models": VARIANTS})

    def do_POST(self):
        ln = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(ln) if ln else b"{}"
        try:
            req = json.loads(raw or b"{}")
        except Exception:
            req = {}
        if not self.path.startswith("/v1/chat/completions"):
            return self._json({"error": "not found"}, 404)
        model_name = req.get("model", "helix-corrective")
        messages = req.get("messages", [])
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        cid = "chatcmpl-helix"

        def send(d):
            self.wfile.write(("data: " + json.dumps(d) + "\n\n").encode())
            self.wfile.flush()

        try:
            send(
                {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "model": model_name,
                    "choices": [
                        {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                    ],
                }
            )
            for delta in gen_stream(model_name, messages):
                send(
                    {
                        "id": cid,
                        "object": "chat.completion.chunk",
                        "model": model_name,
                        "choices": [
                            {"index": 0, "delta": {"content": delta}, "finish_reason": None}
                        ],
                    }
                )
            send(
                {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
            )
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except Exception as e:
            try:
                send(
                    {
                        "id": cid,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": f"[helix error: {e}]"},
                                "finish_reason": "stop",
                            }
                        ],
                    }
                )
                self.wfile.write(b"data: [DONE]\n\n")
            except Exception:
                pass


print(f"[helix-srv] ready on :{PORT}  variants={VARIANTS}", flush=True)
ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
