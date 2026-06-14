#!/usr/bin/env python3
"""Executor code gate: generate code for held-out tasks, EXECUTE it against asserts.
Reports syntax_rate, pass_rate (all tests green), eos_rate, and pass on UNSEEN funcs
(funcs not in the training set -> honest generalization signal)."""
import os, sys, json, re, signal
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ["AURALIS_USE_CUDA_KERNELS"] = "1"
REPO = "/workspace/v2data"; sys.path.insert(0, REPO); sys.path.insert(0, REPO + "/src")
import torch, sentencepiece as spm
from auralis.model import build_model
from auralis.adapters.lora import inject_adapters, freeze_base, load_adapter_state_dict, set_adapter_scale
CFG = REPO + "/configs/model/helix_v2_1b_flash.yaml"; CKPT = REPO + "/checkpoints/corpus20b_codeheavy/step_60000.pt"
ADP = os.environ.get("ADAPTER", REPO + "/checkpoints/sft_code_v3/adapter_best.pt"); TOK = REPO + "/tokenizer/helix_v2_tokenizer.model"
SYS = "Du bist Auralis, ein hilfreicher, ehrlicher KI-Assistent."; ASST = "<|assistant|>\n"; END = "<|end|>"
sp = spm.SentencePieceProcessor(model_file=TOK); dev = torch.device("cuda"); END_ID = sp.EncodeAsIds(END)[-1]
model = build_model(CFG); pl = torch.load(CKPT, map_location="cpu", weights_only=False)
model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in pl["model"].items()}, strict=False)
model = model.to(dev); inject_adapters(model, r=64, alpha=128, kind="lora"); freeze_base(model)
ck = torch.load(ADP, map_location="cpu"); load_adapter_state_dict(model, ck["adapter"])
emb = getattr(model, "embedding", None) or getattr(model, "embed_tokens", None)
for i, tid in enumerate(ck["emb_ids"]): emb.weight.data[tid] = ck["emb_rows"][i].to(emb.weight.device, emb.weight.dtype)
model = model.to(dev).eval(); set_adapter_scale(model, 1.0); print("adapter:", ADP, flush=True)

def chat(p, max_new=256, rep=1.15):
    ids = sp.EncodeAsIds(f"<|system|>\n{SYS}\n{END}\n<|user|>\n{p}\n{END}\n{ASST}"); x = torch.tensor([ids], device=dev); out = []; st = False
    with torch.no_grad():
        for _ in range(max_new):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16): lg = model(input_ids=x)["logits"][0, -1].float()
            for t in set(out): lg[t] = lg[t]/rep if lg[t] > 0 else lg[t]*rep
            nid = int(torch.argmax(lg))
            if nid == END_ID: st = True; break
            out.append(nid); x = torch.cat([x, torch.tensor([[nid]], device=dev)], 1)
            if END in sp.DecodeIds(out[-4:]): st = True; break
    return sp.DecodeIds(out).split(END)[0].strip(), st

def extract(t):
    m = re.search(r"```(?:python)?\s*(.*?)```", t, re.S)
    return (m.group(1) if m else t).strip()
class TO(Exception): pass
def _alarm(s, f): raise TO()
def run_task(src, func, tests):
    try: compiled = compile(src, "<gen>", "exec")
    except Exception as e: return "SYNTAX_ERR", str(e)[:55]
    ns = {}
    try:
        signal.signal(signal.SIGALRM, _alarm); signal.alarm(5)
        exec(compiled, ns)
        if func not in ns: signal.alarm(0); return "NO_FUNC", func
        for args, exp in tests:
            got = ns[func](*args)
            if got != exp: signal.alarm(0); return "WRONG", f"{func}{args}={got!r}!={exp!r}"
        signal.alarm(0); return "PASS", ""
    except TO: signal.alarm(0); return "TIMEOUT", ""
    except Exception as e:
        try: signal.alarm(0)
        except Exception: pass
        return "RUNTIME_ERR", str(e)[:55]

TASKS = [
 ("Schreibe eine Funktion ist_gerade(n), die True zurueckgibt, wenn die ganze Zahl n gerade ist, sonst False.", "ist_gerade", [((4,),True),((7,),False),((0,),True),((-2,),True)]),
 ("Schreibe eine Funktion summe_liste(xs), die die Summe aller Zahlen in der Liste xs zurueckgibt.", "summe_liste", [(([1,2,3],),6),(([],),0),(([5],),5)]),
 ("Schreibe eine Funktion max_wert(xs), die den groessten Wert der nichtleeren Liste xs zurueckgibt.", "max_wert", [(([3,1,2],),3),(([-1,-5],),-1)]),
 ("Schreibe eine Funktion ist_palindrom(s), die True zurueckgibt, wenn der String s vorwaerts und rueckwaerts gleich ist.", "ist_palindrom", [(("otto",),True),(("haus",),False),(("",),True)]),
 ("Schreibe eine Funktion zaehle_vokale(s), die die Anzahl der Vokale (a,e,i,o,u) im String s zurueckgibt.", "zaehle_vokale", [(("hallo",),2),(("xyz",),0),(("aeiou",),5)]),
 ("Schreibe eine Funktion fakultaet(n), die die Fakultaet von n berechnet (n!).", "fakultaet", [((5,),120),((0,),1),((1,),1)]),
 ("Schreibe eine Funktion umkehren(s), die den String s umgekehrt zurueckgibt.", "umkehren", [(("abc",),"cba"),(("",),"")]),
 ("Schreibe eine Funktion fibonacci(n), die die n-te Fibonacci-Zahl zurueckgibt (fibonacci(0)=0, fibonacci(1)=1).", "fibonacci", [((0,),0),((1,),1),((7,),13),((10,),55)]),
 ("Schreibe eine Funktion ist_primzahl(n), die True zurueckgibt, wenn n eine Primzahl ist.", "ist_primzahl", [((7,),True),((8,),False),((1,),False),((2,),True)]),
 ("Schreibe eine Funktion doppelt(xs), die eine neue Liste zurueckgibt, in der jeder Wert aus xs verdoppelt ist.", "doppelt", [(([1,2,3],),[2,4,6]),(([],),[])]),
 ("Schreibe eine Funktion ggt(a, b), die den groessten gemeinsamen Teiler von a und b zurueckgibt.", "ggt", [((12,8),4),((17,5),1),((100,10),10)]),
 ("Write a function count_words(s) that returns the number of words in the string s (words separated by spaces).", "count_words", [(("hello world",),2),(("",),0),(("a b c",),3)]),
 ("Schreibe eine Funktion celsius_zu_fahrenheit(c), die Grad Celsius in Fahrenheit umrechnet (Formel c*9/5+32).", "celsius_zu_fahrenheit", [((0,),32),((100,),212),((20,),68)]),
 ("Write a function remove_duplicates(xs) that returns a list with duplicates removed, preserving order.", "remove_duplicates", [(([1,1,2,3,3],),[1,2,3]),(([],),[]),(([5,5,5],),[5])]),
 ("Schreibe eine Funktion ist_aufsteigend(xs), die True zurueckgibt, wenn die Liste xs aufsteigend sortiert ist.", "ist_aufsteigend", [(([1,2,3],),True),(([3,1,2],),False),(([1],),True)]),
 ("Schreibe eine Funktion quersumme(n), die die Quersumme der nichtnegativen ganzen Zahl n berechnet.", "quersumme", [((123,),6),((0,),0),((99,),18)]),
 ("Schreibe eine Funktion nur_gerade(xs), die eine Liste nur mit den geraden Zahlen aus xs zurueckgibt.", "nur_gerade", [(([1,2,3,4],),[2,4]),(([1,3],),[]),(([2,4,6],),[2,4,6])]),
 ("Schreibe eine Funktion wort_laengen(woerter), die eine Liste mit den Laengen der Woerter aus der Liste woerter zurueckgibt.", "wort_laengen", [((["a","bb","ccc"],),[1,2,3]),(([],),[])]),
 # ---- v4 add-on: 6 more UNSEEN tasks (new names, logic overlaps trained classes -> transfer test) ----
 ("Schreibe eine Funktion summe_quadrate(xs), die die Summe der Quadrate der Zahlen in xs zurueckgibt.", "summe_quadrate", [(([1,2,3],),14),(([],),0),(([4],),16)]),
 ("Schreibe eine Funktion max_minus_min(xs), die die Differenz zwischen groesstem und kleinstem Wert der nichtleeren Liste xs zurueckgibt.", "max_minus_min", [(([3,7,1],),6),(([5],),0),(([-2,2],),4)]),
 ("Schreibe eine Funktion dritte_potenz(xs), die eine neue Liste mit der dritten Potenz jeder Zahl aus xs zurueckgibt.", "dritte_potenz", [(([1,2,3],),[1,8,27]),(([],),[]),(([-2],),[-8])]),
 ("Schreibe eine Funktion enthaelt_negative(xs), die True zurueckgibt, wenn xs mindestens eine negative Zahl enthaelt.", "enthaelt_negative", [(([1,-2,3],),True),(([1,2],),False),(([],),False)]),
 ("Schreibe eine Funktion produkt_liste(xs), die das Produkt aller Zahlen in xs zurueckgibt (leere Liste ergibt 1).", "produkt_liste", [(([1,2,3,4],),24),(([],),1),(([5],),5)]),
 ("Schreibe eine Funktion gerade_filtern(xs), die eine Liste nur mit den geraden Zahlen aus xs zurueckgibt.", "gerade_filtern", [(([1,2,3,4],),[2,4]),(([1,3],),[]),(([2,4],),[2,4])]),
]
TRAINF = set(json.load(open(REPO + "/diag/code_train_funcs.json", encoding="utf-8"))) if os.path.exists(REPO + "/diag/code_train_funcs.json") else set()
syn = pas = eos = n = 0; up = un = 0
for prompt, func, tests in TASKS:
    a, st = chat(prompt); eos += st; n += 1
    src = extract(a)
    try: compile(src, "<g>", "exec"); syntax_ok = True
    except Exception: syntax_ok = False
    syn += syntax_ok
    status, detail = run_task(src, func, tests); ok = (status == "PASS"); pas += ok
    seen = func in TRAINF
    if not seen: un += 1; up += ok
    print(f"[{'seen ' if seen else 'unseen'}] {func:24s} -> {status:11s} {('' if ok else detail)[:50]}", flush=True)
M = {"n": n, "syntax_rate": round(syn/n, 2), "pass_rate": f"{pas}/{n}", "eos_rate": round(eos/n, 2), "unseen_pass": f"{up}/{un}"}
print("\n=== CODE GATE ===")
for k, v in M.items(): print(f"  {k:14s}: {v}")
print(f"\n>>> PRIMARY METRIC: unseen_pass = {M['unseen_pass']}  (seen_pass is only diagnostic / memorization) <<<")
json.dump(M, open(REPO + "/diag/code_gate.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("wrote diag/code_gate.json", flush=True)
