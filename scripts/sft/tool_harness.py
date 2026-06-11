#!/usr/bin/env python3
"""Tool-Use MATH harness (MVP).

Flow:  model emits <tool:python>...</tool>  ->  generation STOPS at </tool>  ->
a SAFE arithmetic evaluator computes the result  ->  harness injects
<result>...</result>  ->  model RESUMES to the final natural-language answer.

Security (math MVP): we do NOT execute arbitrary Python. The code is parsed with
`ast` and evaluated against a WHITELIST (numbers, + - * / // % **, parens, a few
math functions). Anything else (import, attribute access, assignment, unknown
names) is rejected -> safe *by construction*, no RCE surface, no subprocess.
The Docker/nsjail sandbox is only needed for the later GENERAL code runner
(see docs/BLUEPRINT_TOOL_USE_VERIFIER.md, Stufe 4).

Note: step_2100 is NOT yet tool-trained, so the --demo uses few-shot priming to
show the harness mechanics end-to-end. Real behaviour comes after tool-SFT.
"""
import os, sys, re, ast, math, argparse, pathlib

REPO = pathlib.Path("/workspace/v2data"); sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("AURALIS_USE_MAMBA_KERNEL", "1")

TOOL_OPEN = "<tool:python>"; TOOL_CLOSE = "</tool>"
RES_OPEN = "<result>"; RES_CLOSE = "</result>"; END = "<|end|>"
SYS = "Du bist Auralis, ein hilfreicher, ehrlicher KI-Assistent."

# ----------------------------- safe evaluator -----------------------------
_BIN = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b, ast.FloorDiv: lambda a, b: a // b, ast.Mod: lambda a, b: a % b,
        ast.Pow: lambda a, b: a ** b}
_UN = {ast.USub: lambda a: -a, ast.UAdd: lambda a: +a}
_FUNC = {"sqrt": math.sqrt, "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
         "pow": pow, "factorial": math.factorial, "gcd": math.gcd, "log": math.log,
         "log2": math.log2, "log10": math.log10, "exp": math.exp, "sin": math.sin,
         "cos": math.cos, "tan": math.tan, "floor": math.floor, "ceil": math.ceil}
_CONST = {"pi": math.pi, "e": math.e, "tau": math.tau}


class CalcError(Exception):
    pass


def _ev(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise CalcError("nur Zahlen erlaubt")
    if isinstance(node, ast.BinOp):
        op = type(node.op)
        if op not in _BIN:
            raise CalcError("Operator nicht erlaubt")
        a = _ev(node.left); b = _ev(node.right)
        if op is ast.Pow and isinstance(b, (int, float)) and abs(b) > 1000:
            raise CalcError("Exponent zu gross")
        return _BIN[op](a, b)
    if isinstance(node, ast.UnaryOp):
        op = type(node.op)
        if op not in _UN:
            raise CalcError("Operator nicht erlaubt")
        return _UN[op](_ev(node.operand))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNC:
            raise CalcError("Funktion nicht erlaubt")
        if node.func.id == "factorial":
            a = _ev(node.args[0])
            if a > 1000:
                raise CalcError("factorial-Argument zu gross")
        return _FUNC[node.func.id](*[_ev(a) for a in node.args])
    if isinstance(node, ast.Name):
        if node.id in _CONST:
            return _CONST[node.id]
        raise CalcError(f"Name '{node.id}' nicht erlaubt")
    if isinstance(node, (ast.Tuple, ast.List)):
        return [_ev(e) for e in node.elts]
    raise CalcError("Ausdruck nicht erlaubt")


def _fmt(v):
    if isinstance(v, float):
        return str(int(v)) if v.is_integer() else ("%.10g" % v)
    return str(v)


def safe_calc(code):
    """(ok, output). Accepts `print(expr)` and bare `expr` lines, arithmetic only."""
    code = (code or "").strip()
    code = re.sub(r"^```(?:python)?", "", code).strip()
    code = re.sub(r"```$", "", code).strip()
    if not code:
        return False, "leerer Code"
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as e:
        return False, f"Syntaxfehler: {e.msg}"
    out = []
    try:
        for stmt in tree.body:
            if not isinstance(stmt, ast.Expr):
                return False, "nur Arithmetik/print erlaubt (keine Zuweisungen/Imports)"
            node = stmt.value
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
                out.append(" ".join(_fmt(_ev(a)) for a in node.args))
            else:
                out.append(_fmt(_ev(node)))
    except CalcError as e:
        return False, f"abgelehnt: {e}"
    except ZeroDivisionError:
        return False, "Division durch Null"
    except Exception as e:
        return False, f"Fehler: {type(e).__name__}"
    return True, "\n".join(out)


# ----------------------------- model + loop -----------------------------
def load(ckpt, cfg, tok, device):
    import torch
    import sentencepiece as spm
    from auralis.model import build_model
    sp = spm.SentencePieceProcessor(model_file=tok)
    model = build_model(cfg).to(device).eval()
    payload = torch.load(ckpt, map_location=device, weights_only=False)
    state = payload.get("model", payload.get("state_dict", payload))
    miss, extra = model.load_state_dict(state, strict=False)
    print(f"loaded {ckpt} | missing={len(miss)} extra={len(extra)}", flush=True)
    return model, sp


def gen_until(model, sp, text, stops, device, max_new=128, rep_pen=1.3):
    """Greedy generate from `text`; stop when any string in `stops` appears or <|end|>.
    Returns (new_text_including_stop, stop_hit_or_None)."""
    import torch
    end_id = sp.EncodeAsIds(END)[-1]
    ids = sp.EncodeAsIds(text)
    inp = torch.tensor([ids], device=device); gen = []
    for _ in range(max_new):
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(input_ids=inp)["logits"][0, -1].float()
        for t in set(gen):
            logits[t] = logits[t] / rep_pen if logits[t] > 0 else logits[t] * rep_pen
        nxt = int(torch.argmax(logits).item())
        if nxt == end_id:
            return sp.DecodeIds(gen), END
        gen.append(nxt)
        inp = torch.cat([inp, torch.tensor([[nxt]], device=device)], 1)
        dec = sp.DecodeIds(gen)
        for s in stops:
            k = dec.find(s)
            if k != -1:
                return dec[:k + len(s)], s
    return sp.DecodeIds(gen), None


def extract_code(text):
    i = text.rfind(TOOL_OPEN)
    j = text.find(TOOL_CLOSE, i + len(TOOL_OPEN)) if i != -1 else -1
    if i == -1 or j == -1:
        return None
    return text[i + len(TOOL_OPEN):j]


def run_with_tools(model, sp, prompt, device, max_tool_calls=3, verbose=True, genfn=None):
    genfn = genfn or gen_until
    full = prompt; assistant = ""
    for _ in range(max_tool_calls + 1):
        chunk, stop = genfn(model, sp, full, [TOOL_CLOSE], device)
        full += chunk; assistant += chunk
        if stop != TOOL_CLOSE:
            break  # <|end|> or max_new -> done
        code = extract_code(assistant)
        ok, res = safe_calc(code) if code is not None else (False, "kein Tool-Call erkannt")
        block = f"\n{RES_OPEN}\n{res}\n{RES_CLOSE}\n"
        full += block; assistant += block
        if verbose:
            print(f"  [tool] code={code!r} -> ok={ok} result={res!r}")
    return assistant


# ----------------------------- selftest -----------------------------
def selftest():
    cases = [
        ("print(12 + 15)", True, "27"),
        ("print(47 * 83)", True, "3901"),
        ("print(3 * 60 + 25)", True, "205"),
        ("10.08 / 4.2", True, "2.4"),
        ("print(2 ** 10)", True, "1024"),
        ("print(sqrt(144))", True, "12"),
        ("factorial(5)", True, "120"),
        ("```python\nprint(100 - 37)\n```", True, "63"),
        ("import os", False, None),
        ("__import__('os').system('ls')", False, None),
        ("open('/etc/passwd')", False, None),
        ("9**9**9", False, None),   # bomb guard
        ("x = 5", False, None),
        ("1/0", False, None),
    ]
    ok_all = True
    for code, exp_ok, exp_out in cases:
        ok, out = safe_calc(code)
        good = (ok == exp_ok) and (exp_out is None or out == exp_out)
        ok_all = ok_all and good
        print(f"  [{'OK ' if good else 'FAIL'}] {code!r:45} -> ok={ok} out={out!r}")
    print("=== SELFTEST", "PASS ===" if ok_all else "FAIL ===")
    return ok_all


def plumbtest():
    """Prove the harness loop end-to-end WITHOUT the model: a scripted 'model' emits
    a real tool-call, then the final answer. Verifies stop@</tool> -> execute ->
    inject <result> -> resume."""
    script = [
        ("Ich rechne das aus.\n" + TOOL_OPEN + "\nprint(12 + 15)\n" + TOOL_CLOSE, TOOL_CLOSE),
        ("12 + 15 ergibt 27.", END),
    ]
    state = {"i": 0}

    def fake_gen(model, sp, text, stops, device, **kw):
        r = script[state["i"]]; state["i"] += 1; return r

    out = run_with_tools(None, None, "<|assistant|>\n", None, genfn=fake_gen, verbose=True)
    inj = f"{RES_OPEN}\n27\n{RES_CLOSE}" in out          # harness injected the real result
    used = "ergibt 27" in out                             # model resumed using the result
    no_fake = out.count(RES_OPEN) == 1                    # model did NOT write its own <result>
    ok = inj and used and no_fake
    print("PLUMBTEST transcript:\n" + out)
    print(f"  injected-result={inj} resumed-using-it={used} single-result-block={no_fake}")
    print("=== PLUMBTEST", "PASS ===" if ok else "FAIL ===")
    return ok


FEWSHOT = [
    {"role": "user", "content": "Was ist 47 mal 83?"},
    {"role": "assistant", "content": f"{TOOL_OPEN}\nprint(47 * 83)\n{TOOL_CLOSE}\n{RES_OPEN}\n3901\n{RES_CLOSE}\n47 mal 83 ergibt 3901."},
    {"role": "user", "content": "Wie viele Minuten sind 3 Stunden und 25 Minuten?"},
    {"role": "assistant", "content": f"{TOOL_OPEN}\nprint(3 * 60 + 25)\n{TOOL_CLOSE}\n{RES_OPEN}\n205\n{RES_CLOSE}\n3 Stunden und 25 Minuten sind 205 Minuten."},
]
DEMO_Q = ["Was ist 12 plus 15?", "Was ist 47 mal 83?", "Wie viel sind 144 geteilt durch 12?",
          "Was ist 15% von 240?", "Wie viele Sekunden hat ein Tag?"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(REPO / "checkpoints/sft_v2/sft_smoke_step_2100.pt"))
    ap.add_argument("--model-config", default=str(REPO / "configs/model/helix_v2_1b.yaml"))
    ap.add_argument("--tokenizer", default=str(REPO / "tokenizer/helix_v2_tokenizer.model"))
    ap.add_argument("--selftest-only", action="store_true")
    a = ap.parse_args()
    print("=== CALCULATOR SELFTEST (no GPU) ===")
    passed = selftest()
    print("\n=== HARNESS PLUMBING TEST (no GPU, scripted model) ===")
    passed = selftest_plumb = plumbtest() and passed
    if a.selftest_only:
        sys.exit(0 if passed else 1)
    import torch
    from auralis.tokenizer.chat_template import build_inference_prompt
    device = torch.device("cuda")
    print("\n=== loading model (for few-shot harness demo) ===")
    model, sp = load(a.checkpoint, a.model_config, a.tokenizer, device)
    print("\n=== END-TO-END HARNESS DEMO (few-shot primed; step_2100 not tool-trained) ===")
    for q in DEMO_Q:
        msgs = FEWSHOT + [{"role": "user", "content": q}]
        prompt = build_inference_prompt(msgs, default_system=SYS)
        print("=" * 66); print("Q:", q)
        ans = run_with_tools(model, sp, prompt, device)
        print("ANSWER:", ans.strip()[:400])


if __name__ == "__main__":
    main()
