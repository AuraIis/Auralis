#!/usr/bin/env python3
"""Verified math word-problem generation — LOCAL teacher + EXECUTOR ground-truth.

A local Ollama teacher (default qwen3.6:27b) proposes German math word problems,
each with a calculator-evaluable expression and its own claimed result. The SAFE
arithmetic calculator (tool_harness.safe_calc) is the ground truth. Two tiers:

  T1  self-consistency : calc(ausdruck) == teacher "ergebnis"   (catches arithmetic slips)
  T2  cross-solve      : a SECOND model solves the WORD PROBLEM independently and
                          must arrive at calc(ausdruck)          (catches modelling errors)

Only traces passing the requested tier are emitted, in the EXACT gen_tool_traces
format (`<tool:python>print(expr)</tool><result>res</result> answer`), drop-in for
the SFT trainer (which must loss-mask the <result> block). Key-free, fully local.

TWO PHASES (so the 24GB 3090 never swaps qwen3.6<->gemma4 per call):
  gen     : qwen3.6 proposes -> append to raw_proposals.jsonl   (qwen resident)
  verify  : calc + gemma4 cross-solve -> append to verified_math.jsonl (gemma resident)
Both phases APPEND incrementally and are RESUMABLE (a crash loses nothing; rerun skips
already-done questions). `--phase all` runs gen then verify (one model swap total).
"""
import os, sys, re, json, argparse, pathlib, urllib.request

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from tool_harness import safe_calc, TOOL_OPEN, TOOL_CLOSE, RES_OPEN, RES_CLOSE  # noqa

SYS = "Du bist Auralis, ein hilfreicher, ehrlicher KI-Assistent."
OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434") + "/api/generate"


def helix(q, a):
    return f"<|system|>\n{SYS}\n<|end|>\n<|user|>\n{q}\n<|end|>\n<|assistant|>\n{a}\n<|end|>\n"


def full_trace(q, expr, res, answer_text):
    a = (f"{TOOL_OPEN}\nprint({expr})\n{TOOL_CLOSE}\n{RES_OPEN}\n{res}\n{RES_CLOSE}\n" + answer_text)
    return helix(q, a)


def ollama(model, prompt, n_predict=2048, temp=0.7, timeout=180):
    body = json.dumps({
        "model": model, "prompt": prompt, "stream": False, "think": False,
        "keep_alive": "30m",  # keep model resident across calls within a phase
        "options": {"temperature": temp, "num_predict": n_predict},
    }).encode("utf-8")
    req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8")).get("response", "")


def norm_expr(e):
    e = (e or "").strip()
    e = e.replace("×", "*").replace("·", "*").replace("÷", "/").replace("^", "**")
    e = e.replace("€", "").replace("$", "").replace("%", "")
    e = re.sub(r"(?<=\d)\s*,\s*(?=\d)", ".", e)
    e = re.sub(r"\s+", "", e)
    return e


def to_num(x):
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    m = re.search(r"-?\d+(?:[.,]\d+)?", str(x))
    return float(m.group(0).replace(",", ".")) if m else None


def approx(a, b, tol=1e-6):
    return a is not None and b is not None and abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def parse_jsonl(text):
    rows = []
    for line in text.splitlines():
        line = line.strip().strip("`")
        if not line.startswith("{"):
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def read_jsonl(path):
    if not path.exists():
        return []
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


GEN_PROMPT = """Erzeuge {k} deutsche MATHE-TEXTAUFGABEN (Grundrechenarten, Prozent, Einheiten, einfache Textaufgaben).
Fuer jede Aufgabe ein JSON-Objekt mit GENAU diesen Feldern:
  "aufgabe":      die Textaufgabe auf Deutsch, als eine Frage
  "ausdruck":     der Rechenausdruck, der die Aufgabe loest, in PYTHON-SYNTAX.
                  NUR Zahlen und + - * / ( ) ** . Dezimaltrennzeichen ist der PUNKT.
                  KEINE Einheiten, KEIN Prozentzeichen (statt "20%" schreibe die /100-Form, z.B. 80*20/100),
                  KEINE Tausender-Trennzeichen, KEINE Variablen.
  "ergebnis":     das numerische Ergebnis von "ausdruck" (nur die Zahl)
  "antwort_text": ein vollstaendiger, natuerlicher deutscher Antwortsatz, der das Ergebnis nennt

Beispiel:
{{"aufgabe":"Ein Laden hat 12 Kisten mit je 4 Aepfeln und zusaetzlich 8 lose Aepfel. Wie viele Aepfel sind das insgesamt?","ausdruck":"12*4+8","ergebnis":56,"antwort_text":"Insgesamt sind es 56 Aepfel."}}

Gib AUSSCHLIESSLICH {k} verschiedene JSON-Zeilen aus (JSONL), nichts sonst, keine Nummerierung, keine Codeblock-Markierung."""

CROSS_PROMPT = """Loese diese Mathe-Textaufgabe Schritt fuer Schritt im Kopf und gib am Ende NUR die finale Zahl aus (ohne Einheit, ohne Text danach).

Aufgabe: {q}

Antworte mit der finalen Zahl in der letzten Zeile."""

TOPICS = [
    "Grundrechenarten mit mehreren Schritten (Addition, Subtraktion, Multiplikation, Division kombiniert)",
    "Prozentrechnung (Rabatt, Aufschlag, Anteil, einfache Zinsen)",
    "Einheiten umrechnen (Zeit: Stunden/Minuten/Sekunden; Laenge: km/m/cm; Gewicht: kg/g)",
    "Geschwindigkeit, Distanz und Zeit",
    "Durchschnitt und einfache Verhaeltnisse/Anteile",
    "einfache Geometrie (Flaeche und Umfang von Rechteck/Quadrat, Volumen Quader)",
    "Mehrschritt-Sachaufgaben aus dem Alltag (Einkauf, Rezepte skalieren, Wechselgeld)",
    "Brueche und Teile eines Ganzen (die Haelfte, ein Drittel, drei Viertel von ...)",
]


# ----------------------------- phase: generate -----------------------------
def gen_phase(a, raw_path):
    existing = read_jsonl(raw_path)
    seen = {r.get("aufgabe", "").strip() for r in existing}
    target_raw = a.gen_target
    print(f"[gen] resume: {len(existing)} raw on disk, target {target_raw}", flush=True)
    fout = open(raw_path, "a", encoding="utf-8")
    call = 0
    while len(seen) < target_raw and call < a.max_calls:
        topic = TOPICS[call % len(TOPICS)]
        prompt = GEN_PROMPT.format(k=a.per_call) + \
            f"\nSchwerpunkt dieser Charge: {topic}. Verwende andere Zahlen als in fruehzeitigen Chargen."
        try:
            text = ollama(a.teacher, prompt, temp=a.temp)
        except Exception as e:
            print(f"[gen call {call}] teacher error: {e}", file=sys.stderr, flush=True)
            call += 1
            continue
        added = 0
        for r in parse_jsonl(text):
            if not isinstance(r, dict):
                continue
            q = (r.get("aufgabe") or "").strip()
            if not q or q in seen or "ausdruck" not in r or "ergebnis" not in r:
                continue
            seen.add(q)
            r["_topic"] = topic
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            fout.flush()
            added += 1
        call += 1
        print(f"[gen call {call}] +{added} -> raw total {len(seen)}", flush=True)
    fout.close()
    print(f"[gen] done: {len(seen)} raw proposals", flush=True)
    return len(seen)


# ----------------------------- phase: verify -----------------------------
def verify_phase(a, raw_path, ver_path):
    raw = read_jsonl(raw_path)
    done = {r.get("meta", {}).get("q", "") for r in read_jsonl(ver_path)}
    print(f"[verify] {len(raw)} raw, {len(done)} already verified", flush=True)
    fout = open(ver_path, "a", encoding="utf-8")
    st = dict(seen=0, calc_ok=0, consistent=0, cross_calls=0, cross_ok=0, kept=0)
    for r in raw:
        q = (r.get("aufgabe") or "").strip()
        if not q or q in done:
            continue
        done.add(q)
        st["seen"] += 1
        expr = norm_expr(str(r.get("ausdruck")))
        ok, res = safe_calc(f"print({expr})")
        if not ok:
            continue
        st["calc_ok"] += 1
        calc_n, teach_n = to_num(res), to_num(r.get("ergebnis"))
        if not approx(calc_n, teach_n):            # T1
            continue
        st["consistent"] += 1
        res_str = res.strip()
        ans = (r.get("antwort_text") or "").strip()
        if res_str not in ans:
            ans = f"Das Ergebnis ist {res_str}."
        if a.tier == "2":                          # T2
            st["cross_calls"] += 1
            try:
                cross = ollama(a.cross_model, CROSS_PROMPT.format(q=q), n_predict=512, temp=0.2)
            except Exception as e:
                print(f"  cross error: {e}", file=sys.stderr, flush=True)
                continue
            nums = re.findall(r"-?\d+(?:[.,]\d+)?", cross)
            if not approx(to_num(nums[-1]) if nums else None, calc_n):
                continue
            st["cross_ok"] += 1
        fout.write(json.dumps({
            "text": full_trace(q, expr, res_str, ans), "source": "tool_math_verified",
            "has_tool": True, "meta": {"q": q, "expr": expr, "res": res_str, "teacher": a.teacher},
        }, ensure_ascii=False) + "\n")
        fout.flush()
        st["kept"] += 1
        if st["kept"] % 20 == 0:
            print(f"  [verify] kept {st['kept']} (seen {st['seen']})", flush=True)
    fout.close()
    total = len(read_jsonl(ver_path))
    print("\n=== VERIFY DONE | tier=%s ===" % a.tier)
    for k in ["seen", "calc_ok", "consistent", "cross_calls", "cross_ok", "kept"]:
        print(f"  {k:12} {st[k]}")
    if st["calc_ok"]:
        print(f"  T1 self-consistency = {st['consistent']/st['calc_ok']:.0%}")
    if st["cross_calls"]:
        print(f"  T2 cross-solve agree = {st['cross_ok']/st['cross_calls']:.0%}")
    print(f"  -> {ver_path}  (TOTAL on disk: {total})")
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(HERE.parent.parent / "data/training/tool_verified_v1"))
    ap.add_argument("--teacher", default="qwen3.6:27b")
    ap.add_argument("--cross-model", default="gemma4:12b")
    ap.add_argument("--target", type=int, default=400, help="verified traces wanted (sets gen-target)")
    ap.add_argument("--gen-target", type=int, default=0, help="raw proposals to generate (0=auto from target)")
    ap.add_argument("--per-call", type=int, default=10)
    ap.add_argument("--max-calls", type=int, default=80)
    ap.add_argument("--tier", choices=["1", "2"], default="2")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--phase", choices=["gen", "verify", "all"], default="all")
    a = ap.parse_args()
    if not a.gen_target:
        a.gen_target = int(a.target / 0.93) + 10   # over-generate to absorb ~7% drop
    out = pathlib.Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    raw_path = out / "raw_proposals.jsonl"
    ver_path = out / "verified_math.jsonl"
    print(f"=== verified-math | teacher={a.teacher} cross={a.cross_model} phase={a.phase} "
          f"target={a.target} gen_target={a.gen_target} ===", flush=True)
    if a.phase in ("gen", "all"):
        gen_phase(a, raw_path)
    if a.phase in ("verify", "all"):
        verify_phase(a, raw_path, ver_path)


if __name__ == "__main__":
    main()
