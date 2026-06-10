"""Held-out eval gate for the corpus20b code-heavy continued pretrain.

WHY THIS EXISTS: the trainer's val split (tail of en.bin / de_curated etc.)
was TRAINING data of the foundation run — its val_loss is leaked and cannot be
trusted as a quality gate. This gate runs on a CLEAN, hand-authored, fixed
probe set (``data/eval/heldout_gate_v1.jsonl``) that was never part of any
training bin (clean by construction, not carved from the .bins).

Four axes — the four things this project actually cares about:

- **qa**       German factual knowledge: keyword match on the answer.
- **abstain**  questions about invented entities — the model SHOULD hedge
               ("weiß ich nicht") and must not fabricate facts.
- **code**     Python prompts → does the extracted code ``ast.parse``, define
               the required function, pass a one-line check (fresh subprocess).
- **tool**     does the model route to a ``<tool:python>`` call whose code
               evaluates (safe arithmetic, no exec) to the expected result.

Composite = unweighted mean of the four axis means. Fully deterministic:
greedy decoding, fixed item order, no sampling.

Schema (one JSON object per line):

    id              str   unique
    axis            str   qa | abstain | code | tool
    prompt          str   user turn (German)
    max_new_tokens  int?  per-item override
    expect_any      list? (qa) any normalized keyword scores 1
    forbid_any      list? (qa/abstain) any hit scores 0
    must_define     str?  (code) function name the code must define
    check           str?  (code) truthy expression executed against the code
    expect_result   str?  (tool) expected output of the emitted tool code

Usage (kernel backend on — native GLA had the decay-dim bug)::

    AURALIS_USE_MAMBA_KERNEL=1 AURALIS_USE_GLA_KERNEL=1 python scripts/eval/eval_gate.py \\
        --model-config configs/model/helix_v2_1b_flash.yaml \\
        --checkpoint checkpoints/.../step_50000.pt \\
        --out eval/results/gate/step_50000.json

The trainer (``PretrainTrainer.run_eval_gate``) imports :func:`run_gate`
directly and runs it in-process every ``evaluation.gate.every_steps`` steps.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import operator
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

AXES = ("qa", "abstain", "code", "tool")
TOOL_OPEN = "<tool:python>"
TOOL_CLOSE = "</tool>"
END = "<|end|>"

SYSTEM_QA = (
    "Du bist Auralis, ein hilfreicher deutscher KI-Assistent. "
    "Antworte korrekt, knapp und ehrlich. Wenn du etwas nicht weißt, sage das deutlich."
)
SYSTEM_CODE = (
    "Du bist Auralis, ein deutscher Programmier-Assistent. "
    "Antworte nur mit Python-Code, ohne Erklärung."
)
# One-shot primer: the base model is not yet tool-trained; this measures the
# ROUTE (emit a correct call), and the exemplar keeps the format reachable.
SYSTEM_TOOL = (
    "Du bist Auralis. Für Rechnungen nutzt du das Python-Werkzeug: "
    f"du schreibst {TOOL_OPEN}print(AUSDRUCK){TOOL_CLOSE} und bekommst das Ergebnis. "
    f"Beispiel: Frage: Berechne 12 + 15. Antwort: {TOOL_OPEN}print(12 + 15){TOOL_CLOSE}"
)

# An honest abstain contains at least one hedge (same net as gen_verified_abstain).
HEDGE = re.compile(
    r"\b(nicht|kenne|unbekannt|wei(ß|ss)|wüsste|existiert|sicher|vermutlich|keine|fiktiv|leider)\b",
    re.IGNORECASE,
)

# Deny-list pre-check for the code-axis subprocess (gen_verified_code style).
DENY = ("import os", "import sys", "import subprocess", "import socket", "import shutil",
        "open(", "exec(", "eval(", "__import__", "os.", "sys.", "subprocess", "shutil")


def _norm(text: str) -> str:
    return text.lower().replace("ß", "ss")


def _contains(haystack: str, needle: str) -> bool:
    return _norm(needle) in _norm(haystack)


def _sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


# ----------------------------- safe arithmetic (tool axis) -----------------------------
_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod, ast.Pow: operator.pow}
_UN = {ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNC = {"sqrt": math.sqrt, "abs": abs, "round": round, "min": min, "max": max,
         "sum": sum, "pow": pow}


def _ev(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
            and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        a, b = _ev(node.left), _ev(node.right)
        if isinstance(node.op, ast.Pow) and abs(b) > 1000:
            raise ValueError("exponent too large")
        return _BIN[type(node.op)](a, b)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UN:
        return _UN[type(node.op)](_ev(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNC:
        return _FUNC[node.func.id](*[_ev(a) for a in node.args])
    if isinstance(node, (ast.Tuple, ast.List)):
        return [_ev(e) for e in node.elts]
    raise ValueError("expression not allowed")


def _fmt(v) -> str:
    if isinstance(v, float):
        return str(int(v)) if v.is_integer() else f"{v:.10g}"
    return str(v)


def safe_calc(code: str) -> tuple[bool, str]:
    """(ok, output) — arithmetic-only ``print(expr)`` / bare ``expr`` lines."""
    code = re.sub(r"^```(?:python)?|```$", "", (code or "").strip()).strip()
    if not code:
        return False, "empty"
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as e:
        return False, f"syntax: {e.msg}"
    out = []
    try:
        for stmt in tree.body:
            if not isinstance(stmt, ast.Expr):
                return False, "arithmetic/print only"
            node = stmt.value
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                    and node.func.id == "print":
                out.append(" ".join(_fmt(_ev(a)) for a in node.args))
            else:
                out.append(_fmt(_ev(node)))
    except Exception as e:                                     # noqa: BLE001
        return False, f"rejected: {type(e).__name__}"
    return True, "\n".join(out)


# ----------------------------- extraction -----------------------------
def extract_code_block(text: str) -> str:
    """Best-effort code extraction: fenced block first, raw answer otherwise."""
    m = re.search(r"```(?:python)?\s*\n(.*?)(?:```|\Z)", text, flags=re.DOTALL)
    return (m.group(1) if m else text).strip()


def extract_tool_code(text: str) -> str | None:
    i = text.find(TOOL_OPEN)
    if i == -1:
        return None
    j = text.find(TOOL_CLOSE, i + len(TOOL_OPEN))
    end = j if j != -1 else len(text)
    return text[i + len(TOOL_OPEN):end].strip()


# ----------------------------- code-axis subprocess -----------------------------
def run_code_check(code: str, check: str, timeout: float = 5.0) -> bool:
    low = code.lower()
    if any(d in low for d in DENY):
        return False
    src = f"{code}\n\nassert ({check}), 'check failed'\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as fh:
        fh.write(src)
        fp = fh.name
    try:
        r = subprocess.run([sys.executable, "-I", fp], capture_output=True,
                           text=True, timeout=timeout)
        return r.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False
    finally:
        Path(fp).unlink(missing_ok=True)


# ----------------------------- scoring -----------------------------
def score_item(item: dict[str, Any], answer: str) -> float:
    axis = item["axis"]
    if axis == "qa":
        hit = any(_contains(answer, kw) for kw in item.get("expect_any", []))
        bad = any(_contains(answer, kw) for kw in item.get("forbid_any", []))
        return 1.0 if hit and not bad else 0.0
    if axis == "abstain":
        hedged = bool(HEDGE.search(answer))
        fabricated = any(_contains(answer, kw) for kw in item.get("forbid_any", []))
        return 1.0 if hedged and not fabricated else 0.0
    if axis == "code":
        code = extract_code_block(answer)
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return 0.0
        score = 0.5  # parses
        must = item.get("must_define")
        defined = (not must) or any(
            isinstance(n, ast.FunctionDef) and n.name == must for n in ast.walk(tree)
        )
        if defined:
            score += 0.25
        check = item.get("check")
        if check and defined and run_code_check(code, check):
            score += 0.25
        elif not check and defined:
            score += 0.25
        return score
    if axis == "tool":
        code = extract_tool_code(answer)
        if code is None:
            return 0.0
        ok, result = safe_calc(code)
        if not ok:
            return 0.25  # routed, but unparseable call
        return 1.0 if result.strip() == str(item.get("expect_result", "")).strip() else 0.5
    raise ValueError(f"unknown axis {axis!r}")


# ----------------------------- generation -----------------------------
def _build_prompt(item: dict[str, Any]) -> str:
    from auralis.tokenizer.chat_template import build_inference_prompt

    system = {"qa": SYSTEM_QA, "abstain": SYSTEM_QA,
              "code": SYSTEM_CODE, "tool": SYSTEM_TOOL}[item["axis"]]
    return build_inference_prompt(
        [{"role": "user", "content": item["prompt"]}], default_system=system,
    )


def greedy_generate(model, sp, prompt: str, max_new_tokens: int, device,
                    stops: tuple[str, ...] = ()) -> str:
    """Deterministic greedy decode; stops on <|end|> or any stop string."""
    import torch

    end_ids = sp.EncodeAsIds(END)
    end_id = end_ids[-1] if end_ids else sp.eos_id()
    x = torch.tensor([sp.EncodeAsIds(prompt)], dtype=torch.long, device=device)
    new_ids: list[int] = []
    autocast = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if device.type == "cuda" else torch.no_grad())
    with torch.no_grad():
        for _ in range(max_new_tokens):
            with autocast:
                out = model(input_ids=x)
            nxt = int(out["logits"][0, -1].argmax().item())
            if nxt == end_id:
                break
            new_ids.append(nxt)
            x = torch.cat([x, torch.tensor([[nxt]], dtype=torch.long, device=device)], dim=1)
            dec = sp.DecodeIds(new_ids)
            for s in stops:
                if s in dec:
                    return dec
    return sp.DecodeIds(new_ids)


def load_items(path: Path, limit_per_axis: int = 0) -> list[dict[str, Any]]:
    items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for it in items:
        if it.get("axis") not in AXES:
            raise ValueError(f"item {it.get('id')!r} has unknown axis {it.get('axis')!r}")
    if limit_per_axis > 0:
        kept, seen = [], {a: 0 for a in AXES}
        for it in items:  # fixed file order — deterministic subset
            if seen[it["axis"]] < limit_per_axis:
                kept.append(it)
                seen[it["axis"]] += 1
        items = kept
    return items


def run_gate(*, model, tokenizer_path, data_path, device,
             max_new_tokens: int = 48, limit_per_axis: int = 0) -> dict[str, Any]:
    """Run all probes; return per-axis means + composite. Deterministic."""
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    items = load_items(Path(data_path), limit_per_axis)
    results = []
    for item in items:
        n = int(item.get("max_new_tokens", max_new_tokens))
        stops = (TOOL_CLOSE,) if item["axis"] == "tool" else ()
        answer = greedy_generate(model, sp, _build_prompt(item), n, device, stops=stops)
        results.append({"id": item["id"], "axis": item["axis"],
                        "score": score_item(item, answer), "answer": answer})

    by_axis = {a: [r["score"] for r in results if r["axis"] == a] for a in AXES}
    axis_means = {a: (sum(v) / len(v) if v else 0.0) for a, v in by_axis.items()}
    present = [a for a in AXES if by_axis[a]]
    composite = sum(axis_means[a] for a in present) / max(1, len(present))
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "data_path": str(data_path),
        "data_sha16": _sha16(Path(data_path)),
        "n_items": len(results),
        "by_axis": axis_means,
        "composite": composite,
        "results": results,
    }


def main() -> None:
    import os

    import torch

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, default=REPO / "tokenizer" / "helix_v2_tokenizer.model")
    p.add_argument("--data", type=Path, default=REPO / "data" / "eval" / "heldout_gate_v1.jsonl")
    p.add_argument("--device", default="auto")
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--limit-per-axis", type=int, default=0, help="0 = all items")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))
    # Kernel backend ON by default on CUDA — the native GLA path had the
    # decay-dim bug; the gate must measure what production runs.
    if device.type == "cuda":
        os.environ.setdefault("AURALIS_USE_MAMBA_KERNEL", "1")
        os.environ.setdefault("AURALIS_USE_GLA_KERNEL", "1")

    from auralis.model import build_model

    model = build_model(args.model_config).to(device).eval()
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = {k.replace("_orig_mod.", ""): v for k, v in payload["model"].items()}
    missing, extra = model.load_state_dict(state, strict=False)
    if missing or extra:
        raise SystemExit(f"state mismatch: missing={len(missing)} extra={len(extra)}; "
                         f"first_missing={missing[:3]} first_extra={extra[:3]}")
    print(f"loaded {args.checkpoint} (step {payload.get('state', {}).get('step', '?')}) "
          f"| backends mamba={os.environ.get('AURALIS_USE_MAMBA_KERNEL', '')} "
          f"gla={os.environ.get('AURALIS_USE_GLA_KERNEL', '')}", flush=True)

    report = run_gate(model=model, tokenizer_path=args.tokenizer, data_path=args.data,
                      device=device, max_new_tokens=args.max_new_tokens,
                      limit_per_axis=args.limit_per_axis)
    report["checkpoint"] = str(args.checkpoint)
    print(f"composite={report['composite']:.3f} | "
          + " ".join(f"{a}={report['by_axis'][a]:.3f}" for a in AXES))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
