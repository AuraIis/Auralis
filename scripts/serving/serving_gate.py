#!/usr/bin/env python3
"""Routing/serving gate (NOT training): hits the running Helix shim and checks each mode behaves.
katze->no abstain, hardware->explanation, gpu->explained (acronym fix), goblin->honest abstain,
hallo->normal, Paris->correct, 5x5->tool. Run inside auralis-blackwell."""

import json
import urllib.request


def ask(model, prompt):
    data = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(
        "http://localhost:11434/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    buf = ""
    with urllib.request.urlopen(req, timeout=90) as r:
        for line in r:
            line = line.decode("utf-8", "replace").strip()
            if line.startswith("data: "):
                d = line[6:].strip()
                if d == "[DONE]":
                    break
                try:
                    buf += json.loads(d)["choices"][0]["delta"].get("content", "")
                except Exception:
                    pass
    return buf


ABSTAIN = (
    "weiß nicht",
    "weiss nicht",
    "nicht vorzukommen",
    "existiert vermutlich nicht",
    "kein eigenständiger",
    "frei erfunden",
    "nicht als eigenständig",
)


def abstains(t):
    tl = t.lower()
    return any(p in tl for p in ABSTAIN)


CASES = [
    (
        "helix-chat",
        "katze",
        lambda t: not abstains(t) and "katze" in t.lower(),
        "kohaerent, kein Abstain",
    ),
    (
        "helix-chat",
        "hardware",
        lambda t: not abstains(t) and "hardware" in t.lower(),
        "Begriffserklaerung",
    ),
    ("helix-chat", "gpu", lambda t: not abstains(t), "GPU erklaert (Akronym-Fix)"),
    ("helix-chat", "goblin", lambda t: abstains(t), "ehrlich unsicher (Wissensgrenze, ok)"),
    ("helix-chat", "hallo", lambda t: not abstains(t), "normale Antwort"),
    (
        "helix-chat",
        "Was ist die Hauptstadt von Frankreich?",
        lambda t: "paris" in t.lower(),
        "Paris korrekt",
    ),
    (
        "helix-corrective-tools",
        "was sind 5 x 5?",
        lambda t: "25" in t and "result" in t.lower(),
        "Tool -> 25",
    ),
    (
        "helix-grounded",
        "Das Auto wiegt 1500 kg.\n\nFrage: Wie schwer ist das Auto?",
        lambda t: "1500" in t,
        "Grounded-Extraktion",
    ),
]
ok = 0
for model, prompt, check, desc in CASES:
    t = ask(model, prompt)
    p = check(t)
    ok += p
    print(
        f"[{'PASS' if p else 'FAIL'}] {model:24s} {prompt[:30]:30s} | {desc}\n        -> {t[:90]!r}",
        flush=True,
    )
print(f"\n=== ROUTING GATE: {ok}/{len(CASES)} ===")
