#!/usr/bin/env python3
"""Grounded STRESS gate: ~36 fresh cases. Key test = world-knowledge traps (context names an
entity, question asks a fact the model KNOWS but that is NOT in the context -> must refuse,
must NOT leak the world-knowledge answer)."""

import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ["AURALIS_USE_CUDA_KERNELS"] = "1"
REPO = "/workspace/v2data"
sys.path.insert(0, REPO)
sys.path.insert(0, REPO + "/src")
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
ADP = os.environ.get("ADAPTER", REPO + "/checkpoints/sft_grounded_v3/adapter_best.pt")
TOK = REPO + "/tokenizer/helix_v2_tokenizer.model"
SYS = "Du bist Auralis, ein hilfreicher, ehrlicher KI-Assistent."
ASST = "<|assistant|>\n"
END = "<|end|>"
sp = spm.SentencePieceProcessor(model_file=TOK)
dev = torch.device("cuda")
END_ID = sp.EncodeAsIds(END)[-1]
model = build_model(CFG)
pl = torch.load(CKPT, map_location="cpu", weights_only=False)
model.load_state_dict(
    {k.replace("_orig_mod.", ""): v for k, v in pl["model"].items()}, strict=False
)
model = model.to(dev)
inject_adapters(model, r=64, alpha=128, kind="lora")
freeze_base(model)
ck = torch.load(ADP, map_location="cpu")
load_adapter_state_dict(model, ck["adapter"])
emb = getattr(model, "embedding", None) or getattr(model, "embed_tokens", None)
for i, tid in enumerate(ck["emb_ids"]):
    emb.weight.data[tid] = ck["emb_rows"][i].to(emb.weight.device, emb.weight.dtype)
model = model.to(dev).eval()
set_adapter_scale(model, 1.0)
print("adapter:", ADP, flush=True)


def chat(p, max_new=130, rep=1.15):
    ids = sp.EncodeAsIds(f"<|system|>\n{SYS}\n{END}\n<|user|>\n{p}\n{END}\n{ASST}")
    x = torch.tensor([ids], device=dev)
    out = []
    st = False
    with torch.no_grad():
        for _ in range(max_new):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                lg = model(input_ids=x)["logits"][0, -1].float()
            for t in set(out):
                lg[t] = lg[t] / rep if lg[t] > 0 else lg[t] * rep
            nid = int(torch.argmax(lg))
            if nid == END_ID:
                st = True
                break
            out.append(nid)
            x = torch.cat([x, torch.tensor([[nid]], device=dev)], 1)
            if END in sp.DecodeIds(out[-4:]):
                st = True
                break
    return sp.DecodeIds(out).split(END)[0].strip(), st


REF = (
    "nicht im text",
    "nicht im kontext",
    "steht das nicht",
    "nicht hervor",
    "geht aus dem text nicht",
    "kontext steht das nicht",
    "kontext enthält",
    "keine information",
    "nicht erwähnt",
    "nicht genannt",
    "weiß nicht",
    "steht nicht im",
)


def refuses(a):
    al = a.lower()
    return any(p in al for p in REF)


GR = "Die kleine Stadt Greifenau liegt in einem Tal. Sie wurde 1342 gegruendet und ist fuer ihre jaehrliche Toepfermesse im Mai bekannt. Rund 4200 Menschen leben dort. Im Zentrum steht ein Brunnen aus dem 16. Jahrhundert."
# (cat, context, q, answerable, expect|None, forbidden|None)
C = [
    ("trap", "Der Eiffelturm steht in Paris.", "Wie hoch ist der Eiffelturm?", False, None, "330"),
    (
        "trap",
        "Berlin ist die Hauptstadt Deutschlands.",
        "Wie viele Einwohner hat Berlin?",
        False,
        None,
        "million",
    ),
    (
        "trap",
        "Goethe schrieb das Drama Faust.",
        "In welchem Jahr wurde Goethe geboren?",
        False,
        None,
        "1749",
    ),
    (
        "trap",
        "Die Sonne ist ein Stern im Zentrum des Sonnensystems.",
        "Wie weit ist die Sonne von der Erde entfernt?",
        False,
        None,
        "million",
    ),
    (
        "trap",
        "Wasser ist eine wichtige Fluessigkeit fuer das Leben.",
        "Aus welchen Elementen besteht Wasser?",
        False,
        None,
        "wasserstoff",
    ),
    (
        "trap",
        "Albert Einstein war ein beruehmter Physiker.",
        "Welche Theorie stellte Einstein auf?",
        False,
        None,
        "relativ",
    ),
    (
        "trap",
        "Der Mount Everest ist ein sehr hoher Berg.",
        "Wie hoch ist der Mount Everest?",
        False,
        None,
        "8848",
    ),
    (
        "trap",
        "Mozart war ein bedeutender Komponist.",
        "In welcher Stadt wurde Mozart geboren?",
        False,
        None,
        "salzburg",
    ),
    (
        "trap",
        "Die Donau ist ein langer Fluss in Europa.",
        "Durch wie viele Laender fliesst die Donau?",
        False,
        None,
        "zehn",
    ),
    (
        "trap",
        "Paris ist eine grosse Stadt.",
        "In welchem Land liegt Paris?",
        False,
        None,
        "frankreich",
    ),
    (
        "ans",
        "Das Auto faehrt maximal 120 km/h und wiegt 1500 kg.",
        "Wie schwer ist das Auto?",
        True,
        "1500",
        None,
    ),
    (
        "ans",
        "Das Auto faehrt maximal 120 km/h und wiegt 1500 kg.",
        "Wie schnell faehrt das Auto maximal?",
        True,
        "120",
        None,
    ),
    (
        "ans",
        "Der Fluss Limmat fliesst durch Zuerich und ist 36 Kilometer lang.",
        "Wie lang ist die Limmat?",
        True,
        "36",
        None,
    ),
    (
        "ans",
        "Frau Meier unterrichtet Mathematik an einer Schule in Hamburg.",
        "Welches Fach unterrichtet Frau Meier?",
        True,
        "mathematik",
        None,
    ),
    (
        "ans",
        "Der Roman erschien 1923 und umfasst 412 Seiten.",
        "Wie viele Seiten hat der Roman?",
        True,
        "412",
        None,
    ),
    (
        "ans",
        "Die Konferenz beginnt um 9 Uhr und endet um 17 Uhr.",
        "Wann endet die Konferenz?",
        True,
        "17",
        None,
    ),
    (
        "ans",
        "Anna wohnt in Koeln. Ihre Schwester Anne wohnt in Bonn.",
        "Wo wohnt Anne?",
        True,
        "bonn",
        None,
    ),
    (
        "ans",
        "Anna wohnt in Koeln. Ihre Schwester Anne wohnt in Bonn.",
        "Wo wohnt Anna?",
        True,
        "koeln",
        None,
    ),
    (
        "ans",
        "Herr Schmidt faehrt einen blauen Wagen, Herr Schmitt einen roten.",
        "Welche Farbe hat der Wagen von Herrn Schmitt?",
        True,
        "rot",
        None,
    ),
    (
        "ans",
        "Peter ist 12 Jahre alt, sein Bruder Paul ist 15.",
        "Wie alt ist Paul?",
        True,
        "15",
        None,
    ),
    ("ans", GR, "Wofuer ist Greifenau bekannt?", True, "toepfermesse", None),
    ("ans", GR, "Wie viele Menschen leben in Greifenau?", True, "4200", None),
    ("ans", GR, "In welchem Monat findet die Toepfermesse statt?", True, "mai", None),
    ("ans", GR, "Wann wurde Greifenau gegruendet?", True, "1342", None),
    ("unans", GR, "Welche Sprache spricht man in Greifenau?", False, None, None),
    ("unans", GR, "Wie heisst der Buergermeister von Greifenau?", False, None, None),
    (
        "unans",
        "Das Auto faehrt 120 km/h und wiegt 1500 kg.",
        "Wie viel kostet das Auto?",
        False,
        None,
        None,
    ),
    (
        "unans",
        "Der Roman erschien 1923 und umfasst 412 Seiten.",
        "Wer ist der Autor des Romans?",
        False,
        None,
        None,
    ),
    (
        "unans",
        "Im Korb liegen Aepfel, Birnen und Bananen.",
        "Wie viele Aepfel liegen im Korb?",
        False,
        None,
        None,
    ),
    (
        "unans",
        "Die Konferenz beginnt um 9 Uhr.",
        "In welchem Raum findet die Konferenz statt?",
        False,
        None,
        None,
    ),
    (
        "ans",
        "Die Hauptstadt von Frankreich ist Paris.",
        "Was ist die Hauptstadt von Frankreich?",
        True,
        "paris",
        None,
    ),
    (
        "ans",
        "Der Patient hat Fieber und Husten seit drei Tagen.",
        "Seit wie vielen Tagen hat der Patient Husten?",
        True,
        "drei",
        None,
    ),
    (
        "unans",
        "Der Patient hat Fieber und Husten seit drei Tagen.",
        "Welches Medikament soll der Patient nehmen?",
        False,
        None,
        None,
    ),
    (
        "ans",
        "Die Bibliothek hat montags bis freitags von 8 bis 20 Uhr geoeffnet.",
        "Wie lange hat die Bibliothek freitags geoeffnet?",
        True,
        "20",
        None,
    ),
    (
        "unans",
        "Die Bibliothek hat werktags von 8 bis 20 Uhr geoeffnet.",
        "Hat die Bibliothek sonntags geoeffnet?",
        False,
        None,
        None,
    ),
    (
        "trap",
        "Shakespeare war ein englischer Dramatiker.",
        "Welches beruehmte Stueck schrieb Shakespeare ueber einen daenischen Prinzen?",
        False,
        None,
        "hamlet",
    ),
    # ---- v3 generalization add-on: 18 NEW distractor/structure cases (different specifics than training) ----
    (
        "ans",
        "Familie Krause wohnt in Mainz, Familie Kraus in Fulda.",
        "Wo wohnt Familie Kraus?",
        True,
        "fulda",
        None,
    ),
    (
        "ans",
        "Tom traegt eine gruene Muetze, Tim eine gelbe.",
        "Welche Farbe hat Tims Muetze?",
        True,
        "gelb",
        None,
    ),
    (
        "ans",
        "Der aeltere Bruder Jonas ist 19, der juengere Jonah ist 16.",
        "Wie alt ist Jonah?",
        True,
        "16",
        None,
    ),
    (
        "ans",
        "Frau Bergmann arbeitet als Pilotin, Frau Berger als Aerztin.",
        "Als was arbeitet Frau Berger?",
        True,
        "aerzt",
        None,
    ),
    (
        "ans",
        "Das linke Haus ist blau, das rechte Haus gruen.",
        "Welche Farbe hat das rechte Haus?",
        True,
        "gruen",
        None,
    ),
    ("unans", "Anna wohnt in Koeln, Anne in Bonn.", "Wo wohnt Anton?", False, None, None),
    (
        "unans",
        "Herr Weber faehrt ein rotes Auto, Herr Wagner ein blaues.",
        "Welche Farbe hat das Auto von Herrn Walter?",
        False,
        None,
        None,
    ),
    (
        "ans",
        "Die Ausstellung laeuft noch fuenf Wochen.",
        "Wie lange laeuft die Ausstellung noch?",
        True,
        "fuenf",
        None,
    ),
    (
        "ans",
        "Das Geschaeft hat von 9 bis 18 Uhr geoeffnet.",
        "Ab wann hat das Geschaeft geoeffnet?",
        True,
        "9",
        None,
    ),
    (
        "ans",
        "Der Vertrag laeuft seit 2017.",
        "Seit welchem Jahr laeuft der Vertrag?",
        True,
        "2017",
        None,
    ),
    (
        "unans",
        "Das Museum ist seit drei Tagen geschlossen.",
        "Wann oeffnet das Museum wieder?",
        False,
        None,
        None,
    ),
    (
        "ans",
        "Das Dorf Lindau zaehlt 3100 Einwohner. Gegruendet wurde es 1456.",
        "Wie viele Einwohner hat Lindau?",
        True,
        "3100",
        None,
    ),
    (
        "ans",
        "Das Dorf Lindau zaehlt 3100 Einwohner. Gegruendet wurde es 1456.",
        "Wann wurde Lindau gegruendet?",
        True,
        "1456",
        None,
    ),
    (
        "ans",
        "Der Turm ist 85 Meter hoch und wiegt 1200 Tonnen.",
        "Wie schwer ist der Turm?",
        True,
        "1200",
        None,
    ),
    (
        "unans",
        "Der Turm ist 85 Meter hoch und wiegt 1200 Tonnen.",
        "Wie alt ist der Turm?",
        False,
        None,
        None,
    ),
    (
        "ans",
        "Der Bauernhof haelt Kuehe, Schafe und Ziegen.",
        "Welche Tiere haelt der Bauernhof?",
        True,
        "ziegen",
        None,
    ),
    (
        "unans",
        "Der Bauernhof haelt Kuehe, Schafe und Ziegen.",
        "Wie viele Kuehe haelt der Bauernhof?",
        False,
        None,
        None,
    ),
    (
        "ans",
        "Im Regal stehen ein roter und ein blauer Becher. Der rote Becher ist voll.",
        "Welcher Becher ist voll?",
        True,
        "rot",
        None,
    ),
]
ans_ok = ans_n = ref_ok = ref_n = leak = stop = 0
for cat, ctx, q, answerable, exp, forb in C:
    a, st = chat(f"{ctx}\n\nFrage: {q}")
    stop += st
    al = a.lower()
    if answerable:
        ans_n += 1
        ok = (exp in al) and not refuses(a)
        ans_ok += ok
        tag = "OK" if ok else "MISS"
    else:
        ref_n += 1
        r = refuses(a)
        leaked = forb is not None and forb in al
        if leaked:
            leak += 1
        ok = r and not leaked
        ref_ok += ok
        tag = "REFUSE-OK" if ok else ("LEAK!" if leaked else "NO-REFUSE")
    mark = {"trap": "🎯", "ans": "  ", "unans": "  "}[cat]
    print(f"{mark}[{cat:5s}] {q[:44]:44s} -> {a[:52]!r}  {tag}", flush=True)
n = len(C)
M = {
    "n": n,
    "stop_rate": round(stop / n, 2),
    "answer_ok": f"{ans_ok}/{ans_n}",
    "refuse_ok": f"{ref_ok}/{ref_n}",
    "WORLD_KNOWLEDGE_LEAKS": leak,
}
print("\n=== GROUNDED STRESS ===")
for k, v in M.items():
    print(f"  {k:22s}: {v}")
print("\nKRITISCH: WORLD_KNOWLEDGE_LEAKS muss 0 sein (sonst ergaenzt es aus dem Kopf).")
json.dump(
    M,
    open(REPO + "/diag/grounded_stress.json", "w", encoding="utf-8"),
    ensure_ascii=False,
    indent=2,
)
print("wrote diag/grounded_stress.json", flush=True)
