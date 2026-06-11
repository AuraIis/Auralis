#!/usr/bin/env python3
"""Before/after prose comparison: OLD filter (strict boilerplate, no LID/PII/dedup)
vs IMPROVED (density boilerplate + sentence-strip, LID gate, PII mask, dup collapse)."""
import sys, json, argparse, re
from pathlib import Path
from collections import Counter
sys.path.insert(0, "/workspace/v2data")
from scripts.data import filter_quality as fq

import fasttext
_FT = fasttext.load_model("/workspace/v2data/models/lid.176.ftz")
class LID:
    def classify(self, text):
        t=" ".join(text[:1200].split())
        if not t: return "??",0.0
        labs=_FT.f.predict(t+"\n",1,0.0,"strict")
        if not labs: return "??",0.0
        p,l=labs[0]; return l.replace("__label__",""), min(p,1.0)
lid=LID()

def run(path, expect, is_jsonl, min_len):
    old=Counter(); new=Counter()
    recovered=0; junk_caught=Counter(); pii=0; dup=0; boil_strip=0
    n=0
    for line in path.open(encoding="utf-8",errors="replace"):
        if is_jsonl:
            try: text=json.loads(line)["text"]
            except Exception: continue
        else:
            text=line.rstrip("\n")
        n+=1
        # OLD: strict boilerplate, no lid
        r_old=fq._passes(text,min_length=min_len,max_length=100000,preserve_newlines=False,
                         allow_mojibake=False,strict_boilerplate=True)
        # NEW: density boilerplate + lid gate
        r_new=fq._passes(text,min_length=min_len,max_length=100000,preserve_newlines=False,
                         allow_mojibake=False,strict_boilerplate=False,
                         lid=lid,lid_expect=expect,lid_conf=0.65)
        old["kept" if r_old is None else f"drop:{r_old}"]+=1
        # apply new repairs to compute final kept
        if r_new is None:
            norm=fq._normalise(text,False)
            # boilerplate sentence strip
            if fq._boilerplate_hits(norm.lower()):
                s=fq._strip_boilerplate_sentences(norm)
                if len(s) < max(min_len,int(0.7*len(norm))):
                    new["drop:boilerplate"]+=1; continue
                if s!=norm: boil_strip+=1
                norm=s
            c=fq._collapse_dup_sentences(norm)
            if c!=norm: dup+=1
            norm=c
            if len(norm)<min_len:
                new["drop:too_short"]+=1; continue
            m=fq._strip_pii(norm)
            if m!=norm: pii+=1
            new["kept"]+=1
            # was it junk the OLD filter PASSED? (lang mismatch caught only by new)
            if r_old is None and r_new is None:
                pass
        else:
            new[f"drop:{r_new}"]+=1
        # recovered = OLD dropped (boilerplate/repetitive/mojibake) but NEW keeps as good expected-lang
        if r_old is not None and r_old in ("boilerplate","repetitive","mojibake") and r_new is None:
            lang,conf=lid.classify(text)
            if lang==expect and conf>0.8: recovered+=1
        # junk caught = OLD kept but NEW drops as wrong-language
        if r_old is None and r_new is not None and r_new.startswith("lang_"):
            junk_caught[r_new]+=1
    return {"docs":n,"old":dict(old),"new":dict(new),
            "false_drops_recovered":recovered,
            "wrong_lang_junk_caught":dict(junk_caught),
            "pii_masked":pii,"dup_collapsed":dup,"boilerplate_sentences_stripped":boil_strip}

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--out",type=Path,required=True); a=ap.parse_args()
    D=Path("/workspace/v2data/diag/clean_audit_v1")
    jobs=[("de_fresh",D/"de_fresh.10k.jsonl","de",True,300),
          ("gc_edu",D/"gc_edu.10k.jsonl","de",True,300),
          ("se_os",D/"se_os.10k.jsonl","en",True,200),
          ("raw_fineweb2_de",D/"raw_fineweb2_de.10k.txt","de",False,300),
          ("raw_hplt_de",D/"raw_hplt_de.10k.txt","de",False,300),
          ("raw_gc",D/"raw_gc.10k.txt","de",False,300),
          ("raw_fineweb_en",D/"raw_fineweb_en.10k.txt","en",False,200)]
    out={}
    for name,p,exp,jl,ml in jobs:
        if p.exists():
            print("==",name,flush=True); out[name]=run(p,exp,jl,ml)
    a.out.write_text(json.dumps(out,indent=2,ensure_ascii=False))
    print("wrote",a.out)
