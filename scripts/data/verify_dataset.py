#!/usr/bin/env python3
"""verify_dataset.py — pre-launch QA gate for tokenized pretraining bins.

One command checks every <key>.bin/<key>.idx of a training mix and emits a
PASS/WARN/FAIL scorecard. Exit code 1 if any dataset FAILs, so the launch
script can gate on it. Designed to catch the failure classes that have burnt
real money before: stale/misaligned bins, mid-cut docs (german_commons 21%),
eval contamination (MBPP 28%), tokenizer regressions, cross-source dup leaks.

Checks per dataset (sampling-based where a full scan is too slow; fixed seed):
  integrity      idx/bin alignment (offsets contiguous, totals match, no
                 zero-length docs), EOS at end of doc, token ids in [0, vocab)
  truncation     doc-length distribution, fraction of docs at exact max length
  dedup          exact dup rate within bin (full-doc token hash on a sample)
                 + near-dup shingle overlap ACROSS bins
  quality        bytes/token, language ID (de/en stopwords) vs expected
                 language, symbol/whitespace ratios, max repetition share
  contamination  decodes a sample and runs check_code_eval_contamination.py
                 (HumanEval/MBPP/GSM8K/GermanQuAD/MMLU-DE), with self-test

Usage (container):
  python3 scripts/data/verify_dataset.py \
      --manifest configs/training/corpus20b_codeheavy.yaml \
      --tokenizer /workspace/v2data/tokenizer/helix_v2_tokenizer.model \
      --evals-dir /tmp --json-out eval/results/corpus20b_qa_scorecard.json
  python3 scripts/data/verify_dataset.py --bin tokenized/corpus20b/math.bin ...
  python3 scripts/data/verify_dataset.py --self-test --tokenizer ...   # planted defects
"""
from __future__ import annotations
import argparse, hashlib, json, os, pathlib, subprocess, sys, time
import numpy as np

SEED = 20260611
EOS_ID = 3
SAMPLE_HASH_DOCS = 50_000      # exact-dup sample per bin
SAMPLE_DECODE_DOCS = 600       # decoded sample: langid, bytes/token, contamination
SAMPLE_SPOT_DOCS = 4096        # eos / token-range spot checks
SHINGLE_N = 8                  # token shingles for cross-bin near-dup
SHINGLE_CAP = 400_000

DE_STOP = {"der","die","das","und","nicht","ist","ein","eine","den","sich","mit","auch","auf","für","von","dem","sie","werden","wird","sind","bei","oder","wie","wir","aber","nach","über","nur","durch","kann","wenn","mehr","zum","zur","dass","dass","einer","einem","als","des","im","um"}
EN_STOP = {"the","and","of","to","in","is","that","for","it","was","with","as","on","are","this","be","by","at","from","or","an","not","have","but","you","they","his","which","one","were","all","we","can","has","will","more","when","there","their","what","if"}

# expected language per key-prefix (None = no gate, just report)
LANG_EXPECT = [("german", "de"), ("de", "de"), ("code", None), ("math", "en"),
               ("stackexchange", "en"), ("en", "en")]
# which contamination banks gate which keys (others reported, not gated)
CONTAM_GATE = {"code": ["humaneval", "mbpp"], "math": ["gsm8k"],
               "german": ["germanquad", "mmlu_de"], "de": ["germanquad", "mmlu_de"],
               "en": ["humaneval", "mbpp", "gsm8k"],
               "stackexchange": ["humaneval", "mbpp"], "selftest": ["mbpp"]}

TH = {  # (warn, fail)
    "trunc_frac": (0.05, 0.15), "dup_frac": (0.01, 0.05), "cross_overlap": (0.02, 0.10),
    "contam": (0.005, 0.02), "lang_frac_min": (0.80, 0.50),
    "bpt": ((2.0, 7.0), (1.2, 10.0)), "rep_frac": (0.05, 0.20), "sym_ratio": (0.35, 0.60),
}


def sample_indices(n_docs: int, want: int, rng, n_blocks: int = 64) -> np.ndarray:
    """Sample `want` doc indices as contiguous blocks (sequential I/O, not 50k seeks)."""
    if want >= n_docs:
        return np.arange(n_docs)
    per = max(want // n_blocks, 1)
    starts = rng.choice(max(n_docs - per, 1), size=n_blocks, replace=False)
    idx = np.unique(np.concatenate([np.arange(s, min(s + per, n_docs)) for s in np.sort(starts)]))
    return idx[:want]


def sev_max(a, b):
    order = {"PASS": 0, "WARN": 1, "FAIL": 2}
    return a if order[a] >= order[b] else b


class Card:
    def __init__(self, key):
        self.key, self.status, self.checks = key, "PASS", {}
    def add(self, name, status, **kw):
        self.checks[name] = {"status": status, **kw}
        self.status = sev_max(self.status, status)


def load_idx(idx_path: pathlib.Path):
    idx = np.fromfile(idx_path, dtype=np.int64).reshape(-1, 2)
    return idx[:, 0], idx[:, 1]


def load_doc_sample(bin_path, off, ln, rng, want=SAMPLE_HASH_DOCS, n_blocks=64):
    """Read sampled docs via large sequential block reads (not per-doc seeks)."""
    idx = sample_indices(len(off), want, rng, n_blocks)
    docs = []
    with open(bin_path, "rb") as f:
        runs, s = [], 0
        while s < len(idx):
            e = s
            while e + 1 < len(idx) and idx[e + 1] == idx[e] + 1:
                e += 1
            runs.append((idx[s], idx[e])); s = e + 1
        for a, b in runs:
            start, end = int(off[a]), int(off[b] + ln[b])
            f.seek(start * 4)
            buf = np.fromfile(f, dtype=np.uint32, count=end - start)
            for i in range(a, b + 1):
                o = int(off[i] - start)
                docs.append(buf[o:o + int(ln[i])])
    return docs


def check_integrity(card, bin_path, idx_path, vocab, rng):
    nbytes = bin_path.stat().st_size
    if nbytes % 4 or idx_path.stat().st_size % 16:
        card.add("integrity", "FAIL", reason="bin not %4 or idx not %16 bytes"); return None
    off, ln = load_idx(idx_path)
    ntok = nbytes // 4
    bad = []
    if (ln <= 0).any(): bad.append(f"{int((ln<=0).sum())} zero/neg-length docs")
    if len(off) and off[0] != 0: bad.append("first offset != 0")
    if len(off) > 1 and (off[:-1] + ln[:-1] != off[1:]).any():
        bad.append(f"{int((off[:-1]+ln[:-1]!=off[1:]).sum())} non-contiguous offsets")
    if len(off) and off[-1] + ln[-1] != ntok:
        bad.append(f"last doc ends {int(off[-1]+ln[-1])} != {ntok} bin tokens (stale bin?)")
    docs = load_doc_sample(bin_path, off, ln, rng)
    spot = docs if len(docs) <= SAMPLE_SPOT_DOCS else [docs[i] for i in
            np.random.default_rng(SEED).choice(len(docs), SAMPLE_SPOT_DOCS, replace=False)]
    eos_ok = oov = 0
    for d in spot:
        if d.size == 0:
            continue
        eos_ok += int(d[-1] == EOS_ID)
        oov += int((d >= vocab).sum())
    eos_frac = eos_ok / max(len(spot), 1)
    if oov: bad.append(f"{oov} token ids >= vocab({vocab}) in sample")
    if eos_frac < 0.999: bad.append(f"only {eos_frac:.4f} of docs end with EOS")
    card.add("integrity", "FAIL" if bad else "PASS", n_docs=int(len(off)),
             n_tokens=int(ntok), eos_frac=round(eos_frac, 5), oov_in_sample=int(oov),
             problems=bad)
    return off, ln, docs


def check_truncation(card, ln):
    mx = int(ln.max()); frac = float((ln == mx).mean())
    near = float((ln >= mx - 1).mean())
    q = {p: int(np.percentile(ln, p)) for p in (50, 90, 99)}
    w, f = TH["trunc_frac"]
    st = "FAIL" if frac >= f else "WARN" if frac >= w else "PASS"
    card.add("truncation", st, max_len=mx, frac_at_max=round(frac, 4),
             frac_at_max_minus1=round(near, 4), min=int(ln.min()), pct=q,
             mean=round(float(ln.mean()), 1))


def check_dups(card, docs):
    seen, dup = set(), 0
    for d in docs:
        h = hashlib.md5(d.tobytes()).digest()
        dup += h in seen; seen.add(h)
    frac = dup / max(len(docs), 1)
    w, f = TH["dup_frac"]
    card.add("dedup_within", "FAIL" if frac >= f else "WARN" if frac >= w else "PASS",
             sampled=len(docs), dup_frac=round(frac, 4))


def shingles(docs, n_docs=2000):
    out = set()
    step = max(len(docs) // n_docs, 1)
    for d in docs[::step]:
        if len(d) < SHINGLE_N: continue
        v = np.lib.stride_tricks.sliding_window_view(d, SHINGLE_N)[::4]
        out.update(hashlib.md5(r.tobytes()).digest()[:8] for r in v[:2000])
        if len(out) > SHINGLE_CAP: break
    return out


def decode_sample(sp, docs, n=SAMPLE_DECODE_DOCS, clip=4096):
    step = max(len(docs) // n, 1)
    texts, toks = [], 0
    for d in docs[::step][:n]:
        ids = d[:clip].tolist()
        toks += len(ids)
        texts.append(sp.decode([t for t in ids if t != EOS_ID]))
    return texts, toks


def check_quality(card, key, texts, toks):
    blob = "".join(texts); nb = len(blob.encode("utf-8"))
    bpt = nb / max(toks, 1)
    ws = sum(c.isspace() for c in blob) / max(len(blob), 1)
    sym = sum(not (c.isalnum() or c.isspace()) for c in blob) / max(len(blob), 1)
    de = en = 0
    for t in texts:
        w = [x.strip(".,;:!?()\"'").lower() for x in t.split()[:400]]
        d = sum(x in DE_STOP for x in w); e = sum(x in EN_STOP for x in w)
        de += d > e; en += e > d
    de_frac = de / max(len(texts), 1)
    rep = 0
    for t in texts:
        w = t.split()
        if len(w) >= 64:
            wins = [" ".join(w[i:i + 32]) for i in range(0, len(w) - 32, 32)]
            rep += (1 - len(set(wins)) / len(wins)) > 0.5
    rep_frac = rep / max(len(texts), 1)
    (bw_lo, bw_hi), (bf_lo, bf_hi) = TH["bpt"]
    st = "FAIL" if not (bf_lo <= bpt <= bf_hi) else "WARN" if not (bw_lo <= bpt <= bw_hi) else "PASS"
    st = sev_max(st, "WARN" if rep_frac >= TH["rep_frac"][0] else "PASS")
    st = sev_max(st, "FAIL" if rep_frac >= TH["rep_frac"][1] else "PASS")
    exp = next((v for s, v in LANG_EXPECT if key.startswith(s)), None)
    lang_st, frac = "PASS", de_frac if exp == "de" else 1 - de_frac
    if exp:
        w, f = TH["lang_frac_min"]
        lang_st = "FAIL" if frac < f else "WARN" if frac < w else "PASS"
    card.add("quality", st, bytes_per_token=round(bpt, 2), ws_ratio=round(ws, 3),
             sym_ratio=round(sym, 3), rep_frac=round(rep_frac, 3))
    card.add("language", lang_st, de_frac=round(de_frac, 3), expected=exp or "any")


def run_contamination(cards_texts, evals_dir, tmpdir, ngram_override=0):
    script = pathlib.Path(__file__).with_name("check_code_eval_contamination.py")
    files, paths = [], {}
    for key, texts in cards_texts.items():
        p = tmpdir / f"sample_{key}.jsonl"
        p.write_text("\n".join(json.dumps({"text": t}) for t in texts), encoding="utf-8")
        files.append(str(p)); paths[p.name] = key
    ev = {n: evals_dir / f for n, f in [("humaneval", "HumanEval.jsonl.gz"), ("mbpp", "mbpp.jsonl"),
          ("gsm8k", "gsm8k_test.jsonl"), ("germanquad", "germanquad_test.jsonl"), ("mmlu_de", "mmlu_de_test.jsonl")]}
    jout = tmpdir / "contam.json"
    cmd = [sys.executable, str(script), "--inputs", ",".join(files), "--self-test",
           "--json-out", str(jout), "--dump-hits", "2"]
    for n, p in ev.items():
        if p.exists(): cmd += [f"--{n.replace('_','-')}", str(p)]
    if ngram_override: cmd += ["--ngram", str(ngram_override)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    if r.returncode:
        sys.stdout.write(r.stderr)
        return None
    rep = json.loads(jout.read_text())
    return {paths[name]: v for name, v in rep["inputs"].items()}


def contam_status(key, per_set):
    gated = next((v for s, v in CONTAM_GATE.items() if key.startswith(s)), [])
    worst, gr = "PASS", 0.0
    w, f = TH["contam"]
    for s in gated:
        r = per_set.get(s, 0.0); gr = max(gr, r)
        worst = sev_max(worst, "FAIL" if r >= f else "WARN" if r >= w else "PASS")
    return worst, gr, gated


def verify_bin(key, bin_path, vocab, sp, evals_dir, decode_n):
    rng = np.random.default_rng(SEED)
    card = Card(key)
    t0 = time.time()
    res = check_integrity(card, bin_path, bin_path.with_suffix(".idx"), vocab, rng)
    if res is None:
        return card, None, None
    off, ln, docs = res
    check_truncation(card, ln)
    check_dups(card, docs)
    sh = shingles(docs)
    texts, toks = decode_sample(sp, docs, n=decode_n)
    check_quality(card, key, texts, toks)
    card.checks["_secs"] = round(time.time() - t0, 1)
    return card, sh, texts


def make_selftest_bins(tmpdir, sp):
    """Plant: truncated doc (no EOS) + 30% exact dups + MBPP problem doc."""
    mbpp_text = ("Write a function to find the similar elements from the given two tuple "
                 "lists. assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)")
    docs = []
    base = sp.encode("Der schnelle braune Fuchs springt über den faulen Hund. " * 30) + [EOS_ID]
    for _ in range(40): docs.append(base)                      # exact dups
    for i in range(60): docs.append(sp.encode(f"Einzigartiges deutsches Dokument Nummer {i} über Geschichte und Wissenschaft. " * 20) + [EOS_ID])
    docs.append(sp.encode(mbpp_text) + [EOS_ID])               # contaminated
    docs.append(sp.encode("abgeschnittenes Dokument ohne Ende " * 50))  # no EOS (truncated)
    flat, idx, o = [], [], 0
    for d in docs:
        flat += d; idx += [o, len(d)]; o += len(d)
    (tmpdir / "selftest.bin").write_bytes(np.asarray(flat, np.uint32).tobytes())
    (tmpdir / "selftest.idx").write_bytes(np.asarray(idx, np.int64).tobytes())
    # stale-idx variant: idx claims one more doc than bin holds
    (tmpdir / "stale.bin").write_bytes(np.asarray(flat[:o - 100], np.uint32).tobytes())
    (tmpdir / "stale.idx").write_bytes(np.asarray(idx, np.int64).tobytes())
    return tmpdir / "selftest.bin", tmpdir / "stale.bin"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=pathlib.Path, help="training yaml (data.data_dir + mix_ratios)")
    ap.add_argument("--bin", action="append", default=[], help="explicit <key>.bin (repeatable)")
    ap.add_argument("--data-dir", type=pathlib.Path)
    ap.add_argument("--tokenizer", required=True, type=pathlib.Path)
    ap.add_argument("--evals-dir", type=pathlib.Path, default=pathlib.Path("/tmp"))
    ap.add_argument("--json-out", type=pathlib.Path, default=pathlib.Path("qa_scorecard.json"))
    ap.add_argument("--decode-docs", type=int, default=SAMPLE_DECODE_DOCS)
    ap.add_argument("--self-test", action="store_true", help="verify planted defects are flagged, then exit")
    a = ap.parse_args()

    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=str(a.tokenizer))
    vocab = sp.vocab_size()

    bins: dict[str, pathlib.Path] = {}
    if a.manifest:
        import yaml
        cfg = yaml.safe_load(a.manifest.read_text())
        dd = pathlib.Path(cfg["data"]["data_dir"])
        for k in cfg["data"]["mix_ratios"]:
            bins[k] = dd / f"{k}.bin"
    for b in a.bin:
        p = pathlib.Path(b); bins[p.stem] = p
    if a.data_dir:
        for p in sorted(a.data_dir.glob("*.bin")): bins[p.stem] = p

    tmpdir = pathlib.Path("/tmp/qa_harness"); tmpdir.mkdir(exist_ok=True)

    if a.self_test:
        good, stale = make_selftest_bins(tmpdir, sp)
        card, _, texts = verify_bin("selftest_german", good, vocab, sp, a.evals_dir, 200)
        contam = run_contamination({"selftest_german": texts}, a.evals_dir, tmpdir)
        crate = contam["selftest_german"]["rate"] if contam else 0
        card2, _, _ = verify_bin("stale", stale, vocab, sp, a.evals_dir, 10)
        ok = (card.checks["dedup_within"]["status"] != "PASS"
              and card.checks["integrity"]["eos_frac"] < 1.0
              and crate > 0 and card2.checks["integrity"]["status"] == "FAIL")
        print(json.dumps({k: v for k, v in card.checks.items()}, indent=1))
        print(f"stale-bin integrity: {card2.checks['integrity']}")
        print(f"planted contamination rate: {crate:.4f}")
        print("SELF-TEST", "PASS — all planted defects flagged" if ok else "FAIL")
        return 0 if ok else 1

    if not bins:
        sys.exit("no bins given (--manifest/--bin/--data-dir)")

    cards, shing, samples = {}, {}, {}
    for key, bp in bins.items():
        print(f"--- {key} ({bp})", flush=True)
        card, sh, texts = verify_bin(key, bp, vocab, sp, a.evals_dir, a.decode_docs)
        cards[key] = card
        if sh: shing[key] = sh
        if texts: samples[key] = texts

    # cross-bin near-dup (shingle overlap, sampled)
    keys = sorted(shing)
    cross = {}
    for i, k1 in enumerate(keys):
        worst, who = 0.0, ""
        for k2 in keys:
            if k1 == k2: continue
            inter = len(shing[k1] & shing[k2]) / max(min(len(shing[k1]), len(shing[k2])), 1)
            cross.setdefault(k1, {})[k2] = round(inter, 4)
            if inter > worst: worst, who = inter, k2
        w, f = TH["cross_overlap"]
        st = "FAIL" if worst >= f else "WARN" if worst >= w else "PASS"
        cards[k1].add("dedup_cross", st, max_overlap=round(worst, 4), with_bin=who)

    contam = run_contamination(samples, a.evals_dir, tmpdir)
    for key, card in cards.items():
        info = (contam or {}).get(key)
        if not info:
            card.add("contamination", "WARN", reason="checker did not run"); continue
        st, gr, gated = contam_status(key, info["per_set"])
        card.add("contamination", st, rate_any=round(info["rate"], 4),
                 gated_rate=round(gr, 4), gated=gated,
                 per_set={k: round(v, 4) for k, v in info["per_set"].items()})

    overall = "PASS"
    print(f"\n{'='*72}\n{'dataset':<16}{'status':<7} key numbers")
    for key, c in cards.items():
        overall = sev_max(overall, c.status)
        i, t, q = c.checks["integrity"], c.checks.get("truncation", {}), c.checks.get("quality", {})
        print(f"{key:<16}{c.status:<7} docs={i.get('n_docs')} tok={i.get('n_tokens')} "
              f"eos={i.get('eos_frac')} trunc@max={t.get('frac_at_max')} "
              f"dup={c.checks.get('dedup_within',{}).get('dup_frac')} "
              f"xdup={c.checks.get('dedup_cross',{}).get('max_overlap')} "
              f"bpt={q.get('bytes_per_token')} de={c.checks.get('language',{}).get('de_frac')} "
              f"contam={c.checks.get('contamination',{}).get('gated_rate')}")
        for n, ch in c.checks.items():
            if isinstance(ch, dict) and ch.get("status") in ("WARN", "FAIL"):
                print(f"    {ch['status']} {n}: " + json.dumps({k: v for k, v in ch.items() if k != 'status'})[:200])
    print(f"{'='*72}\nOVERALL: {overall}")
    a.json_out.parent.mkdir(parents=True, exist_ok=True)
    a.json_out.write_text(json.dumps({"overall": overall, "seed": SEED, "generated": time.strftime("%F %T"),
        "cross_overlap": cross, "datasets": {k: {"status": c.status, "checks": c.checks} for k, c in cards.items()}}, indent=1))
    print(f"report -> {a.json_out}")
    return 1 if overall == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
