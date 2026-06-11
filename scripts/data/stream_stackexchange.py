#!/usr/bin/env python3
"""Stream common-pile/stackexchange, keep only target sites, write JSONL ({"text": doc}).

The user's gap is OS/sysadmin troubleshooting knowledge ("when a problem appears he can
fix it / install what he needs") plus code-Q&A-with-explanation. common-pile/stackexchange
(33.4M docs, CC BY-SA, all SE sites) has exactly that — but mixed and ordered ALPHABETICALLY
by site, so target sites are scattered across the whole stream; we iterate all and write only
the targets.

Output is JSONL with the FULL document (newlines preserved) -> tokenize_anneal.py gives one
EOS per Q&A doc (NOT the phase-1 line-per-doc atomization bug).

Per-site token caps keep the giant (stackoverflow) from dominating while taking the small
OS sites whole. APPROX token budget via bytes/--bytes-per-token (English ~4.5).

    python scripts/data/stream_stackexchange.py --output data/fresh/stackexchange_os.jsonl
"""
from __future__ import annotations
import argparse, ast, json, time
from pathlib import Path

# site -> token cap (None = take all). The OS/sysadmin sites are the priority (small, uncapped);
# stackoverflow is capped (huge, and we already have starcoder code).
DEFAULT_TARGETS = {
    # OS/sysadmin troubleshooting = the valuable niche (the user's gap); modest caps
    "askubuntu.com": 500_000_000,
    "superuser.com": 400_000_000,
    "serverfault.com": 300_000_000,
    "unix.stackexchange.com": 400_000_000,
    # problem-solving / design (small)
    "softwareengineering.stackexchange.com": 150_000_000,
    "codereview.stackexchange.com": 100_000_000,
    "dba.stackexchange.com": 100_000_000,
    "devops.stackexchange.com": 80_000_000,
    "security.stackexchange.com": 120_000_000,
    # StackOverflow = code-Q&A, overlaps with the real code corpus -> light supplement only
    "stackoverflow.com": 500_000_000,
}


def site_of(md) -> str | None:
    if isinstance(md, dict):
        return md.get("site")
    if isinstance(md, str):
        try:
            return ast.literal_eval(md).get("site")
        except Exception:
            return None
    return None


def count_lines(p: Path) -> int:
    if not p.exists():
        return 0
    n = 0
    with p.open("rb") as f:
        for _ in f:
            n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--repo", default="common-pile/stackexchange")
    ap.add_argument("--bytes-per-token", type=float, default=4.5)
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--skip-meta", action="store_true", default=True, help="skip *.meta.* sites")
    ap.add_argument("--progress-every", type=int, default=200000)
    a = ap.parse_args()

    from datasets import load_dataset

    a.output.parent.mkdir(parents=True, exist_ok=True)
    already = count_lines(a.output)
    print(f"[se] resume: {already} docs already written", flush=True)

    ds = load_dataset(a.repo, split="train", streaming=True)
    if already:
        ds = ds.skip(already)

    caps = dict(DEFAULT_TARGETS)
    site_bytes: dict[str, int] = {s: 0 for s in caps}
    site_docs: dict[str, int] = {s: 0 for s in caps}
    seen = already
    written = already
    t0 = time.monotonic()

    with a.output.open("a" if already else "w", encoding="utf-8") as out:
        for ex in ds:
            seen += 1
            site = site_of(ex.get("metadata"))
            if site not in caps:
                continue
            if a.skip_meta and ".meta." in (site or ""):
                continue
            cap = caps[site]
            if cap is not None and site_bytes[site] >= cap * a.bytes_per_token:
                continue
            txt = ex.get("text") or ""
            if len(txt) < a.min_chars:
                continue
            out.write(json.dumps({"text": txt, "site": site}, ensure_ascii=False) + "\n")
            b = len(txt.encode("utf-8")) + 1
            site_bytes[site] += b
            site_docs[site] += 1
            written += 1
            if written % a.progress_every == 0:
                out.flush()
                tot_tok = sum(site_bytes.values()) / a.bytes_per_token
                rate = seen / max(1e-9, time.monotonic() - t0)
                print(f"  seen {seen:,} | written {written:,} | ~{tot_tok/1e9:.2f}B tok | {rate:.0f} doc/s", flush=True)
            # stop early if every capped site is full and we've passed the alphabet tail
            if all(caps[s] is not None and site_bytes[s] >= caps[s] * a.bytes_per_token
                   for s in caps if caps[s] is not None) and seen > 30_000_000:
                break

    summary = {s: {"docs": site_docs[s], "approx_tokens": round(site_bytes[s] / a.bytes_per_token)}
               for s in caps if site_docs[s]}
    tot = sum(site_bytes.values()) / a.bytes_per_token
    a.output.with_suffix(a.output.suffix + ".summary.json").write_text(
        json.dumps({"seen": seen, "written": written, "approx_tokens_total": round(tot),
                    "per_site": summary}, indent=2, ensure_ascii=False))
    print(f"[se] DONE seen {seen:,} written {written:,} ~{tot/1e9:.2f}B tok", flush=True)
    for s, d in summary.items():
        print(f"   {s}: {d['docs']:,} docs, ~{d['approx_tokens']/1e9:.2f}B tok", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
