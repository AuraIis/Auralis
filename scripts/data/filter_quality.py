"""Apply lightweight quality filters to a text corpus file.

This is a practical local-quality pass for already-downloaded corpora.
It is intentionally simple and CPU-friendly:

- normalise encoding / whitespace (for non-code)
- reject very short / very long lines
- reject URL-dense / symbol-dense garbage
- reject obvious mojibake and boilerplate-heavy lines
- reject extremely repetitive lines

Optional PROSE-ONLY upgrades (all behind explicit flags, all hard-disabled for
--language code so code bytes are never touched):

- --lid-expect de|en   fastText language-ID gate (drops cross-language docs)
- --strip-pii          mask emails / IBANs with placeholder tokens
- --collapse-dup-paragraphs  drop exact duplicate >=60-char segments inside a doc
- boilerplate is density-based: 1 phrase in a long doc no longer kills the doc

The script is useful before tokenisation and before final mixing.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Let the script run via `python scripts/data/filter_quality.py` without an
# editable install: make the repo root importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.data._common import atomic_text_writer, clean_text, now_iso


BOILERPLATE_PATTERNS = (
    # English web chrome
    "cookie policy",
    "privacy policy",
    "accept all cookies",
    "subscribe to our newsletter",
    "all rights reserved",
    "sign in to continue",
    "javascript is disabled",
    # German web chrome — the corpus is German-heavy but the list used to be
    # English-only, so German cookie banners / shop / login boilerplate slipped
    # through unfiltered. Multi-word phrases only, to avoid nuking legit prose.
    "diese website verwendet cookies",
    "wir verwenden cookies",
    "cookies akzeptieren",
    "alle cookies akzeptieren",
    "alle rechte vorbehalten",
    "newsletter abonnieren",
    "javascript ist deaktiviert",
    "bitte aktivieren sie javascript",
    "in den warenkorb",
    "zur kasse",
    "passwort vergessen",
    "anmelden oder registrieren",
)


@dataclass
class FilterManifest:
    input_file: str
    output_file: str
    language: str
    preserve_newlines: bool
    started_at: str
    finished_at: str = ""
    lines_in: int = 0
    lines_written: int = 0
    bytes_written: int = 0
    dropped: dict[str, int] = field(default_factory=dict)
    repaired: dict[str, int] = field(default_factory=dict)
    flags: dict = field(default_factory=dict)


def _drop(manifest: FilterManifest, reason: str) -> None:
    manifest.dropped[reason] = manifest.dropped.get(reason, 0) + 1


def _url_density(line: str) -> float:
    tokens = line.split()
    if not tokens:
        return 0.0
    url_like = sum(1 for token in tokens if token.startswith(("http://", "https://", "www.")))
    return url_like / len(tokens)


def _symbol_density(line: str) -> float:
    if not line:
        return 0.0
    symbol_like = sum(1 for ch in line if not ch.isalnum() and not ch.isspace())
    return symbol_like / len(line)


def _repetition_score(line: str) -> float:
    tokens = line.split()
    if not tokens:
        return 0.0
    unique = len(set(tokens))
    return 1.0 - (unique / len(tokens))


MOJIBAKE_MARKERS = (
    # UTF-8 read as cp1252/latin-1 (German umlauts, punctuation) + replacement char
    "â€™", "â€œ", "â€", "Ã¼", "Ã¶", "Ã¤", "Ã„", "Ã–", "Ãœ", "ÃŸ", "Ã©", "Ã¨",
    "Â»", "Â«", "Â ", "�",
)


def _looks_mojibake(line: str) -> bool:
    return any(marker in line for marker in MOJIBAKE_MARKERS)


def _normalise(line: str, preserve_newlines: bool) -> str:
    if preserve_newlines:
        # CODE path: byte-conservative. Only NUL removal and CRLF -> LF.
        # (The old version replaced \r with a SPACE, which appended a trailing
        # space to every CRLF line — pointless byte churn in code.)
        return line.replace("\x00", "").rstrip("\r\n")
    line = line.replace("\x00", "").replace("\r", " ")
    return clean_text(line)


# --- prose-only helpers (never applied when --language code) -----------------

EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}(?:\s?\d{4}){4,7}(?:\s?\d{1,3})?\b")
HTML_TAG_RE = re.compile(r"</?(?:div|span|p|a|li|ul|ol|td|tr|table|img|br|h[1-6]|html|body)\b[^>]*>", re.I)
SENT_SPLIT_RE = re.compile(r"(?<=[.!?:])\s+")


def _strip_pii(text: str) -> str:
    text = EMAIL_RE.sub("<email>", text)
    return IBAN_RE.sub("<iban>", text)


def _boilerplate_hits(lower: str) -> int:
    return sum(1 for p in BOILERPLATE_PATTERNS if p in lower)


def _strip_boilerplate_sentences(text: str) -> str:
    """Remove only the sentences that contain a boilerplate phrase (footer/cookie
    chrome) instead of nuking a whole multi-paragraph doc for one footer line."""
    parts = SENT_SPLIT_RE.split(text)
    kept = [s for s in parts if not _boilerplate_hits(s.lower())]
    return " ".join(kept)


def _collapse_dup_sentences(text: str, min_len: int = 60) -> str:
    """Drop exact duplicate long sentences inside one doc (nav menus, repeated
    teasers). Order-preserving, first occurrence wins. Prose only."""
    parts = SENT_SPLIT_RE.split(text)
    seen: set[str] = set()
    out = []
    for s in parts:
        key = s.strip().lower()
        if len(key) >= min_len:
            if key in seen:
                continue
            seen.add(key)
        out.append(s)
    return " ".join(out)


class LangID:
    """Thin fastText LID wrapper (lid.176). Uses the low-level predict to avoid
    the numpy>=2 copy bug in the fasttext python wrapper."""

    def __init__(self, model_path: str):
        import fasttext  # local import: only needed when --lid-expect is used
        self.model = fasttext.load_model(model_path)

    def classify(self, text: str) -> tuple[str, float]:
        sample = " ".join(text[:1200].split())
        if not sample:
            return "??", 0.0
        labels = self.model.f.predict(sample + "\n", 1, 0.0, "strict")
        if not labels:
            return "??", 0.0
        prob, label = labels[0]
        return label.replace("__label__", ""), min(prob, 1.0)


# Markers we never drop, regardless of length — they are file/code boundaries
# that the tokenizer/model needs to keep intact. Adding to this list is cheap.
PROTECTED_PREFIXES = (
    "<|code|>",
    "<|endcode|>",
    "<filename>",
    "<file_sep>",
    "<|file|>",
)


def _passes(
    line: str,
    *,
    min_length: int,
    max_length: int,
    preserve_newlines: bool,
    allow_mojibake: bool,
    max_repetition: float = 0.60,
    lid: "LangID | None" = None,
    lid_expect: str = "",
    lid_conf: float = 0.65,
    strict_boilerplate: bool = True,
) -> str | None:
    normalized = _normalise(line, preserve_newlines=preserve_newlines)

    # Boundary markers: only check too_long, otherwise let through.
    if any(normalized.startswith(p) for p in PROTECTED_PREFIXES):
        if len(normalized) > max_length:
            return "too_long"
        return None

    if len(normalized) < min_length:
        return "too_short"
    if len(normalized) > max_length:
        return "too_long"
    lower = normalized.lower()
    hits = _boilerplate_hits(lower)
    if hits:
        # Density-based: one footer phrase in a long doc is fixable by sentence
        # stripping (caller handles it); only drop when chrome dominates.
        if strict_boilerplate or hits >= 3 or len(normalized) < 600:
            return "boilerplate"
    if _url_density(normalized) > 0.20:
        return "url_dense"
    if _symbol_density(normalized) > 0.35:
        return "symbol_dense"
    if _repetition_score(normalized) > max_repetition:
        return "repetitive"
    if not allow_mojibake and _looks_mojibake(normalized):
        return "mojibake"
    if lid is not None and not preserve_newlines and len(normalized) >= 200:
        lang, conf = lid.classify(normalized)
        if lang != lid_expect and conf >= lid_conf:
            return f"lang_{lang}"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--language", choices=["english", "german", "code"], required=True)
    parser.add_argument("--min-length", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--allow-mojibake", action="store_true")
    parser.add_argument("--max-repetition", type=float, default=0.60,
                        help="Reject lines where this fraction of tokens are duplicates. "
                             "Default 0.60 (strict). Math/code with formal-repetitive content "
                             "benefits from 0.80-0.85.")
    # --- prose-only upgrades (explicit flags; hard-disabled for --language code) ---
    parser.add_argument("--lid-expect", choices=["de", "en"], default=None,
                        help="fastText language-ID gate: drop docs confidently identified "
                             "as a different language. Prose only.")
    parser.add_argument("--lid-model", default="/workspace/v2data/models/lid.176.ftz",
                        help="Path to fastText lid.176 model (used with --lid-expect).")
    parser.add_argument("--lid-conf", type=float, default=0.65,
                        help="Min LID confidence to drop a wrong-language doc (default 0.65).")
    parser.add_argument("--strip-pii", action="store_true",
                        help="Mask emails/IBANs with <email>/<iban>. Prose only.")
    parser.add_argument("--strip-boilerplate", action="store_true",
                        help="Remove boilerplate SENTENCES from long docs instead of dropping "
                             "the whole doc; doc still dropped if chrome dominates. Prose only.")
    parser.add_argument("--collapse-dup-paragraphs", action="store_true",
                        help="Collapse exact duplicate >=60-char sentences inside a doc "
                             "(nav menus / repeated teasers). Prose only.")
    args = parser.parse_args()

    if args.language == "code" and (args.lid_expect or args.strip_pii
                                    or args.strip_boilerplate or args.collapse_dup_paragraphs):
        parser.error("--lid-expect/--strip-pii/--strip-boilerplate/--collapse-dup-paragraphs "
                     "are prose-only; refusing to run them on --language code "
                     "(code bytes must never be altered).")

    # Per-language length defaults: (min, max, preserve_newlines).
    # Code: lowered min from 50 to 10 so imports / short statements / closing
    # braces stay in. Pre-fix the filter wiped 78.8 percent of starcoderdata
    # because a typical code line is much shorter than 50 chars.
    defaults = {
        "english": (200, 100_000, False),
        "german": (300, 100_000, False),
        "code": (10, 30_000, True),
    }
    min_length, max_length, preserve_newlines = defaults[args.language]
    if args.min_length is not None:
        min_length = args.min_length
    if args.max_length is not None:
        max_length = args.max_length

    manifest = FilterManifest(
        input_file=str(args.input),
        output_file=str(args.output),
        language=args.language,
        preserve_newlines=preserve_newlines,
        started_at=now_iso(),
    )
    manifest.flags = {
        "lid_expect": args.lid_expect, "lid_conf": args.lid_conf,
        "strip_pii": args.strip_pii, "strip_boilerplate": args.strip_boilerplate,
        "collapse_dup_paragraphs": args.collapse_dup_paragraphs,
    }

    lid = LangID(args.lid_model) if args.lid_expect else None

    with atomic_text_writer(args.output) as out_fh, args.input.open(
        "r", encoding="utf-8", errors="replace"
    ) as in_fh:
        for line in in_fh:
            manifest.lines_in += 1
            reason = _passes(
                line,
                min_length=min_length,
                max_length=max_length,
                preserve_newlines=preserve_newlines,
                allow_mojibake=args.allow_mojibake,
                max_repetition=args.max_repetition,
                lid=lid,
                lid_expect=args.lid_expect or "",
                lid_conf=args.lid_conf,
                strict_boilerplate=not args.strip_boilerplate,
            )
            if reason is not None:
                _drop(manifest, reason)
                continue
            normalized = _normalise(line, preserve_newlines=preserve_newlines)
            # prose-only repairs (parser guarantees these are off for code)
            if args.strip_boilerplate and _boilerplate_hits(normalized.lower()):
                stripped = _strip_boilerplate_sentences(normalized)
                if len(stripped) < max(min_length, int(0.7 * len(normalized))):
                    _drop(manifest, "boilerplate")   # chrome dominated the doc
                    continue
                normalized = stripped
                manifest.repaired["boilerplate_stripped"] = \
                    manifest.repaired.get("boilerplate_stripped", 0) + 1
            if args.collapse_dup_paragraphs:
                collapsed = _collapse_dup_sentences(normalized)
                if len(collapsed) != len(normalized):
                    manifest.repaired["dup_sentences_collapsed"] = \
                        manifest.repaired.get("dup_sentences_collapsed", 0) + 1
                    normalized = collapsed
                if len(normalized) < min_length:
                    _drop(manifest, "too_short")
                    continue
            if args.strip_pii:
                masked = _strip_pii(normalized)
                if masked != normalized:
                    manifest.repaired["pii_masked"] = manifest.repaired.get("pii_masked", 0) + 1
                    normalized = masked
            out_fh.write(normalized + "\n")
            manifest.lines_written += 1
            manifest.bytes_written += len((normalized + "\n").encode("utf-8"))

    manifest.finished_at = now_iso()
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(asdict(manifest), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
