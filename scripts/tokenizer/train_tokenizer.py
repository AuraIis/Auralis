"""Train the Helix v2 SentencePiece tokenizer.

Reads the mixed corpus produced by ``prepare_corpus.py``, trains a 200 k-vocab
Unigram model per ``configs/tokenizer/helix_v2.yaml``, and writes:

- ``<out>/helix_v2_tokenizer.model``    — SentencePiece model
- ``<out>/helix_v2_tokenizer.vocab``    — text vocab for inspection
- ``<out>/training_manifest.yaml``      — full run record (config, corpus path,
  git SHA, total input bytes, effective sentence count, timing)

On a workstation this typically takes 1-3 hours for 200 k vocab / 15 GB input.
It will spill RAM; set ``--num-threads`` and be patient rather than bumping
``input_sentence_size``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import sentencepiece as spm
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CFG = REPO_ROOT / "configs" / "tokenizer" / "helix_v2.yaml"


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _user_defined_symbols(cfg: dict) -> list[str]:
    """Special tokens after the first 4 (pad/unk/s/eos), which SP assigns itself."""
    return list(cfg["special_tokens"][4:])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CFG)
    parser.add_argument("--corpus", type=Path, required=True,
                        help="Path to the mixed tokenizer training corpus (from prepare_corpus.py)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Where to write helix_v2_tokenizer.{model,vocab} and manifest")
    parser.add_argument("--num-threads", type=int, default=0,
                        help="0 = auto (os.cpu_count()). SentencePiece requires 1-1024.")
    parser.add_argument("--input-sentence-size", type=int, default=None,
                        help="Override cap on sentences fed to EM training.")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not args.corpus.exists():
        sys.exit(f"corpus not found: {args.corpus}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_prefix = args.output_dir / "helix_v2_tokenizer"

    # SentencePiece reserves ids 0-3 for the first four specials;
    # the remainder go into user_defined_symbols.
    training = cfg["training"]
    input_sentence_size = args.input_sentence_size or int(training["input_sentence_size"])

    sp_args = {
        "input": str(args.corpus),
        "model_prefix": str(model_prefix),
        "vocab_size": int(cfg["vocab_size"]),
        "model_type": training["model_type"],
        "character_coverage": float(cfg["character_coverage"]),
        "pad_id": 0, "unk_id": 1, "bos_id": 2, "eos_id": 3,
        "pad_piece": cfg["special_tokens"][0],
        "unk_piece": cfg["special_tokens"][1],
        "bos_piece": cfg["special_tokens"][2],
        "eos_piece": cfg["special_tokens"][3],
        "user_defined_symbols": _user_defined_symbols(cfg),
        "input_sentence_size": input_sentence_size,
        "shuffle_input_sentence": bool(training["shuffle_input_sentence"]),
        "num_threads": args.num_threads or max(1, (os.cpu_count() or 1)),
        "max_sentence_length": int(training["max_sentence_length"]),
        "normalization_rule_name": training["normalization_rule_name"],
        "remove_extra_whitespaces": bool(training["remove_extra_whitespaces"]),
        "byte_fallback": bool(training["byte_fallback"]),
        "train_extremely_large_corpus": True,
    }

    print(f"Training tokenizer ({cfg['algorithm']}, vocab={cfg['vocab_size']:,})")
    print(f"Corpus: {args.corpus} ({args.corpus.stat().st_size/1e9:.2f} GB)")
    print(f"Output: {model_prefix}.*\n")

    t0 = time.time()
    spm.SentencePieceTrainer.train(**sp_args)
    elapsed = time.time() - t0

    # Sanity-load and report first/last tokens.
    sp = spm.SentencePieceProcessor(model_file=str(model_prefix) + ".model")
    n_pieces = sp.GetPieceSize()

    manifest = {
        "name": cfg["name"],
        "version": cfg["version"],
        "algorithm": cfg["algorithm"],
        "vocab_size_requested": int(cfg["vocab_size"]),
        "vocab_size_actual": n_pieces,
        "corpus_file": str(args.corpus),
        "corpus_bytes": args.corpus.stat().st_size,
        "config_file": str(args.config),
        "config_sha": _sha256_file(args.config),
        "git_sha": _git_sha(),
        "sp_args": {k: v for k, v in sp_args.items() if k != "user_defined_symbols"},
        "num_user_defined_symbols": len(sp_args["user_defined_symbols"]),
        "training_seconds": round(elapsed, 1),
        "first_10_pieces": [sp.IdToPiece(i) for i in range(10)],
        "last_10_pieces": [sp.IdToPiece(n_pieces - 10 + i) for i in range(10)],
    }
    manifest_path = args.output_dir / "training_manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    print(f"\nDone in {elapsed/60:.1f} min. vocab={n_pieces:,}")
    print(f"Model : {model_prefix}.model")
    print(f"Vocab : {model_prefix}.vocab")
    print(f"Manifest: {manifest_path}")


def _sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    main()
