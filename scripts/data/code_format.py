#!/usr/bin/env python3
"""Shared code-formatting transforms.

The pretrain code corpus is tokenized with tab-indentation (4 leading spaces -> \t),
so EVERY consumer that tokenizes code into prompts/targets (SFT data builders,
code-ppl eval, future pass@k prompts) must apply the SAME transform, otherwise the
model sees an indent distribution it never trained on. Import from here — don't
duplicate the logic."""

TAB_WIDTH = 4  # PEP8 indent unit; starcoder/OpenCoder python is overwhelmingly 4-space


def tab_indent(text: str, tab_width: int = TAB_WIDTH) -> str:
    """Convert LEADING space-runs to tabs (CODE only). The helix tokenizer has no
    whitespace-run pieces, so '    ' costs 4 tokens while '\t' costs 1 —
    measured ~17% fewer tokens on real Python. Only leading indentation is
    touched; interior spacing/operators/strings on the line are untouched.
    KNOWN LIMITATION: leading spaces inside triple-quoted strings are also
    converted (a line scanner can't see string state). For training data this
    only changes docstring indentation chars, which is acceptable; do NOT use
    this on text whose exact bytes must round-trip."""
    out = []
    for line in text.split("\n"):
        n = len(line) - len(line.lstrip(" "))
        if n >= tab_width:
            tabs, rem = divmod(n, tab_width)
            line = "\t" * tabs + " " * rem + line[n:]
        out.append(line)
    return "\n".join(out)


def tab_indent_fenced(text: str, tab_width: int = TAB_WIDTH) -> str:
    """Apply tab_indent ONLY inside ``` fenced code blocks; prose lines are
    untouched. Use for mixed chat/SFT texts (German prose + code blocks)."""
    out, in_fence = [], False
    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence:
            n = len(line) - len(line.lstrip(" "))
            if n >= tab_width:
                tabs, rem = divmod(n, tab_width)
                line = "\t" * tabs + " " * rem + line[n:]
        out.append(line)
    return "\n".join(out)
