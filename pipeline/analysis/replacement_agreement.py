"""What each resolver wrote where the oracle replaced a single word (a pronoun).

    python -m pipeline.analysis.replacement_agreement RUN_DIR/artifacts/texts LLMv2,LapinLiass

Aligns every text with the original (NoResolution) by word-level difflib and, for each oracle
edit that replaces exactly one original word, classifies the resolver's edit at the same span:
the oracle's string (ignoring case and possessive 's), a different string sharing a content
word with it (typically another name for the same referent), a different string sharing none
(typically another referent), or no edit at that span (pronoun kept, or the resolver rewrote
the surrounding words so the spans do not align). Word overlap is a proxy: it does not decide
whether two strings denote the same entity.
"""

from __future__ import annotations

import argparse
import difflib
from collections import Counter
from pathlib import Path

STOP = {"the", "and", "of", "a", "an", "that", "this", "these", "those",
        "его", "её", "их", "который", "это", "и", "в"}
CATEGORIES = ("same_as_oracle", "different_string_sharing_a_word", "different_string_no_shared_word",
              "no_edit_at_span")


def _edits(original: list[str], text: list[str]) -> dict[tuple[int, int], str]:
    sm = difflib.SequenceMatcher(None, original, text, autojunk=False)
    return {(i1, i2): " ".join(text[j1:j2]) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal"}


def _norm(s: str) -> str:
    return s.lower().replace(" 's", "").replace("'s", "")


def _words(s: str) -> set[str]:
    return {w for w in _norm(s).split() if len(w) > 2 and w not in STOP}


def classify(oracle_target: str, resolver_text: str | None) -> str:
    if resolver_text is None:
        return "no_edit_at_span"
    if _norm(resolver_text) == _norm(oracle_target):
        return "same_as_oracle"
    return "different_string_sharing_a_word" if _words(resolver_text) & _words(oracle_target) \
        else "different_string_no_shared_word"


def agreement(texts_dir: Path, resolvers: list[str]) -> dict:
    counts = {r: Counter() for r in resolvers}
    total = 0
    for p in sorted((texts_dir / "NoResolution").glob("*.txt")):
        original = p.read_text(encoding="utf-8").split()
        oracle = _edits(original, (texts_dir / "ORACLE" / p.name).read_text(encoding="utf-8").split())
        edits = {r: _edits(original, (texts_dir / r / p.name).read_text(encoding="utf-8").split()) for r in resolvers}
        for span, target in oracle.items():
            if span[1] - span[0] != 1:
                continue
            total += 1
            for r in resolvers:
                counts[r][classify(target, edits[r].get(span))] += 1
    return {"oracle_single_word_replacements": total,
            "resolvers": {r: {c: counts[r][c] for c in CATEGORIES} for r in resolvers}}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("texts_dir", type=Path)
    parser.add_argument("resolvers")
    args = parser.parse_args(argv)
    result = agreement(args.texts_dir, args.resolvers.split(","))
    n = result["oracle_single_word_replacements"]
    print(f"oracle single-word replacements: {n}")
    for r, c in result["resolvers"].items():
        print(r, ", ".join(f"{k} {v} ({v / max(1, n):.0%})" for k, v in c.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
