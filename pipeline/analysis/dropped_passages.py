"""How much of the original text a resolver drops in long passages.

    python -m pipeline.analysis.dropped_passages RUN_DIR/artifacts/texts LLMv2,LapinLiass [--min-words 10]

Aligns each resolver's text with the original (NoResolution) by word-level difflib and counts
the net words lost in every delete/replace edit that shortens the text by at least
``--min-words`` words: whole sentences or clauses that a rewriting resolver left out, as opposed
to a pronoun replaced by a longer or shorter name.
"""

from __future__ import annotations

import argparse
import difflib
from pathlib import Path


def dropped_words(original: list[str], text: list[str], min_words: int = 10) -> int:
    sm = difflib.SequenceMatcher(None, original, text, autojunk=False)
    lost = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        net = (i2 - i1) - (j2 - j1)
        if tag in ("delete", "replace") and net >= min_words:
            lost += net
    return lost


def dropped_passages(texts_dir: Path, resolvers: list[str], min_words: int = 10) -> dict:
    out = {}
    for r in resolvers:
        words = lost = affected = docs = 0
        for p in sorted((texts_dir / "NoResolution").glob("*.txt")):
            original = p.read_text(encoding="utf-8").split()
            d = dropped_words(original, (texts_dir / r / p.name).read_text(encoding="utf-8").split(), min_words)
            words += len(original); lost += d; docs += 1; affected += d > 0
        out[r] = {"words": words, "words_lost": lost, "share_lost": lost / max(1, words),
                  "documents": docs, "documents_affected": affected}
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("texts_dir", type=Path)
    parser.add_argument("resolvers")
    parser.add_argument("--min-words", type=int, default=10)
    args = parser.parse_args(argv)
    for r, s in dropped_passages(args.texts_dir, args.resolvers.split(","), args.min_words).items():
        print(f"{r}: {s['words_lost']}/{s['words']} words lost ({s['share_lost']:.1%}) in "
              f"{s['documents_affected']}/{s['documents']} documents")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
