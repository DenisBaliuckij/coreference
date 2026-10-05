from __future__ import annotations

from ..types import CorefDocument, MentionSpan
from .pronouns import PRONOUNS

ORACLE_MODES = ("all", "pronouns")


def _span_text(doc: CorefDocument, span: MentionSpan) -> str:
    start_char = doc.tokens[span.start_token].start_char
    end_char = doc.tokens[span.end_token].end_char
    return doc.text[start_char:end_char]


def _is_pronoun(doc: CorefDocument, span: MentionSpan, pronouns: set[str]) -> bool:
    words = [t.text.lower() for t in doc.tokens[span.start_token: span.end_token + 1] if t.text]
    return len(words) == 1 and words[0] in pronouns


def build_oracle_text(doc: CorefDocument, mode: str = "all", language: str = "en") -> str:
    """Replace non-head mentions of each gold cluster with the head's surface text.
    Head = the cluster's first mention in document order.

    mode "all": every non-head mention (nominal ones too, including predicate nominals that
    a corpus like GUM annotates as coreferent); mode "pronouns": only single-word pronominal
    mentions (pipeline.corpus.pronouns), i.e. what a pronoun resolver is asked to do.

    Mentions nest ("his" inside "his native Bohemia"). A replaced span replaces everything in
    it, so a replacement nested in (or crossing) one that starts earlier is dropped; applying
    both used to shift the outer span's end offset and garble the text."""
    if mode not in ORACLE_MODES:
        raise ValueError(f"unknown oracle mode {mode!r}; expected one of {ORACLE_MODES}")
    pronouns = PRONOUNS.get(language, set())
    candidates: list[tuple[int, int, str]] = []
    for cluster in doc.clusters:
        if len(cluster) < 2:
            continue
        head_text = _span_text(doc, cluster[0])
        for mention in cluster[1:]:
            if mention.is_zero:
                continue
            if mode == "pronouns" and not _is_pronoun(doc, mention, pronouns):
                continue
            start_char = doc.tokens[mention.start_token].start_char
            end_char = doc.tokens[mention.end_token].end_char
            candidates.append((start_char, end_char, head_text))

    # outermost first: earlier start, then longer span
    candidates.sort(key=lambda r: (r[0], -r[1]))
    kept: list[tuple[int, int, str]] = []
    for start, end, text in candidates:
        if kept and start < kept[-1][1]:
            continue  # nested in / crossing the previous kept replacement
        kept.append((start, end, text))

    out = doc.text
    for start, end, replacement in reversed(kept):
        out = out[:start] + replacement + out[end:]
    return out
