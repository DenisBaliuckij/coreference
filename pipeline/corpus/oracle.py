from __future__ import annotations

from ..types import CorefDocument, MentionSpan


def _span_text(doc: CorefDocument, span: MentionSpan) -> str:
    start_char = doc.tokens[span.start_token].start_char
    end_char = doc.tokens[span.end_token].end_char
    return doc.text[start_char:end_char]


def build_oracle_text(doc: CorefDocument) -> str:
    """Replace every non-head mention in each gold cluster with the head's
    surface text. Head = the cluster's first mention in document order."""
    replacements: list[tuple[int, int, str]] = []

    for cluster in doc.clusters:
        if len(cluster) < 2:
            continue
        head_text = _span_text(doc, cluster[0])
        for mention in cluster[1:]:
            if mention.is_zero:
                continue
            start_char = doc.tokens[mention.start_token].start_char
            end_char = doc.tokens[mention.end_token].end_char
            replacements.append((start_char, end_char, head_text))

    replacements.sort(key=lambda r: r[0], reverse=True)
    out = doc.text
    for start, end, replacement in replacements:
        out = out[:start] + replacement + out[end:]
    return out
