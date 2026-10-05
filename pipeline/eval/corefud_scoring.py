"""Coreference scoring with the official CorefUD scorer (CRAC shared tasks), at corpus level.

The scorer (https://github.com/ufal/corefud-scorer) compares a key and a response CoNLL-U
file that must be aligned token by token and differ only in their coreference annotation.
``write_response`` builds that response file from the key: a line-for-line copy in which
every gold coreference attribute is removed and the resolver's predicted clusters are
written in CorefUD 1.0 bracket notation. ``score_corpus`` runs the scorer once over the
whole corpus, so MUC/B3/CEAFe are micro-averaged over all documents as in the shared
tasks, instead of averaging per-document scores.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

from ..types import MentionSpan

SCORER = Path(os.environ.get("COREFUD_SCORER", "/opt/corefud-scorer/corefud-scorer.py"))
MATCH_MODES = ("exact", "partial", "head")
# coreference-related MISC attributes of the key that must not leak into the response
_COREF_MISC_PREFIXES = ("Entity=", "SplitAnte=", "Bridge=")
RESPONSE_ENTITY_SCHEMA = "# global.Entity = eid-etype-head-other"


def scorer_available() -> bool:
    return SCORER.is_file()


def conllu_text(text: str) -> str:
    """``text`` ending with exactly one blank line: CoNLL-U ends every sentence with one, and
    the scorer's reader (Udapi) fails on a file whose last sentence lacks it."""
    return text.rstrip("\n") + "\n\n"


def key_text_for_scorer(text: str) -> str:
    """Gold CoNLL-U as the scorer needs it: Udapi refuses Entity= annotations without a
    '# global.Entity' header (e.g. the bundled sample), so the standard CorefUD one is added."""
    if "Entity=" in text and "# global.Entity" not in text:
        text = RESPONSE_ENTITY_SCHEMA + "\n" + text
    return conllu_text(text)


def normalized_key(key_text: str, gold: dict[str, list[list[MentionSpan]]]) -> tuple[str, int]:
    """The key the scorer compares against: the gold clusters as the loader read them,
    rewritten exactly like a response (same Entity schema, same tree-based head rule).

    Corpora differ in how -- and whether -- their Entity= attributes name the mention head
    (GUM's schema has no head field, converted RuCoCo has no syntax at all). Head matching
    then compared the key's fallback heads with the response's tree-based ones: gold scored
    against itself got CoNLL 0.53 on GUM. Writing key and responses with one rule makes gold
    vs gold exactly 1.0 under every match mode. Returns (text, dropped gold mentions)."""
    return write_response(key_text_for_scorer(key_text), gold)


def _is_token_line(cols: list[str]) -> bool:
    return len(cols) == 10 and "-" not in cols[0]


def _document_ids(lines: list[str]) -> list[str]:
    """Document ids in file order, exactly as ``corefud_loader.parse_conllu`` assigns them."""
    ids: list[str] = []
    leading_has_tokens = False
    for line in lines:
        if line.startswith("# newdoc"):
            m = re.search(r"newdoc(?:\s+id\s*=\s*(\S+))?", line)
            ids.append(m.group(1) if m and m.group(1) else f"doc{len(ids) + 1}")
        elif not ids and _is_token_line(line.split("\t")):
            leading_has_tokens = True
    if leading_has_tokens:
        ids.insert(0, "doc1" if not ids else "doc0")
    return ids


def _mention_heads(token_rows: list[tuple[str, str]], span: MentionSpan) -> int:
    """1-based position of the mention's head word: the first word whose syntactic parent
    (CoNLL-U HEAD) lies outside the mention; 1 when the key has no dependency tree."""
    ids = [token_rows[i][0] for i in range(span.start_token, span.end_token + 1)]
    inside = set(ids)
    for pos, i in enumerate(range(span.start_token, span.end_token + 1), start=1):
        head = token_rows[i][1]
        if head not in ("_", "0") and head not in inside:
            return pos
        if head == "0":
            return pos
    return 1


def entity_brackets(clusters: list[list[MentionSpan]], token_rows: list[tuple[str, str]],
                    sent_of: list[int], eid_prefix: str = "c") -> tuple[dict[int, str], int]:
    """CorefUD 1.0 Entity= values per token index (attributes eid-etype-head-other) for
    ``clusters`` over tokens given as (CoNLL-U ID, HEAD) rows with their sentence numbers.
    Mentions crossing a sentence boundary are dropped (CorefUD mentions are
    sentence-internal) and counted."""
    opens: dict[int, list[tuple[int, str]]] = {}
    closes: dict[int, list[tuple[int, str]]] = {}
    singles: dict[int, list[str]] = {}
    dropped = 0
    for k, cluster in enumerate(clusters, start=1):
        eid = f"{eid_prefix}{k}"
        for span in sorted(set(cluster), key=lambda s: (s.start_token, s.end_token)):
            if sent_of[span.start_token] != sent_of[span.end_token]:
                dropped += 1
                continue
            head = _mention_heads(token_rows, span)
            if span.start_token == span.end_token:
                singles.setdefault(span.start_token, []).append(f"({eid}--{head})")
            else:
                opens.setdefault(span.start_token, []).append((span.end_token, f"({eid}--{head}"))
                closes.setdefault(span.end_token, []).append((span.start_token, f"{eid})"))
    values: dict[int, str] = {}
    for idx in set(opens) | set(closes) | set(singles):
        # closings first (innermost = latest start first), then openings (outermost = latest
        # end first), then one-word mentions, which nest inside anything opened here
        parts = [p for _, p in sorted(closes.get(idx, []), key=lambda t: -t[0])]
        parts += [p for _, p in sorted(opens.get(idx, []), key=lambda t: -t[0])]
        parts += singles.get(idx, [])
        values[idx] = "".join(parts)
    return values, dropped


def write_response(key_text: str, predicted: dict[str, list[list[MentionSpan]] | None]) -> tuple[str, int]:
    """Response CoNLL-U for ``predicted`` (doc_id -> clusters in the loader's token indices;
    missing or None = no predicted coreference). Returns (text, dropped mention count)."""
    lines = key_text.splitlines()
    doc_ids = _document_ids(lines)

    # token rows (sentence-local id, HEAD) and sentence numbers per document, in loader order
    rows: dict[str, list[tuple[str, str]]] = {d: [] for d in doc_ids}
    sents: dict[str, list[int]] = {d: [] for d in doc_ids}
    sent_no = 0
    # tokens before the first '# newdoc' form their own document (see _document_ids)
    has_leading_doc = len(doc_ids) > sum(line.startswith("# newdoc") for line in lines)
    current = doc_ids[0] if has_leading_doc else None
    newdoc_index = 1 if has_leading_doc else 0
    plan: list[tuple[str | None, int | None]] = []  # per line: (doc_id, token index) for token lines
    for line in lines:
        if line.startswith("# newdoc"):
            current = doc_ids[newdoc_index]
            newdoc_index += 1
            sent_no += 1
            plan.append((None, None))
            continue
        cols = line.split("\t")
        if current is not None and _is_token_line(cols):
            plan.append((current, len(rows[current])))
            rows[current].append((cols[0], cols[6]))
            sents[current].append(sent_no)
        else:
            if not line.strip():
                sent_no += 1
            plan.append((None, None))

    values: dict[str, dict[int, str]] = {}
    dropped = 0
    for d in doc_ids:
        clusters = predicted.get(d) or []
        values[d], n = entity_brackets(clusters, rows[d], sents[d])
        dropped += n

    out: list[str] = []
    for line, (d, idx) in zip(lines, plan):
        if line.startswith("# global.Entity"):
            out.append(RESPONSE_ENTITY_SCHEMA)
            continue
        cols = line.split("\t")
        if len(cols) == 10 and not line.startswith("#"):
            misc = [f for f in cols[9].split("|") if f != "_" and not f.startswith(_COREF_MISC_PREFIXES)]
            if d is not None and idx in values[d]:
                misc.append("Entity=" + values[d][idx])
            cols[9] = "|".join(misc) if misc else "_"
            out.append("\t".join(cols))
        else:
            out.append(line)
    text = conllu_text("\n".join(out))
    if RESPONSE_ENTITY_SCHEMA not in text:
        text = RESPONSE_ENTITY_SCHEMA + "\n" + text
    return text, dropped


_METRIC_NAMES = {"muc": "muc", "bcub": "b_cubed", "ceafe": "ceafe"}


def parse_scorer_output(output: str) -> dict:
    """MUC/B3/CEAFe precision/recall/F1 and CoNLL F1 (fractions 0..1) from scorer stdout."""
    results: dict = {}
    current = None
    for line in output.splitlines():
        name = line.strip().lower()
        if name in _METRIC_NAMES:
            current = _METRIC_NAMES[name]
            continue
        m = re.search(r"Recall:\s*([\d.]+)\s+Precision:\s*([\d.]+)\s+F1:\s*([\d.]+)", line)
        if m and current:
            r, p, f = (float(x) / 100 for x in m.groups())
            results[current] = {"precision": p, "recall": r, "f1": f}
            current = None
    missing = [m for m in _METRIC_NAMES.values() if m not in results]
    if missing:
        raise RuntimeError(f"CorefUD scorer output lacks {missing}:\n{output}")
    results["conll_f1"] = sum(results[m]["f1"] for m in _METRIC_NAMES.values()) / 3
    return results


def score_corpus(key_path: Path, response_path: Path, match: str = "exact") -> dict:
    """Run the official CorefUD scorer on the whole corpus (singletons ignored, as in CRAC)."""
    if match not in MATCH_MODES:
        raise ValueError(f"match must be one of {MATCH_MODES}, not {match!r}")
    if not scorer_available():
        raise RuntimeError(f"CorefUD scorer not found at {SCORER} (set COREFUD_SCORER)")
    proc = subprocess.run(
        [sys.executable, str(SCORER), str(key_path), str(response_path), "-a", match,
         "-m", "muc", "bcub", "ceafe"],
        capture_output=True, text=True, cwd=str(SCORER.parent),
    )
    if proc.returncode != 0:
        raise RuntimeError(f"CorefUD scorer failed ({proc.returncode}):\n{proc.stderr or proc.stdout}")
    results = parse_scorer_output(proc.stdout)
    results.update({"scorer": "corefud-scorer", "match": match, "scope": "corpus"})
    return results
