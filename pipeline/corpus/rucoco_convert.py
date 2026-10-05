"""Convert RuCoCo (Russian Coreference Corpus) JSON documents to CorefUD CoNLL-U.

RuCoCo (https://github.com/vdobrovolskii/rucoco, release v1.0.0) stores each document as
JSON: ``text``, ``entities`` (one list of [start, end) character spans per entity) and
``includes`` (split antecedents: per entity, the entities it includes). The stand and the
official CorefUD scorer need CoNLL-U with ``Entity=`` brackets, so each document is
tokenised with spaCy's rule-based Russian tokenizer (no model download) and split into
sentences by the rule-based sentencizer plus paragraph breaks -- never inside a mention,
since CorefUD mentions are sentence-internal. Character spans are mapped onto the tokens
they overlap.

The output has no syntax: HEAD is 0 and mention heads are the first word, so score it with
``--coref-match exact``. Split antecedents are counted but not written: the CorefUD scorer
does not evaluate them.

    python -m pipeline.corpus.rucoco_convert INPUT_DIR OUTPUT_DIR
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..eval.corefud_scoring import entity_brackets
from ..types import MentionSpan

ENTITY_SCHEMA = "# global.Entity = eid-etype-head-other"


def _nlp():
    import spacy

    nlp = spacy.blank("ru")
    nlp.add_pipe("sentencizer")
    return nlp


def convert_document(doc_id: str, data: dict, nlp) -> tuple[str, dict]:
    """CoNLL-U text for one RuCoCo document, plus conversion statistics."""
    text = data["text"]
    doc = nlp(text)
    words = [t for t in doc if not t.is_space]
    stats = {"tokens": len(words), "entities": 0, "mentions": 0, "unmappable_mentions": 0,
             "split_antecedents": sum(1 for inc in data.get("includes", []) if inc)}

    # character spans -> word index spans
    clusters: list[list[MentionSpan]] = []
    for spans in data.get("entities", []):
        mentions = []
        for start, end in spans:
            covered = [i for i, w in enumerate(words) if w.idx < end and w.idx + len(w.text) > start]
            if not covered:
                stats["unmappable_mentions"] += 1
                continue
            mentions.append(MentionSpan(covered[0], covered[-1]))
        if mentions:
            clusters.append(mentions)
            stats["entities"] += 1
            stats["mentions"] += len(mentions)

    # sentence ends: sentencizer ends and paragraph breaks, except inside a mention
    word_pos = {w.i: n for n, w in enumerate(words)}
    ends = set()
    for sent in doc.sents:
        inside = [word_pos[t.i] for t in sent if t.i in word_pos]
        if inside:
            ends.add(inside[-1])
    for t in doc:
        if t.is_space and "\n" in t.text:
            before = [word_pos[x.i] for x in doc[: t.i] if x.i in word_pos]
            if before:
                ends.add(before[-1])
    for cluster in clusters:
        for m in cluster:
            ends -= set(range(m.start_token, m.end_token))
    ends.add(len(words) - 1)

    sent_of, sent_no = [], 0
    for i in range(len(words)):
        sent_of.append(sent_no)
        if i in ends:
            sent_no += 1

    ids, local = [], 0
    for i in range(len(words)):
        local = 1 if i == 0 or sent_of[i] != sent_of[i - 1] else local + 1
        ids.append(str(local))
    brackets, dropped = entity_brackets(clusters, [(x, "0") for x in ids], sent_of, eid_prefix="e")
    assert dropped == 0  # sentence ends are never placed inside a mention

    out = [ENTITY_SCHEMA, f"# newdoc id = {doc_id}"]
    for s in range(sent_no):
        members = [i for i in range(len(words)) if sent_of[i] == s]
        if not members:
            continue
        first, last = words[members[0]], words[members[-1]]
        out.append(f"# sent_id = {doc_id}-{s + 1}")
        out.append("# text = " + text[first.idx: last.idx + len(last.text)].replace("\n", " "))
        for i in members:
            w = words[i]
            misc = []
            if i in brackets:
                misc.append("Entity=" + brackets[i])
            if not w.whitespace_ and i != members[-1]:
                misc.append("SpaceAfter=No")
            out.append("\t".join([ids[i], w.text, "_", "_", "_", "_", "0", "_", "_", "|".join(misc) or "_"]))
        out.append("")
    return "\n".join(out) + "\n", stats


def convert_corpus(input_dir: Path, output_dir: Path) -> dict:
    nlp = _nlp()
    output_dir.mkdir(parents=True, exist_ok=True)
    totals: dict = {"documents": 0}
    for path in sorted(input_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        conllu, stats = convert_document(path.stem, data, nlp)
        (output_dir / f"{path.stem}.conllu").write_text(conllu, encoding="utf-8")
        totals["documents"] += 1
        for k, v in stats.items():
            totals[k] = totals.get(k, 0) + v
    (output_dir / "conversion_stats.json").write_text(json.dumps(totals, indent=1), encoding="utf-8")
    return totals


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RuCoCo JSON -> CorefUD CoNLL-U")
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(convert_corpus(args.input_dir, args.output_dir), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
