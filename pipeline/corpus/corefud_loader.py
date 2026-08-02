from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from ..types import CorefDocument, MentionSpan, Token

_ENTITY_PIECE_RE = re.compile(r"^(?P<open>\()?(?P<eid>[A-Za-z0-9]+)(?:-[^()]*)?(?P<close>\))?$")


def _parse_entity_misc(misc: str) -> list[tuple[str, bool, bool]]:
    events: list[tuple[str, bool, bool]] = []
    if not misc or misc == "_":
        return events
    for field in misc.split("|"):
        if not field.startswith("Entity="):
            continue
        value = field[len("Entity="):]
        m = _ENTITY_PIECE_RE.match(value)
        if not m:
            continue
        events.append((m.group("eid"), bool(m.group("open")), bool(m.group("close"))))
    return events


def _misc_has_no_space_after(misc: str) -> bool:
    return misc != "_" and "SpaceAfter=No" in misc.split("|")


def parse_conllu(text: str) -> list[CorefDocument]:
    """Parse CoNLL-U with CorefUD Entity= bracket annotations into documents.

    Splits on '# newdoc' comment lines; if none are present, the whole input
    is treated as a single document named "doc1".
    """
    doc_blocks: list[list[str]] = []
    doc_ids: list[str] = []
    current: list[str] = []

    for line in text.splitlines():
        if line.startswith("# newdoc"):
            if current:
                doc_blocks.append(current)
                current = []
            m = re.search(r"newdoc(?:\s+id\s*=\s*(\S+))?", line)
            doc_id = m.group(1) if m and m.group(1) else f"doc{len(doc_blocks) + 1}"
            doc_ids.append(doc_id)
            continue
        current.append(line)

    if current:
        doc_blocks.append(current)
    if not doc_ids:
        doc_ids = ["doc1"]

    return [
        _parse_document_block(doc_id, block_lines)
        for doc_id, block_lines in zip(doc_ids, doc_blocks)
    ]


def _parse_document_block(doc_id: str, lines: list[str]) -> CorefDocument:
    tokens: list[Token] = []
    open_stack: dict[str, list[int]] = defaultdict(list)
    clusters_by_eid: dict[str, list[MentionSpan]] = defaultdict(list)

    text_parts: list[str] = []
    char_pos = 0
    sent_index = 0
    pending_space = False
    seen_token_in_sentence = False

    for line in lines:
        line = line.rstrip("\n")
        if line.startswith("#"):
            continue
        if not line.strip():
            if seen_token_in_sentence:
                sent_index += 1
                seen_token_in_sentence = False
            continue

        cols = line.split("\t")
        if len(cols) != 10:
            continue
        tok_id, form, _lemma, _upos, _xpos, _feats, _head, _deprel, _deps, misc = cols

        if "-" in tok_id:
            continue  # multiword-token range line: see module docstring limitation

        is_empty = "." in tok_id

        if pending_space:
            text_parts.append(" ")
            char_pos += 1
            pending_space = False

        idx = len(tokens)
        if is_empty:
            start_char = end_char = char_pos
        else:
            text_parts.append(form)
            start_char = char_pos
            char_pos += len(form)
            end_char = char_pos
            pending_space = not _misc_has_no_space_after(misc)

        seen_token_in_sentence = True
        tokens.append(Token(
            index=idx,
            text="" if is_empty else form,
            start_char=start_char,
            end_char=end_char,
            sent_index=sent_index,
            is_empty=is_empty,
        ))

        for eid, is_open, is_close in _parse_entity_misc(misc):
            if is_open:
                open_stack[eid].append(idx)
            if is_close:
                if not open_stack[eid]:
                    raise ValueError(f"close without open for entity {eid} in doc {doc_id}")
                start = open_stack[eid].pop()
                clusters_by_eid[eid].append(
                    MentionSpan(
                        start_token=start,
                        end_token=idx,
                        is_zero=(start == idx and tokens[start].is_empty),
                    )
                )

    for eid, stack in open_stack.items():
        if stack:
            raise ValueError(f"unclosed entity {eid} in doc {doc_id}")

    clusters = [spans for spans in clusters_by_eid.values() if spans]
    return CorefDocument(doc_id=doc_id, text="".join(text_parts), tokens=tokens, clusters=clusters)


def load_corefud_corpus(path: str | Path) -> list[CorefDocument]:
    path = Path(path)
    if path.is_dir():
        documents: list[CorefDocument] = []
        for file_path in sorted(path.glob("*.conllu")):
            documents.extend(parse_conllu(file_path.read_text(encoding="utf-8")))
        return documents
    return parse_conllu(path.read_text(encoding="utf-8"))
