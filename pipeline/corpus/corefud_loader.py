from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from ..types import CorefDocument, MentionSpan, Token

# A single Entity= value may pack SEVERAL bracket pieces, concatenated with no
# separator between them (CorefUD does NOT "|"-separate them -- "|" separates
# whole MISC fields). Examples of one Entity= value:
#   (e1-person-1-new)              -- opens and closes e1 on this token
#   (e1-person-1-new)(e2-org-2-new)-- two entities open+close on this token
#   (e3-org-1-new(e4-loc-1-new)    -- opens e3, opens+closes nested e4
#   e5)(e6-loc-1-new)              -- closes e5, opens+closes e6
#   e2)                            -- closes e2
# So we must scan the whole value left-to-right with finditer, not try to
# match the entire value as one piece.
_ENTITY_PIECE_RE = re.compile(
    r"\((?P<open_eid>[^-()\s|]+)(?:-[^()|]*)?(?P<open_close>\))?"  # "(eid..."  / "(eid...)"
    r"|(?P<close_eid>[^-()\s|]+)\)"  # "eid)"
)


def _parse_entity_misc(misc: str) -> list[tuple[str, bool, bool]]:
    """Extract (entity_id, opens_here, closes_here) events from a MISC column.

    Returns the events in the order they appear in the Entity= value, so a
    token that closes one entity and then opens another is handled correctly.
    """
    events: list[tuple[str, bool, bool]] = []
    if not misc or misc == "_":
        return events
    for field in misc.split("|"):
        if not field.startswith("Entity="):
            continue
        value = field[len("Entity="):]
        for m in _ENTITY_PIECE_RE.finditer(value):
            if m.group("open_eid") is not None:
                events.append((m.group("open_eid"), True, bool(m.group("open_close"))))
            else:
                events.append((m.group("close_eid"), False, True))
    return events


def _misc_has_no_space_after(misc: str) -> bool:
    return misc != "_" and "SpaceAfter=No" in misc.split("|")


def _is_token_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return False
    return len(line.split("\t")) == 10


def parse_conllu(text: str) -> list[CorefDocument]:
    """Parse CoNLL-U with CorefUD Entity= bracket annotations into documents.

    Splits on '# newdoc' comment lines; if none are present, the whole input
    is treated as a single document named "doc1".

    Content appearing *before* the first '# newdoc' line (real CorefUD files
    open with a '# global.Entity = ...' header) is kept in lockstep with its
    own id rather than being silently attached to the wrong document: if it
    holds no token lines it is discarded, otherwise it becomes its own
    document.
    """
    # (doc_id, lines) pairs, built in lockstep so ids and blocks can never
    # desync (a previous zip() of two parallel lists silently misaligned every
    # document and dropped the last one whenever a header preceded '# newdoc').
    docs: list[tuple[str, list[str]]] = []
    leading: list[str] = []

    for line in text.splitlines():
        if line.startswith("# newdoc"):
            m = re.search(r"newdoc(?:\s+id\s*=\s*(\S+))?", line)
            doc_id = m.group(1) if m and m.group(1) else f"doc{len(docs) + 1}"
            docs.append((doc_id, []))
            continue
        if docs:
            docs[-1][1].append(line)
        else:
            leading.append(line)

    if any(_is_token_line(line) for line in leading):
        # Either a file with no '# newdoc' at all (docstring says: one "doc1"),
        # or -- pathologically -- real tokens before the first '# newdoc', which
        # get their own id so they cannot steal another document's id.
        synthetic_id = "doc1" if not docs else "doc0"
        docs.insert(0, (synthetic_id, leading))

    return [_parse_document_block(doc_id, block_lines) for doc_id, block_lines in docs]


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
