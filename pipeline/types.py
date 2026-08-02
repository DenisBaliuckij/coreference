from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Token:
    index: int
    text: str
    start_char: int
    end_char: int
    sent_index: int
    is_empty: bool = False


@dataclass(frozen=True)
class MentionSpan:
    start_token: int
    end_token: int  # inclusive
    is_zero: bool = False


@dataclass
class CorefDocument:
    doc_id: str
    text: str
    tokens: list[Token]
    clusters: list[list[MentionSpan]]


@dataclass
class ResolverOutput:
    resolved_text: str
    clusters: list[list[MentionSpan]] | None
