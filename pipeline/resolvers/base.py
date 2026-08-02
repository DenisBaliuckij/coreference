from __future__ import annotations

from typing import Literal, Protocol

from ..types import CorefDocument, MentionSpan, ResolverOutput, Token


class ResolverAdapter(Protocol):
    name: str
    language_support: set[str] | Literal["any"]

    def resolve(self, doc: CorefDocument) -> ResolverOutput: ...


def supports_language(adapter, language: str) -> bool:
    return adapter.language_support == "any" or language in adapter.language_support


def project_char_span_to_gold_tokens(
    start_char: int, end_char: int, gold_tokens: list[Token]
) -> MentionSpan | None:
    overlapping = [
        tok.index for tok in gold_tokens
        if not tok.is_empty and tok.start_char < end_char and tok.end_char > start_char
    ]
    if not overlapping:
        return None
    return MentionSpan(start_token=min(overlapping), end_token=max(overlapping))


class UnionFind:
    def __init__(self) -> None:
        self._parent: dict[int, int] = {}

    def _find(self, x: int) -> int:
        self._parent.setdefault(x, x)
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self._find(a), self._find(b)
        if ra != rb:
            self._parent[ra] = rb

    def groups(self) -> list[list[int]]:
        buckets: dict[int, list[int]] = {}
        for x in self._parent:
            buckets.setdefault(self._find(x), []).append(x)
        return list(buckets.values())
