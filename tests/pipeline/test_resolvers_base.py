from pipeline.resolvers.base import UnionFind, project_char_span_to_gold_tokens, supports_language
from pipeline.types import MentionSpan, Token


def test_project_char_span_picks_overlapping_gold_tokens():
    gold_tokens = [
        Token(0, "John", 0, 4, 0),
        Token(1, "arrived", 5, 12, 0),
    ]
    span = project_char_span_to_gold_tokens(0, 4, gold_tokens)
    assert span == MentionSpan(start_token=0, end_token=0)


def test_project_char_span_spans_multiple_gold_tokens():
    gold_tokens = [
        Token(0, "the", 0, 3, 0),
        Token(1, "United", 4, 10, 0),
        Token(2, "Nations", 11, 18, 0),
    ]
    span = project_char_span_to_gold_tokens(4, 18, gold_tokens)
    assert span == MentionSpan(start_token=1, end_token=2)


def test_project_char_span_returns_none_when_no_overlap():
    gold_tokens = [Token(0, "John", 0, 4, 0)]
    assert project_char_span_to_gold_tokens(10, 14, gold_tokens) is None


def test_project_char_span_skips_empty_tokens():
    gold_tokens = [
        Token(0, "John", 0, 4, 0),
        Token(1, "", 4, 4, 0, is_empty=True),
    ]
    span = project_char_span_to_gold_tokens(4, 4, gold_tokens)
    assert span is None


def test_union_find_groups_transitively_linked_indices():
    uf = UnionFind()
    uf.union(3, 7)
    uf.union(7, 12)
    uf.union(0, 1)
    groups = {frozenset(g) for g in uf.groups()}
    assert frozenset({3, 7, 12}) in groups
    assert frozenset({0, 1}) in groups


class _FakeAdapter:
    language_support = {"en"}


class _AnyLanguageAdapter:
    language_support = "any"


def test_supports_language_checks_set_membership():
    assert supports_language(_FakeAdapter(), "en") is True
    assert supports_language(_FakeAdapter(), "ru") is False


def test_supports_language_any_accepts_everything():
    assert supports_language(_AnyLanguageAdapter(), "ru") is True
