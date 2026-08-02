from pipeline.corpus.oracle import build_oracle_text
from pipeline.types import CorefDocument, MentionSpan, Token


def _doc(text, tokens, clusters):
    return CorefDocument(doc_id="d1", text=text, tokens=tokens, clusters=clusters)


def test_substitutes_non_head_mentions_with_head_surface_text():
    text = "John arrived. He left."
    tokens = [
        Token(0, "John", 0, 4, 0),
        Token(1, "arrived", 5, 12, 0),
        Token(2, ".", 12, 13, 0),
        Token(3, "He", 14, 16, 1),
        Token(4, "left", 17, 21, 1),
        Token(5, ".", 21, 22, 1),
    ]
    clusters = [[MentionSpan(0, 0), MentionSpan(3, 3)]]
    doc = _doc(text, tokens, clusters)
    assert build_oracle_text(doc) == "John arrived. John left."


def test_singleton_clusters_are_left_unchanged():
    text = "John left."
    tokens = [Token(0, "John", 0, 4, 0), Token(1, "left", 5, 9, 0), Token(2, ".", 9, 10, 0)]
    doc = _doc(text, tokens, clusters=[[MentionSpan(0, 0)]])
    assert build_oracle_text(doc) == "John left."


def test_zero_mentions_are_skipped_not_substituted_into_text():
    text = "John arrived.  left."
    tokens = [
        Token(0, "John", 0, 4, 0),
        Token(1, "arrived", 5, 12, 0),
        Token(2, ".", 12, 13, 0),
        Token(3, "", 14, 14, 1, is_empty=True),
        Token(4, "left", 15, 19, 1),
        Token(5, ".", 19, 20, 1),
    ]
    clusters = [[MentionSpan(0, 0), MentionSpan(3, 3, is_zero=True)]]
    doc = _doc(text, tokens, clusters)
    assert build_oracle_text(doc) == text  # unchanged: nothing to replace at a zero-width span


def test_multiple_clusters_applied_without_offset_corruption():
    text = "John met Mary. He greeted her."
    tokens = [
        Token(0, "John", 0, 4, 0), Token(1, "met", 5, 8, 0), Token(2, "Mary", 9, 13, 0), Token(3, ".", 13, 14, 0),
        Token(4, "He", 15, 17, 1), Token(5, "greeted", 18, 25, 1), Token(6, "her", 26, 29, 1), Token(7, ".", 29, 30, 1),
    ]
    clusters = [[MentionSpan(0, 0), MentionSpan(4, 4)], [MentionSpan(2, 2), MentionSpan(6, 6)]]
    doc = _doc(text, tokens, clusters)
    assert build_oracle_text(doc) == "John met Mary. John greeted Mary."
