from pipeline.types import CorefDocument, MentionSpan, ResolverOutput, Token


def test_mention_span_equality_and_defaults():
    a = MentionSpan(start_token=1, end_token=2)
    b = MentionSpan(start_token=1, end_token=2, is_zero=False)
    assert a == b
    assert a.is_zero is False


def test_coref_document_holds_tokens_and_clusters():
    tok = Token(index=0, text="John", start_char=0, end_char=4, sent_index=0)
    doc = CorefDocument(doc_id="d1", text="John", tokens=[tok], clusters=[])
    assert doc.tokens[0].text == "John"
    assert doc.clusters == []


def test_resolver_output_allows_none_clusters():
    out = ResolverOutput(resolved_text="John left.", clusters=None)
    assert out.clusters is None
