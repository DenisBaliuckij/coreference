# tests/pipeline/test_corefud_loader.py
from pipeline.corpus.corefud_loader import parse_conllu
from pipeline.types import MentionSpan


SINGLE_TOKEN_MENTIONS = """\
# newdoc id = doc1
# sent_id = doc1-1
1\tJohn\tJohn\tPROPN\t_\t_\t2\tnsubj\t_\tEntity=(e1-person-1-new-1-sgl-1)
2\tarrived\tarrive\tVERB\t_\t_\t0\troot\t_\tSpaceAfter=No
3\t.\t.\tPUNCT\t_\t_\t2\tpunct\t_\t_

# sent_id = doc1-2
1\t0\t0\tPRON\t_\t_\t2\tnsubj\t_\tEntity=(e1)
1.1\tPro\tpro\tPRON\t_\t_\t_\t_\t_\t_
2\tleft\tleave\tVERB\t_\t_\t0\troot\t_\tSpaceAfter=No
3\t.\t.\tPUNCT\t_\t_\t2\tpunct\t_\t_
"""


def test_reconstructs_text_with_space_after_no_respected():
    docs = parse_conllu(SINGLE_TOKEN_MENTIONS)
    assert len(docs) == 1
    assert docs[0].text == "John arrived. 0 left."


def test_sentence_indices_increment_per_blank_line():
    doc = parse_conllu(SINGLE_TOKEN_MENTIONS)[0]
    assert doc.tokens[0].sent_index == 0  # John
    assert doc.tokens[3].sent_index == 1  # 0 (second sentence)


def test_empty_node_marked_is_empty_and_not_counted_in_entity_span():
    doc = parse_conllu(SINGLE_TOKEN_MENTIONS)[0]
    empty_tokens = [t for t in doc.tokens if t.is_empty]
    assert len(empty_tokens) == 1
    assert empty_tokens[0].text == ""


def test_two_mentions_of_same_entity_form_one_cluster():
    doc = parse_conllu(SINGLE_TOKEN_MENTIONS)[0]
    assert doc.clusters == [[MentionSpan(0, 0, False), MentionSpan(3, 3, False)]]


MULTI_TOKEN_SPAN = """\
1\tthe\tthe\tDET\t_\t_\t3\tdet\t_\tEntity=(e2-org-3-new-1-sgl-1
2\tUnited\tUnited\tPROPN\t_\t_\t3\tcompound\t_\t_
3\tNations\tNations\tPROPN\t_\t_\t0\troot\t_\tEntity=e2)
"""


def test_multi_token_span_open_and_close_on_different_tokens():
    doc = parse_conllu(MULTI_TOKEN_SPAN)[0]
    assert doc.clusters == [[MentionSpan(0, 2, False)]]


OVERLAPPING_ENTITIES = """\
1\tthe\tthe\tDET\t_\t_\t4\tdet\t_\tEntity=(e3-org-1-new-1-sgl-1
2\tCity\tcity\tPROPN\t_\t_\t4\tcompound\t_\tEntity=(e4-loc-1-new-1-sgl-1)
3\tof\tof\tADP\t_\t_\t4\tcase\t_\t_
4\tLondon\tLondon\tPROPN\t_\t_\t0\troot\t_\tEntity=e3)
"""


def test_overlapping_entities_use_independent_stacks():
    doc = parse_conllu(OVERLAPPING_ENTITIES)[0]
    clusters_by_span = {c[0].start_token: c for c in doc.clusters}
    assert clusters_by_span[0] == [MentionSpan(0, 3, False)]
    assert clusters_by_span[1] == [MentionSpan(1, 1, False)]


def test_multiple_newdoc_blocks_produce_multiple_documents():
    text = (
        "# newdoc id = a\n"
        "1\tHi\thi\tINTJ\t_\t_\t0\troot\t_\t_\n"
        "\n"
        "# newdoc id = b\n"
        "1\tBye\tbye\tINTJ\t_\t_\t0\troot\t_\t_\n"
    )
    docs = parse_conllu(text)
    assert [d.doc_id for d in docs] == ["a", "b"]
    assert docs[0].text == "Hi"
    assert docs[1].text == "Bye"


def test_unclosed_entity_raises():
    text = "1\tJohn\tJohn\tPROPN\t_\t_\t0\troot\t_\tEntity=(e1-person-1\n"
    try:
        parse_conllu(text)
        assert False, "expected ValueError"
    except ValueError:
        pass
