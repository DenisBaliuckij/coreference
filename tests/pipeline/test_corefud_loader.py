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


# --- Finding 1: content before the first "# newdoc" must not desync doc ids ---

REAL_SHAPED_TWO_DOCS = """\
# global.Entity = eid-etype-head-other
# newdoc id = doc_a
# sent_id = doc_a-1
1\tJohn\tJohn\tPROPN\t_\t_\t2\tnsubj\t_\tEntity=(e1-person-1-new)
2\tarrived\tarrive\tVERB\t_\t_\t0\troot\t_\tSpaceAfter=No
3\t.\t.\tPUNCT\t_\t_\t2\tpunct\t_\t_

# sent_id = doc_a-2
1\tHe\the\tPRON\t_\t_\t2\tnsubj\t_\tEntity=(e1)
2\tleft\tleave\tVERB\t_\t_\t0\troot\t_\tSpaceAfter=No
3\t.\t.\tPUNCT\t_\t_\t2\tpunct\t_\t_

# newdoc id = doc_b
# sent_id = doc_b-1
1\tMary\tMary\tPROPN\t_\t_\t2\tnsubj\t_\tEntity=(e2-person-1-new)
2\tsmiled\tsmile\tVERB\t_\t_\t0\troot\t_\tSpaceAfter=No
3\t.\t.\tPUNCT\t_\t_\t2\tpunct\t_\t_

# sent_id = doc_b-2
1\tShe\tshe\tPRON\t_\t_\t2\tnsubj\t_\tEntity=(e2)
2\twaved\twave\tVERB\t_\t_\t0\troot\t_\tSpaceAfter=No
3\t.\t.\tPUNCT\t_\t_\t2\tpunct\t_\t_
"""


def test_global_entity_header_before_first_newdoc_does_not_desync_documents():
    docs = parse_conllu(REAL_SHAPED_TWO_DOCS)
    assert [d.doc_id for d in docs] == ["doc_a", "doc_b"]
    assert docs[0].text == "John arrived. He left."
    assert docs[1].text == "Mary smiled. She waved."
    assert docs[0].clusters == [[MentionSpan(0, 0, False), MentionSpan(3, 3, False)]]
    assert docs[1].clusters == [[MentionSpan(0, 0, False), MentionSpan(3, 3, False)]]


def test_leading_comment_only_block_is_discarded_not_emitted_as_a_document():
    text = (
        "# global.Entity = eid-etype-head-other\n"
        "# some other corpus-level comment\n"
        "# newdoc id = only\n"
        "1\tHi\thi\tINTJ\t_\t_\t0\troot\t_\t_\n"
    )
    docs = parse_conllu(text)
    assert [d.doc_id for d in docs] == ["only"]
    assert docs[0].text == "Hi"


def test_file_without_any_newdoc_still_becomes_a_single_doc1():
    text = (
        "# global.Entity = eid-etype-head-other\n"
        "# sent_id = s1\n"
        "1\tHi\thi\tINTJ\t_\t_\t0\troot\t_\t_\n"
    )
    docs = parse_conllu(text)
    assert [d.doc_id for d in docs] == ["doc1"]
    assert docs[0].text == "Hi"


def test_newdoc_without_explicit_id_gets_sequential_synthetic_ids():
    text = (
        "# newdoc\n"
        "1\tHi\thi\tINTJ\t_\t_\t0\troot\t_\t_\n"
        "\n"
        "# newdoc\n"
        "1\tBye\tbye\tINTJ\t_\t_\t0\troot\t_\t_\n"
    )
    docs = parse_conllu(text)
    assert [d.doc_id for d in docs] == ["doc1", "doc2"]


# --- Finding 2: multi-piece Entity= values (concatenated, not "|"-separated) ---

TWO_ENTITIES_ON_ONE_TOKEN = """\
1\tJohn\tJohn\tPROPN\t_\t_\t0\troot\t_\tEntity=(e1-person-1-new)(e2-org-2-new)
2\tand\tand\tCCONJ\t_\t_\t1\tcc\t_\t_
3\tHe\the\tPRON\t_\t_\t1\tnsubj\t_\tEntity=(e1)
4\tthere\tthere\tADV\t_\t_\t1\tadvmod\t_\t_
5\tit\tit\tPRON\t_\t_\t1\tobj\t_\tEntity=(e2)
"""


def test_two_entities_opening_and_closing_on_the_same_token():
    doc = parse_conllu(TWO_ENTITIES_ON_ONE_TOKEN)[0]
    clusters = {tuple(sorted((s.start_token, s.end_token) for s in c)) for c in doc.clusters}
    assert clusters == {((0, 0), (2, 2)), ((0, 0), (4, 4))}


MIXED_CLOSE_AND_OPEN_ON_ONE_TOKEN = """\
1\tthe\tthe\tDET\t_\t_\t3\tdet\t_\tEntity=(e3-org-1-new
2\tCity\tcity\tPROPN\t_\t_\t3\tcompound\t_\t_
3\tLondon\tLondon\tPROPN\t_\t_\t0\troot\t_\tEntity=e3)(e4-loc-1-new-1-sgl-1)
"""


def test_one_entity_closes_while_another_opens_and_closes_on_the_same_token():
    doc = parse_conllu(MIXED_CLOSE_AND_OPEN_ON_ONE_TOKEN)[0]
    spans = {(c[0].start_token, c[0].end_token) for c in doc.clusters}
    assert spans == {(0, 2), (2, 2)}


NESTED_OPEN_INSIDE_OPEN = """\
1\tthe\tthe\tDET\t_\t_\t4\tdet\t_\tEntity=(e3-org-1-new(e4-loc-1-new-1-sgl-1)
2\tof\tof\tADP\t_\t_\t4\tcase\t_\t_
3\tGreater\tGreater\tPROPN\t_\t_\t4\tcompound\t_\t_
4\tLondon\tLondon\tPROPN\t_\t_\t0\troot\t_\tEntity=e3)
"""


def test_nested_open_inside_an_open_in_one_entity_value():
    doc = parse_conllu(NESTED_OPEN_INSIDE_OPEN)[0]
    spans = {(c[0].start_token, c[0].end_token) for c in doc.clusters}
    assert spans == {(0, 3), (0, 0)}


DISTANT_CLOSE_PLUS_COMBINED = """\
1\tAcme\tAcme\tPROPN\t_\t_\t0\troot\t_\tEntity=(e5-org-1-new
2\tCorp\tCorp\tPROPN\t_\t_\t1\tflat\t_\t_
3\tof\tof\tADP\t_\t_\t4\tcase\t_\t_
4\tParis\tParis\tPROPN\t_\t_\t1\tnmod\t_\tEntity=e5)(e6-loc-1-new)
5\tit\tit\tPRON\t_\t_\t0\troot\t_\tEntity=(e6)
"""


def test_token_closes_one_entity_and_opens_closes_another_forming_a_real_cluster():
    doc = parse_conllu(DISTANT_CLOSE_PLUS_COMBINED)[0]
    clusters = {tuple(sorted((s.start_token, s.end_token) for s in c)) for c in doc.clusters}
    assert clusters == {((0, 3),), ((3, 3), (4, 4))}


def test_multi_piece_entity_value_alongside_other_misc_fields():
    text = (
        "1\tJohn\tJohn\tPROPN\t_\t_\t0\troot\t_\t"
        "Entity=(e1-person-1-new)(e2-org-2-new)|SpaceAfter=No\n"
        "2\t.\t.\tPUNCT\t_\t_\t1\tpunct\t_\t_\n"
    )
    doc = parse_conllu(text)[0]
    assert doc.text == "John."
    assert len(doc.clusters) == 2
