import pytest

from pipeline.corpus.corefud_loader import parse_conllu
from pipeline.eval.corefud_scoring import (
    RESPONSE_ENTITY_SCHEMA,
    conllu_text,
    key_text_for_scorer,
    normalized_key,
    parse_scorer_output,
    score_corpus,
    scorer_available,
    write_response,
)
from pipeline.types import MentionSpan

KEY = """\
# global.Entity = eid-etype-head-other
# newdoc id = docA
# sent_id = docA-1
1\tJohn\tJohn\tPROPN\t_\t_\t2\tnsubj\t_\tEntity=(e1-person-1-new)
2\tarrived\tarrive\tVERB\t_\t_\t0\troot\t_\tSpaceAfter=No
3\t.\t.\tPUNCT\t_\t_\t2\tpunct\t_\t_

# sent_id = docA-2
1\tHe\the\tPRON\t_\t_\t3\tnsubj\t_\tEntity=(e1)
2\tthen\tthen\tADV\t_\t_\t3\tadvmod\t_\t_
3\tleft\tleave\tVERB\t_\t_\t0\troot\t_\t_
4\tthe\tthe\tDET\t_\t_\t5\tdet\t_\tEntity=(e2-place-2-new
5\thouse\thouse\tNOUN\t_\t_\t3\tobj\t_\tEntity=e2)|SpaceAfter=No
6\t.\t.\tPUNCT\t_\t_\t3\tpunct\t_\t_

# newdoc id = docB
# sent_id = docB-1
1\tMary\tMary\tPROPN\t_\t_\t2\tnsubj\t_\tEntity=(e3-person-1-new)
2\tsmiled\tsmile\tVERB\t_\t_\t0\troot\t_\tSpaceAfter=No
3\t.\t.\tPUNCT\t_\t_\t2\tpunct\t_\t_

# sent_id = docB-2
1\tShe\tshe\tPRON\t_\t_\t2\tnsubj\t_\tEntity=(e3)|SplitAnte=e3<e1
2\twaved\twave\tVERB\t_\t_\t0\troot\t_\tSpaceAfter=No
3\t.\t.\tPUNCT\t_\t_\t2\tpunct\t_\t_
"""


def _token_lines(text):
    return [l.split("\t") for l in text.splitlines() if l and not l.startswith("#")]


def _gold(text=KEY):
    return {d.doc_id: d.clusters for d in parse_conllu(text)}


def test_response_keeps_every_token_and_strips_all_gold_coreference():
    response, dropped = write_response(KEY, {})
    key_rows, resp_rows = _token_lines(KEY), _token_lines(response)
    assert dropped == 0
    assert [r[:9] for r in resp_rows] == [r[:9] for r in key_rows]  # aligned, as the scorer requires
    assert "Entity=" not in response and "SplitAnte=" not in response
    assert "SpaceAfter=No" in response  # unrelated MISC attributes survive
    assert response.count(RESPONSE_ENTITY_SCHEMA) == 1


def test_response_of_the_gold_clusters_round_trips_through_the_loader():
    response, _ = write_response(KEY, _gold())
    reparsed = {d.doc_id: sorted(sorted((m.start_token, m.end_token) for m in c) for c in d.clusters)
                for d in parse_conllu(response)}
    expected = {doc_id: sorted(sorted((m.start_token, m.end_token) for m in c) for c in clusters)
                for doc_id, clusters in _gold().items()}
    assert reparsed == expected


def test_multiword_mention_gets_open_and_close_brackets_and_head_from_the_tree():
    response, _ = write_response(KEY, {"docA": [[MentionSpan(6, 7)]]})  # "the house"
    misc = [r[9] for r in _token_lines(response)]
    assert misc[6] == "Entity=(c1--2"  # head = "house" (its HEAD, "left", is outside the mention)
    assert "Entity=c1)" in misc[7].split("|") and "SpaceAfter=No" in misc[7].split("|")


def test_nested_mentions_on_one_token_are_ordered_so_the_loader_can_read_them():
    # "the house" and "house" both end on token 7; "house" also starts there
    predicted = {"docA": [[MentionSpan(6, 7)], [MentionSpan(7, 7)]]}
    response, _ = write_response(KEY, predicted)
    reparsed = parse_conllu(response)[0]
    spans = sorted((m.start_token, m.end_token) for c in reparsed.clusters for m in c)
    assert spans == [(6, 7), (7, 7)]


def test_mentions_crossing_a_sentence_boundary_are_dropped_and_counted():
    response, dropped = write_response(KEY, {"docA": [[MentionSpan(0, 0), MentionSpan(2, 3)]]})
    assert dropped == 1
    reparsed = parse_conllu(response)[0]
    assert [(m.start_token, m.end_token) for c in reparsed.clusters for m in c] == [(0, 0)]


def test_tokens_before_the_first_newdoc_form_their_own_document():
    # the loader names leading tokens "doc0" when other documents follow
    text = KEY.replace("# newdoc id = docA\n", "")
    predicted = {"doc0": [[MentionSpan(0, 0), MentionSpan(3, 3)]]}
    response, _ = write_response(text, predicted)
    docs = parse_conllu(response)
    assert [d.doc_id for d in docs] == ["doc0", "docB"]
    assert [(m.start_token, m.end_token) for m in docs[0].clusters[0]] == [(0, 0), (3, 3)]


def test_key_and_response_files_end_with_the_blank_line_conllu_requires():
    assert conllu_text("1\ta\n") == "1\ta\n\n"
    assert conllu_text("1\ta\n\n\n") == "1\ta\n\n"
    assert write_response(KEY, {})[0].endswith("_\n\n")


def test_key_without_entity_header_gets_the_standard_one():
    no_header = KEY.replace("# global.Entity = eid-etype-head-other\n", "")
    fixed = key_text_for_scorer(no_header)
    assert fixed.startswith(RESPONSE_ENTITY_SCHEMA + "\n")
    assert key_text_for_scorer(KEY).count("# global.Entity") == 1  # an existing header is kept


def test_parse_scorer_output_reads_fractions_and_conll_average():
    out = (
        "muc\nRecall: 50.00  Precision: 100.00  F1: 66.67\n"
        "bcub\nRecall: 75.00  Precision: 100.00  F1: 85.71\n"
        "ceafe\nRecall: 80.00  Precision: 80.00  F1: 80.00\n"
        "CoNLL score: 77.46\n"
    )
    result = parse_scorer_output(out)
    assert result["muc"] == {"precision": 1.0, "recall": 0.5, "f1": pytest.approx(0.6667)}
    assert result["b_cubed"]["f1"] == pytest.approx(0.8571)
    assert result["conll_f1"] == pytest.approx((0.6667 + 0.8571 + 0.80) / 3)


def test_parse_scorer_output_fails_loudly_on_missing_metrics():
    with pytest.raises(RuntimeError):
        parse_scorer_output("muc\nRecall: 1.00  Precision: 1.00  F1: 1.00\n")


needs_scorer = pytest.mark.skipif(not scorer_available(), reason="official CorefUD scorer not installed")


def _score(tmp_path, predicted, match="exact"):
    key = tmp_path / "key.conllu"
    key.write_text(conllu_text(KEY), encoding="utf-8")
    response = tmp_path / "response.conllu"
    response.write_text(write_response(KEY, predicted)[0], encoding="utf-8")
    return score_corpus(key, response, match=match)


@needs_scorer
@pytest.mark.parametrize("match", ["exact", "head"])
def test_gold_clusters_score_one_with_the_official_scorer(tmp_path, match):
    result = _score(tmp_path, _gold(), match)
    assert result["conll_f1"] == pytest.approx(1.0)
    assert result["scope"] == "corpus" and result["scorer"] == "corefud-scorer" and result["match"] == match


GUM_STYLE_KEY = """\
# global.Entity = GRP-etype-infstat-salience-centering-minspan-link-identity
# newdoc id = gum1
# sent_id = gum1-1
1\tThe\tthe\tDET\t_\t_\t2\tdet\t_\tEntity=(1-person-new-nnn-cf1-1,2-sgl
2\tdog\tdog\tNOUN\t_\t_\t3\tnsubj\t_\tEntity=1)
3\tbarked\tbark\tVERB\t_\t_\t0\troot\t_\tSpaceAfter=No
4\t.\t.\tPUNCT\t_\t_\t3\tpunct\t_\t_

# sent_id = gum1-2
1\tThe\tthe\tDET\t_\t_\t2\tdet\t_\tEntity=(1-person-giv-nnn-cf1-1,2-sgl
2\tdog\tdog\tNOUN\t_\t_\t3\tnsubj\t_\tEntity=1)
3\tslept\tsleep\tVERB\t_\t_\t0\troot\t_\tSpaceAfter=No
4\t.\t.\tPUNCT\t_\t_\t3\tpunct\t_\t_
"""


@needs_scorer
@pytest.mark.parametrize("match", ["exact", "partial", "head"])
def test_gold_scores_one_under_every_match_mode_even_without_head_attributes(tmp_path, match):
    """GUM's Entity schema has no head field: with the original key, head matching scored
    gold against itself at 0.53 on GUM test. The normalized key uses the response's rule."""
    gold = {d.doc_id: d.clusters for d in parse_conllu(GUM_STYLE_KEY)}
    key_text, dropped = normalized_key(GUM_STYLE_KEY, gold)
    key = tmp_path / "key.conllu"
    key.write_text(key_text, encoding="utf-8")
    response = tmp_path / "response.conllu"
    response.write_text(write_response(key_text, gold)[0], encoding="utf-8")
    assert dropped == 0
    assert score_corpus(key, response, match=match)["conll_f1"] == pytest.approx(1.0)


@needs_scorer
def test_missing_one_document_lowers_the_corpus_level_score(tmp_path):
    gold = _gold()
    result = _score(tmp_path, {"docA": gold["docA"]})
    assert 0.0 < result["conll_f1"] < 1.0
    assert result["muc"]["recall"] == pytest.approx(0.5)  # 1 of 2 coreference links found


@needs_scorer
def test_no_predictions_score_zero(tmp_path):
    assert _score(tmp_path, {})["conll_f1"] == 0.0
