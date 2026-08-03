from pipeline.eval.coref_scoring import score_coreference
from pipeline.types import MentionSpan


def _cluster(*pairs):
    return [MentionSpan(a, b) for a, b in pairs]


def test_perfect_match_scores_one_on_every_metric():
    gold = [_cluster((0, 0), (5, 5), (10, 10))]
    sysc = [_cluster((0, 0), (5, 5), (10, 10))]
    result = score_coreference(gold, sysc)
    assert result["muc"]["f1"] == 1.0
    assert result["b_cubed"]["f1"] == 1.0
    assert result["ceafe"]["f1"] == 1.0
    assert result["conll_f1"] == 1.0


def test_partial_match_between_zero_and_one():
    gold = [_cluster((0, 0), (5, 5), (10, 10))]
    sysc = [_cluster((0, 0), (5, 5))]  # missed the third mention
    result = score_coreference(gold, sysc)
    assert 0.0 < result["conll_f1"] < 1.0


def test_empty_system_output_scores_zero():
    gold = [_cluster((0, 0), (5, 5))]
    result = score_coreference(gold, [])
    assert result["conll_f1"] == 0.0


def test_gold_singleton_clusters_are_stripped_before_scoring():
    """CoNLL-2012 convention: singleton KEY clusters are removed before scoring.

    The CorefUD loader keeps every annotated mention, including clusters of
    size 1. Resolvers structurally cannot emit singletons, so leaving gold
    singletons in deflates every metric: a resolver that perfectly recovers
    the one real coreference link scored ~0.657 instead of 1.0.
    """
    gold = [
        _cluster((0, 0), (5, 5)),  # the one real coreference chain
        _cluster((2, 2)),          # singleton: annotated mention, no partner
        _cluster((7, 7)),          # singleton
        _cluster((9, 9)),          # singleton
    ]
    sysc = [_cluster((0, 0), (5, 5))]
    result = score_coreference(gold, sysc)
    assert result["conll_f1"] == 1.0


def test_gold_of_only_singletons_scores_zero_against_a_guessing_system():
    gold = [_cluster((0, 0)), _cluster((3, 3))]
    sysc = [_cluster((0, 0), (3, 3))]
    result = score_coreference(gold, sysc)
    assert result["conll_f1"] == 0.0


def test_all_returned_values_are_plain_python_floats():
    gold = [_cluster((0, 0), (5, 5))]
    sysc = [_cluster((0, 0), (5, 5))]
    result = score_coreference(gold, sysc)
    for metric_name in ("muc", "b_cubed", "ceafe"):
        for key in ("precision", "recall", "f1"):
            assert isinstance(result[metric_name][key], float)
    assert isinstance(result["conll_f1"], float)
