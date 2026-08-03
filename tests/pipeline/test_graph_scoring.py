import sys
import types

import networkx as nx

from pipeline.eval.graph_scoring import compute_graph_scores, compute_node_duplication_rate, compute_smatch


def _install_fake_graph_metrics(monkeypatch):
    """Kept for parity with the other suites' sys.modules pattern.

    graph_scoring no longer imports graphMetrics._to_networkx (it reads the
    backend-native dicts directly, so that node duplicates survive and edge
    direction is preserved), so this fake is inert for these tests -- but it
    guarantees no accidental re-introduction of the dependency goes unnoticed
    in an environment without text-corpuses-processing checked out.
    """
    fake = types.ModuleType("graphMetrics")

    def _to_networkx(graph_dict, backend):
        G = nx.Graph()
        if backend == "RuleBased":
            for node in graph_dict.get("nodes", []):
                G.add_node(node)
            for edge in graph_dict.get("edges", []):
                G.add_edge(edge["agent_1"], edge["agent_2"], weight=edge.get("weight", 1), label=edge.get("meaning", ""))
        else:
            for node in graph_dict.get("nodes", []):
                G.add_node(node["id"], label=node["label"])
            for edge in graph_dict.get("edges", []):
                G.add_edge(edge["source"], edge["target"], weight=edge.get("weight", 1), label=edge.get("label", ""))
        return G

    fake._to_networkx = _to_networkx
    monkeypatch.setitem(sys.modules, "graphMetrics", fake)


def _rule_based_graph(nodes, edges):
    return {"nodes": nodes, "edges": [{"agent_1": a, "agent_2": b, "meaning": m, "weight": 1} for a, b, m in edges]}


def test_identical_graphs_score_perfect_on_every_metric(monkeypatch):
    _install_fake_graph_metrics(monkeypatch)
    g = _rule_based_graph(["obama", "hawaii"], [("obama", "hawaii", "born in")])
    scores = compute_graph_scores(g, g, backend="RuleBased")
    assert scores["node_precision_recall_f1"]["f1"] == 1.0
    assert scores["edge_precision_recall_f1"]["f1"] == 1.0
    assert scores["smatch"]["f1"] == 1.0


def test_predicted_missing_a_node_reduces_recall_not_precision(monkeypatch):
    _install_fake_graph_metrics(monkeypatch)
    oracle = _rule_based_graph(["obama", "hawaii"], [("obama", "hawaii", "born in")])
    predicted = _rule_based_graph(["obama"], [])
    scores = compute_graph_scores(oracle, predicted, backend="RuleBased")
    assert scores["node_precision_recall_f1"]["precision"] == 1.0
    assert scores["node_precision_recall_f1"]["recall"] == 0.5
    assert 0.0 < scores["smatch"]["f1"] < 1.0


def test_node_duplication_rate_counts_repeated_labels(monkeypatch):
    _install_fake_graph_metrics(monkeypatch)
    llm_graph = {
        "nodes": [{"id": "n0", "label": "obama"}, {"id": "n1", "label": "obama"}, {"id": "n2", "label": "hawaii"}],
        "edges": [],
    }
    rate = compute_node_duplication_rate(llm_graph, backend="LLMv2")
    assert rate == 1 / 3


def test_node_duplication_rate_sees_duplicates_in_a_rule_based_graph_dict(monkeypatch):
    """Duplication must be read off the raw dict, not off networkx.

    graphMetrics._to_networkx adds RuleBased nodes as bare string keys, which
    nx.Graph.add_node silently dedupes, so routing through it made this metric
    unconditionally 0.0 for the only backend with end-to-end coverage. (In
    practice graphBuilder.merge_graph also dedupes via set(), so real RuleBased
    graphs carry no duplicates -- but the metric must not be the thing hiding
    them.)
    """
    _install_fake_graph_metrics(monkeypatch)
    graph_with_dupes = _rule_based_graph(["obama", "obama", "hawaii", "hawaii"], [])
    assert compute_node_duplication_rate(graph_with_dupes, backend="RuleBased") == 0.5


def test_node_duplication_rate_zero_for_empty_graph(monkeypatch):
    _install_fake_graph_metrics(monkeypatch)
    empty_graph = {"nodes": [], "edges": []}
    assert compute_node_duplication_rate(empty_graph, backend="RuleBased") == 0.0


def test_smatch_cache_is_cleared_between_independent_comparisons(monkeypatch):
    """Regression test for the module-level smatch.match_triple_dict cache.

    smatch caches mapping-search results keyed by tuple(mapping), whose
    length equals the predicted graph's node count -- NOT by graph content.
    Two independent comparisons whose predicted graphs happen to have the
    same node count can collide on that key unless compute_smatch() clears
    the cache before every call. This test uses two same-size (3-node)
    comparisons: the first is a non-trivial, imperfect match (populating the
    cache for 3-length mapping tuples), the second is a trivially perfect
    match (identical predicted/oracle) that must score f1 == 1.0 regardless
    of what the first call left behind.
    """
    _install_fake_graph_metrics(monkeypatch)

    # First comparison: same node count (3), NOT identical -- populates
    # match_triple_dict with entries for 3-length mapping tuples.
    oracle_a = _rule_based_graph(["a", "b", "c"], [("a", "b", "r1"), ("b", "c", "r2")])
    predicted_a = _rule_based_graph(["a", "b", "x"], [("a", "b", "r1")])
    first = compute_smatch(oracle_a, predicted_a, backend="RuleBased")
    assert first["f1"] < 1.0

    # Second comparison: also 3 nodes, but predicted is identical to oracle
    # -- must score a perfect 1.0. Without smatch.match_triple_dict.clear()
    # immediately before get_amr_match(), this silently returns a stale,
    # lower score inherited from the first comparison's cache entries.
    oracle_b = _rule_based_graph(["p", "q", "r"], [("p", "q", "rel"), ("q", "r", "rel2")])
    predicted_b = _rule_based_graph(["p", "q", "r"], [("p", "q", "rel"), ("q", "r", "rel2")])
    second = compute_smatch(oracle_b, predicted_b, backend="RuleBased")
    assert second["f1"] == 1.0


# --- Finding 4: empty-vs-empty must agree across P/R and smatch ---


def test_two_empty_graphs_score_perfect_on_node_edge_and_smatch(monkeypatch):
    """A document where the extractor found nothing on both sides is agreement.

    _prf used to fall through its empty-set guards to 0.0/0.0/0.0 while
    compute_smatch returned 1.0 for the very same input (both serialize to the
    same trivial 'empty-graph' AMR line), so three metrics computed from one
    input disagreed about whether it was total failure or a perfect match.
    """
    _install_fake_graph_metrics(monkeypatch)
    empty = {"nodes": [], "edges": []}
    scores = compute_graph_scores(empty, empty, backend="RuleBased")
    assert scores["node_precision_recall_f1"] == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    assert scores["edge_precision_recall_f1"] == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    assert scores["smatch"]["f1"] == 1.0


def test_one_sided_empty_graph_is_still_a_failure(monkeypatch):
    _install_fake_graph_metrics(monkeypatch)
    oracle = _rule_based_graph(["obama", "hawaii"], [("obama", "hawaii", "born in")])
    empty = {"nodes": [], "edges": []}
    scores = compute_graph_scores(oracle, empty, backend="RuleBased")
    assert scores["node_precision_recall_f1"]["f1"] == 0.0
    assert scores["edge_precision_recall_f1"]["f1"] == 0.0


def test_graph_with_nodes_but_no_edges_still_scores_edges_as_agreeing(monkeypatch):
    _install_fake_graph_metrics(monkeypatch)
    g = _rule_based_graph(["obama", "hawaii"], [])
    scores = compute_graph_scores(g, g, backend="RuleBased")
    assert scores["node_precision_recall_f1"]["f1"] == 1.0
    assert scores["edge_precision_recall_f1"]["f1"] == 1.0


# --- Finding 7: edge direction must be preserved ---


def test_reversed_edge_direction_is_not_scored_as_a_match(monkeypatch):
    """"john --hit--> mary" and "mary --hit--> john" are different claims.

    _canonicalize used to sort edge endpoints alphabetically on an undirected
    nx.Graph, so a predicted graph that reversed who-did-what-to-whom scored a
    perfect edge F1.
    """
    _install_fake_graph_metrics(monkeypatch)
    oracle = _rule_based_graph(["john", "mary"], [("john", "mary", "hit")])
    reversed_pred = _rule_based_graph(["john", "mary"], [("mary", "john", "hit")])
    scores = compute_graph_scores(oracle, reversed_pred, backend="RuleBased")
    assert scores["node_precision_recall_f1"]["f1"] == 1.0  # same nodes
    assert scores["edge_precision_recall_f1"]["f1"] == 0.0  # opposite direction
    assert scores["smatch"]["f1"] < 1.0


def test_edge_direction_preserved_for_id_based_backends(monkeypatch):
    _install_fake_graph_metrics(monkeypatch)
    nodes = [{"id": "n0", "label": "john"}, {"id": "n1", "label": "mary"}]
    oracle = {"nodes": nodes, "edges": [{"source": "n0", "target": "n1", "label": "hit", "weight": 1}]}
    reversed_pred = {"nodes": nodes, "edges": [{"source": "n1", "target": "n0", "label": "hit", "weight": 1}]}
    scores = compute_graph_scores(oracle, reversed_pred, backend="LLMv2")
    assert scores["node_precision_recall_f1"]["f1"] == 1.0
    assert scores["edge_precision_recall_f1"]["f1"] == 0.0


def test_amr_line_for_directed_edges_is_still_parseable_by_smatch(monkeypatch):
    """Same direction on both sides must still be a perfect smatch."""
    _install_fake_graph_metrics(monkeypatch)
    g = _rule_based_graph(["john", "mary"], [("john", "mary", "hit")])
    assert compute_smatch(g, g, backend="RuleBased")["f1"] == 1.0
