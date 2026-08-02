import sys
import types

import networkx as nx

from pipeline.eval.graph_scoring import compute_graph_scores, compute_node_duplication_rate


def _install_fake_graph_metrics(monkeypatch):
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
    graph_with_dupes = _rule_based_graph(["obama", "obama", "hawaii"], [])
    # note: RuleBased backend nodes are a flat string list in the real schema,
    # but networkx dedupes identical string nodes on add_node -- duplication
    # is only observable in ID-based backends. Test duplication with an
    # ID-based (LLMv2-shaped) graph instead:
    llm_graph = {
        "nodes": [{"id": "n0", "label": "obama"}, {"id": "n1", "label": "obama"}, {"id": "n2", "label": "hawaii"}],
        "edges": [],
    }
    rate = compute_node_duplication_rate(llm_graph, backend="LLMv2")
    assert rate == 1 / 3


def test_node_duplication_rate_zero_for_empty_graph(monkeypatch):
    _install_fake_graph_metrics(monkeypatch)
    empty_graph = {"nodes": [], "edges": []}
    assert compute_node_duplication_rate(empty_graph, backend="RuleBased") == 0.0
