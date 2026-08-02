import sys
import types

from pipeline.graph.rule_based_graph_adapter import RuleBasedGraphAdapter


def _install_fake_module(monkeypatch):
    fake = types.ModuleType("graphBuilder")

    def _fake_extract_graph_edges(text):
        return [("john", "arrive", "")] if "John" in text else []

    def _fake_merge_graph(graph, new_edges):
        nodes = set(graph["nodes"])
        for a1, a2, meaning in new_edges:
            nodes.add(a1)
            nodes.add(a2)
            graph["edges"].append({"agent_1": a1, "agent_2": a2, "meaning": meaning, "weight": 1})
        graph["nodes"] = list(nodes)
        return graph

    fake.extract_graph_edges = _fake_extract_graph_edges
    fake.merge_graph = _fake_merge_graph
    monkeypatch.setitem(sys.modules, "graphBuilder", fake)


def test_build_returns_nodes_and_edges_shape(monkeypatch):
    _install_fake_module(monkeypatch)
    adapter = RuleBasedGraphAdapter()
    graph = adapter.build("John arrived.")
    assert set(graph["nodes"]) == {"john", "arrive"}
    assert graph["edges"] == [{"agent_1": "john", "agent_2": "arrive", "meaning": "", "weight": 1}]


def test_adapter_metadata():
    adapter = RuleBasedGraphAdapter()
    assert adapter.name == "RuleBased"
    assert adapter.backend_name == "RuleBased"
    assert adapter.language_support == {"en"}
