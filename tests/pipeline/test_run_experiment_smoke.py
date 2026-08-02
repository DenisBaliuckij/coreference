# tests/pipeline/test_run_experiment_smoke.py
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pipeline.run_experiment import run_experiment

FIXTURE_CORPUS = Path(__file__).resolve().parents[2] / "data" / "corpora" / "sample_en_mini.conllu"


@dataclass
class _FakeSpacyToken:
    idx: int
    text: str


class _FakeSpacyDoc:
    def __init__(self, tokens):
        self._tokens = tokens

    def __getitem__(self, i):
        return self._tokens[i]


@dataclass
class _FakeResolution:
    pronoun: str
    pronoun_index: int
    antecedent: Optional[str]
    antecedent_index: Optional[int]
    score: float


class _FakeBatchAnaphoraResolver:
    def resolve_document(self, text):
        tokens = [
            _FakeSpacyToken(0, "John"), _FakeSpacyToken(5, "arrived"), _FakeSpacyToken(12, "."),
            _FakeSpacyToken(14, "He"), _FakeSpacyToken(17, "left"), _FakeSpacyToken(21, "."),
        ]
        doc = _FakeSpacyDoc(tokens)
        resolutions = [
            _FakeResolution("He", 3, "John", 0, 100.0)
        ]
        return {"text": text, "doc": doc, "resolutions": resolutions}


def _fake_build_substitutions(doc, resolutions, mark=False):
    subs = []
    for r in resolutions:
        if r.antecedent is None:
            continue
        tok = doc[r.pronoun_index]
        subs.append(types.SimpleNamespace(start=tok.idx, end=tok.idx + len(tok.text), replacement=r.antecedent))
    subs.sort(key=lambda s: s.start, reverse=True)
    return subs


def _fake_apply_substitutions(text, substitutions):
    out = text
    for s in substitutions:
        out = out[: s.start] + s.replacement + out[s.end :]
    return out


def _install_fake_resolver_module(monkeypatch):
    fake = types.ModuleType("anaphoraResolverLapinLiass")
    fake.BatchAnaphoraResolver = _FakeBatchAnaphoraResolver
    fake.build_substitutions = _fake_build_substitutions
    fake.apply_substitutions = _fake_apply_substitutions
    monkeypatch.setitem(sys.modules, "anaphoraResolverLapinLiass", fake)


def _install_fake_graph_builder_module(monkeypatch):
    fake = types.ModuleType("graphBuilder")

    def _extract_graph_edges(text):
        edges = [("john", "arrive", "")]
        if "left" in text.lower():
            edges.append(("john", "leave", ""))
        return edges

    def _merge_graph(graph, new_edges):
        nodes = set(graph["nodes"])
        for a1, a2, meaning in new_edges:
            nodes.add(a1)
            nodes.add(a2)
            graph["edges"].append({"agent_1": a1, "agent_2": a2, "meaning": meaning, "weight": 1})
        graph["nodes"] = list(nodes)
        return graph

    fake.extract_graph_edges = _extract_graph_edges
    fake.merge_graph = _merge_graph
    monkeypatch.setitem(sys.modules, "graphBuilder", fake)


def _install_fake_graph_metrics_module(monkeypatch):
    import networkx as nx

    fake = types.ModuleType("graphMetrics")

    def _to_networkx(graph_dict, backend):
        G = nx.Graph()
        for node in graph_dict.get("nodes", []):
            G.add_node(node)
        for edge in graph_dict.get("edges", []):
            G.add_edge(edge["agent_1"], edge["agent_2"], weight=edge.get("weight", 1), label=edge.get("meaning", ""))
        return G

    fake._to_networkx = _to_networkx
    monkeypatch.setitem(sys.modules, "graphMetrics", fake)


def test_run_experiment_end_to_end_produces_results_json(monkeypatch, tmp_path):
    _install_fake_resolver_module(monkeypatch)
    _install_fake_graph_builder_module(monkeypatch)
    _install_fake_graph_metrics_module(monkeypatch)

    output_dir = tmp_path / "run1"
    results_path = run_experiment(
        corpus_path=FIXTURE_CORPUS,
        language="en",
        resolver_names=["LapinLiass"],
        graph_backend_names=["RuleBased"],
        output_dir=output_dir,
    )

    assert results_path.exists()
    assert (output_dir / "report.html").exists()

    import json
    data = json.loads(results_path.read_text(encoding="utf-8"))
    assert data["language"] == "en"
    assert len(data["results"]) == 1
    pairing = data["results"][0]
    assert pairing["resolver"] == "LapinLiass"
    assert pairing["graph_backend"] == "RuleBased"
    assert pairing["coreference_metrics"]["conll_f1"] == 1.0  # LapinLiass resolves He->John, matching gold e1
