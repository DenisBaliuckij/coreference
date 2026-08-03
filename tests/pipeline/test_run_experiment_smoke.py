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
    # Finding 6b: duplication rate is aggregated to the pairing level, not
    # left buried in the per-document entries.
    assert pairing["graph_metrics"]["oracle_node_duplication_rate"] == 0.0
    assert pairing["graph_metrics"]["predicted_node_duplication_rate"] == 0.0
    html = (output_dir / "report.html").read_text(encoding="utf-8")
    assert "Oracle node dup. rate" in html
    assert "Predicted node dup. rate" in html


# --- shared fakes for the orchestration-level tests below ---


class _CountingResolver:
    """Records how many times resolve() is called, and with which documents."""

    language_support = {"en"}

    def __init__(self, name, clusters_factory=None):
        self.name = name
        self.calls = []
        self._clusters_factory = clusters_factory

    def resolve(self, doc):
        from pipeline.types import ResolverOutput

        self.calls.append(doc.doc_id)
        clusters = self._clusters_factory(doc) if self._clusters_factory else None
        # Tagged with the resolver name so a predicted text can never be
        # mistaken for the oracle text when counting build() calls.
        resolved = doc.text.replace("He", "John").replace("She", "Mary")
        return ResolverOutput(resolved_text=f"{resolved} {self.name}", clusters=clusters)


class _CountingBackend:
    """Records every text passed to build()."""

    language_support = {"en"}

    def __init__(self, name):
        self.name = name
        self.backend_name = "RuleBased"  # keeps compute_graph_scores on the flat-node shape
        self.built_texts = []

    def build(self, text):
        self.built_texts.append(text)
        words = sorted({w.strip(".").lower() for w in text.split() if w.strip(".")})
        return {
            "nodes": words,
            "edges": [
                {"agent_1": words[i], "agent_2": words[i + 1], "meaning": "next", "weight": 1}
                for i in range(len(words) - 1)
            ],
        }


TWO_DOC_CORPUS = """\
# global.Entity = eid-etype-head-other
# newdoc id = docA
# sent_id = docA-1
1\tJohn\tJohn\tPROPN\t_\t_\t2\tnsubj\t_\tEntity=(e1-person-1-new)
2\tarrived\tarrive\tVERB\t_\t_\t0\troot\t_\tSpaceAfter=No
3\t.\t.\tPUNCT\t_\t_\t2\tpunct\t_\t_

# sent_id = docA-2
1\tHe\the\tPRON\t_\t_\t2\tnsubj\t_\tEntity=(e1)
2\tleft\tleave\tVERB\t_\t_\t0\troot\t_\tSpaceAfter=No
3\t.\t.\tPUNCT\t_\t_\t2\tpunct\t_\t_

# newdoc id = docB
# sent_id = docB-1
1\tMary\tMary\tPROPN\t_\t_\t2\tnsubj\t_\tEntity=(e2-person-1-new)
2\tsmiled\tsmile\tVERB\t_\t_\t0\troot\t_\tSpaceAfter=No
3\t.\t.\tPUNCT\t_\t_\t2\tpunct\t_\t_

# sent_id = docB-2
1\tShe\tshe\tPRON\t_\t_\t2\tnsubj\t_\tEntity=(e2)
2\twaved\twave\tVERB\t_\t_\t0\troot\t_\tSpaceAfter=No
3\t.\t.\tPUNCT\t_\t_\t2\tpunct\t_\t_
"""


def _write_two_doc_corpus(tmp_path):
    path = tmp_path / "two_docs.conllu"
    path.write_text(TWO_DOC_CORPUS, encoding="utf-8")
    return path


def _patch_factories(monkeypatch, resolvers_by_name, backends_by_name):
    import pipeline.run_experiment as run_experiment_module

    monkeypatch.setattr(
        run_experiment_module, "_build_resolver",
        lambda name, language, llm_client=None: resolvers_by_name[name],
    )
    monkeypatch.setattr(
        run_experiment_module, "_build_graph_backend",
        lambda name, language, llm_client=None, embedder=None: backends_by_name[name],
    )


# --- Finding 5: resolve() and the oracle graph must not be recomputed R x B ---


def test_resolve_runs_once_per_resolver_document_and_oracle_graph_once_per_backend_document(
    monkeypatch, tmp_path
):
    _install_fake_graph_metrics_module(monkeypatch)

    resolvers = {"R1": _CountingResolver("R1"), "R2": _CountingResolver("R2")}
    backends = {"B1": _CountingBackend("B1"), "B2": _CountingBackend("B2")}
    _patch_factories(monkeypatch, resolvers, backends)

    run_experiment(
        corpus_path=_write_two_doc_corpus(tmp_path),
        language="en",
        resolver_names=["R1", "R2"],
        graph_backend_names=["B1", "B2"],
        output_dir=tmp_path / "run",
    )

    # 2 documents, 2 backends: resolve() must run 2x per resolver, not 4x.
    for resolver in resolvers.values():
        assert resolver.calls == ["docA", "docB"], resolver.calls

    # Per backend: 2 oracle builds (one per document, shared across resolvers)
    # + 4 predicted builds (2 resolvers x 2 documents) = 6, not 8.
    oracle_texts = {"John arrived. John left.", "Mary smiled. Mary waved."}
    for backend in backends.values():
        assert len(backend.built_texts) == 6, backend.built_texts
        for oracle_text in oracle_texts:
            assert backend.built_texts.count(oracle_text) == 1


def test_restructured_loop_produces_the_same_results_as_before(monkeypatch, tmp_path):
    """The hoisting must be a pure performance/determinism change."""
    _install_fake_resolver_module(monkeypatch)
    _install_fake_graph_builder_module(monkeypatch)
    _install_fake_graph_metrics_module(monkeypatch)

    import json

    outputs = []
    for run_name in ("runA", "runB"):
        path = run_experiment(
            corpus_path=FIXTURE_CORPUS,
            language="en",
            resolver_names=["LapinLiass"],
            graph_backend_names=["RuleBased"],
            output_dir=tmp_path / run_name,
        )
        data = json.loads(path.read_text(encoding="utf-8"))
        data.pop("generated_at")
        data.pop("run_id")
        outputs.append(data)
    assert outputs[0] == outputs[1]


# --- Finding 8: a language mismatch must fail loudly, not write an empty run ---


def test_language_mismatch_raises_instead_of_writing_an_empty_report(tmp_path):
    output_dir = tmp_path / "run_ru"
    try:
        run_experiment(
            corpus_path=FIXTURE_CORPUS,
            language="ru",
            resolver_names=["LapinLiass"],  # English-only
            graph_backend_names=["RuleBased"],  # English-only
            output_dir=output_dir,
        )
    except RuntimeError as exc:
        message = str(exc)
        assert "'ru'" in message
        assert "LapinLiass" in message
        assert "RuleBased" in message
    else:
        raise AssertionError("expected RuntimeError for a language with no viable pairing")

    assert not (output_dir / "results.json").exists()


def test_language_mismatch_on_the_backend_alone_also_raises(monkeypatch, tmp_path):
    """A resolver that supports the language is not enough on its own."""
    import pipeline.run_experiment as run_experiment_module

    any_language_resolver = _CountingResolver("Any")
    any_language_resolver.language_support = "any"
    monkeypatch.setattr(
        run_experiment_module, "_build_resolver",
        lambda name, language, llm_client=None: any_language_resolver,
    )

    try:
        run_experiment(
            corpus_path=FIXTURE_CORPUS,
            language="ru",
            resolver_names=["Any"],
            graph_backend_names=["RuleBased"],  # English-only
            output_dir=tmp_path / "run_ru2",
        )
    except RuntimeError as exc:
        assert "RuleBased" in str(exc)
    else:
        raise AssertionError("expected RuntimeError when every graph backend is filtered out")
    assert any_language_resolver.calls == []  # nothing was run before failing


# --- Finding 9: the clusters=None path, end to end through run_experiment ---


def test_resolver_returning_none_clusters_flows_through_to_na_in_the_report(
    monkeypatch, tmp_path
):
    """The design's headline asymmetry (LLMv2 has no mention spans) as one path.

    clusters=None must reach results.json as coreference_metrics=None for both
    the per-document entry and the pairing average, and render as "N/A" in the
    HTML -- previously only covered by isolated unit tests of each piece.
    """
    _install_fake_graph_builder_module(monkeypatch)
    _install_fake_graph_metrics_module(monkeypatch)

    import pipeline.run_experiment as run_experiment_module

    text_only_resolver = _CountingResolver("TextOnly")  # clusters_factory=None -> clusters=None
    monkeypatch.setattr(
        run_experiment_module, "_build_resolver",
        lambda name, language, llm_client=None: text_only_resolver,
    )

    output_dir = tmp_path / "run_text_only"
    results_path = run_experiment(
        corpus_path=FIXTURE_CORPUS,
        language="en",
        resolver_names=["TextOnly"],
        graph_backend_names=["RuleBased"],  # the real RuleBasedGraphAdapter, over the fake graphBuilder
        output_dir=output_dir,
    )

    import json

    data = json.loads(results_path.read_text(encoding="utf-8"))
    pairing = data["results"][0]
    assert pairing["coreference_metrics"] is None
    assert [d["coreference_metrics"] for d in pairing["documents"]] == [None]
    assert pairing["graph_metrics"]["node_precision_recall_f1"]["f1"] is not None

    html = (output_dir / "report.html").read_text(encoding="utf-8")
    row = next(line for line in html.splitlines() if "TextOnly" in line)
    assert "<td>N/A</td>" in row  # the CoNLL F1 cell for this pairing
