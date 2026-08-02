import sys
import types

from pipeline.graph.llm_v2_graph_adapter import LLMv2GraphAdapter


class _FakeLLMClient:
    def generate(self, prompt):
        return "obama | born in | hawaii"


class _FakeEmbedder:
    pass


def _install_fake_modules(monkeypatch, captured):
    def _mod(name):
        m = types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, m)
        return m

    _mod("llm_v2")
    _mod("llm_v2.stages")

    preprocessing = _mod("llm_v2.stages.preprocessing")
    preprocessing.preprocess = lambda text, language="ru": [types.SimpleNamespace(id=0, text=text)]

    chunking = _mod("llm_v2.stages.chunking")
    chunking.build_chunks = lambda sentences, config: [
        types.SimpleNamespace(id="chunk_0", text=sentences[0].text, sentence_ids=[0])
    ]

    extraction = _mod("llm_v2.stages.extraction")

    def _extract_triplets(chunks, llm, config, base_dir=None):
        captured["extraction_base_dir"] = base_dir
        return [types.SimpleNamespace(subject="obama", relation="born in", object="hawaii", chunk_id="chunk_0")]

    extraction.extract_triplets = _extract_triplets

    normalization = _mod("llm_v2.stages.normalization")
    normalization.normalize_triplets = lambda triplets, config: [
        types.SimpleNamespace(
            subject=t.subject, relation=t.relation, object=t.object, chunk_id=t.chunk_id,
            norm_subject=t.subject, norm_relation=t.relation, norm_object=t.object,
        )
        for t in triplets
    ]

    deduplication = _mod("llm_v2.stages.deduplication")
    deduplication.deduplicate_triplets = lambda triplets, embedder, config: list(triplets)

    graph_assembly = _mod("llm_v2.stages.graph_assembly")

    class _FakeRawGraph:
        def __init__(self, nodes, edges):
            self._nodes = nodes
            self._edges = edges

        def model_dump(self):
            return {"meta": {}, "chunks": [], "nodes": self._nodes, "edges": self._edges}

    def _assemble_graph(triplets, chunks, source_text, config):
        nodes = []
        edges = []
        seen = {}
        for t in triplets:
            for label in (t.norm_subject, t.norm_object):
                if label not in seen:
                    seen[label] = f"n{len(nodes)}"
                    nodes.append({"id": seen[label], "label": label})
            edges.append({
                "id": f"e{len(edges)}", "source": seen[t.norm_subject], "target": seen[t.norm_object],
                "label": t.norm_relation, "weight": 1,
            })
        return _FakeRawGraph(nodes, edges)

    graph_assembly.assemble_graph = _assemble_graph

    config_schema = _mod("llm_v2.config_schema")
    for cls_name in ("ExtractionConfig", "NormalizationConfig", "DeduplicationConfig", "PipelineConfig"):
        def _make_init(name):
            def _init(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
                if name == "ExtractionConfig" and "prompt_file" not in kwargs:
                    self.prompt_file = ""
                if name == "NormalizationConfig" and "language" not in kwargs:
                    self.language = "en"
            return _init
        setattr(config_schema, cls_name, type(cls_name, (), {"__init__": _make_init(cls_name)}))


def test_build_returns_raw_graph_dict_shape(monkeypatch):
    captured: dict = {}
    _install_fake_modules(monkeypatch, captured)

    adapter = LLMv2GraphAdapter(llm_client=_FakeLLMClient(), embedder=_FakeEmbedder(), language="en")
    graph = adapter.build("Obama was born in Hawaii.")

    assert graph["nodes"] == [{"id": "n0", "label": "obama"}, {"id": "n1", "label": "hawaii"}]
    assert graph["edges"] == [{"id": "e0", "source": "n0", "target": "n1", "label": "born in", "weight": 1}]


def test_adapter_metadata():
    adapter = LLMv2GraphAdapter(llm_client=_FakeLLMClient(), embedder=_FakeEmbedder(), language="en")
    assert adapter.name == "LLMv2"
    assert adapter.backend_name == "LLMv2"
    assert adapter.language_support == "any"
