import sys
import types

from pipeline.resolvers.llm_v2_adapter import LLMv2Adapter
from pipeline.types import CorefDocument, Token


class _FakeLLMClient:
    def generate(self, prompt):
        return "unused"


def _install_fake_modules(monkeypatch, captured):
    preprocessing_mod = types.ModuleType("llm_v2.stages.preprocessing")

    def _fake_preprocess(text, language="ru"):
        captured["preprocess_language"] = language
        return [types.SimpleNamespace(id=0, text=text)]

    preprocessing_mod.preprocess = _fake_preprocess

    coreference_mod = types.ModuleType("llm_v2.stages.coreference")

    def _fake_resolve_coreferences(sentences, llm, config, base_dir=None):
        captured["config_prompt_file"] = config.prompt_file
        captured["base_dir"] = base_dir
        return "John arrived. John left.", sentences

    coreference_mod.resolve_coreferences = _fake_resolve_coreferences

    config_schema_mod = types.ModuleType("llm_v2.config_schema")

    class _FakeCoreferenceConfig:
        def __init__(self, enabled=True, prompt_file=""):
            self.enabled = enabled
            self.prompt_file = prompt_file

    config_schema_mod.CoreferenceConfig = _FakeCoreferenceConfig

    monkeypatch.setitem(sys.modules, "llm_v2", types.ModuleType("llm_v2"))
    monkeypatch.setitem(sys.modules, "llm_v2.stages", types.ModuleType("llm_v2.stages"))
    monkeypatch.setitem(sys.modules, "llm_v2.stages.preprocessing", preprocessing_mod)
    monkeypatch.setitem(sys.modules, "llm_v2.stages.coreference", coreference_mod)
    monkeypatch.setitem(sys.modules, "llm_v2.config_schema", config_schema_mod)


def test_resolve_returns_resolved_text_and_none_clusters(monkeypatch):
    captured: dict = {}
    _install_fake_modules(monkeypatch, captured)

    doc = CorefDocument(
        doc_id="d1", text="John arrived. He left.",
        tokens=[Token(0, "John", 0, 4, 0)], clusters=[],
    )
    adapter = LLMv2Adapter(llm_client=_FakeLLMClient(), language="en")
    result = adapter.resolve(doc)

    assert result.resolved_text == "John arrived. John left."
    assert result.clusters is None
    assert captured["preprocess_language"] == "en"
    assert captured["config_prompt_file"] == "prompts/coreference_en.txt"


def test_adapter_metadata():
    adapter = LLMv2Adapter(llm_client=_FakeLLMClient(), language="en")
    assert adapter.name == "LLMv2"
    assert adapter.language_support == "any"
