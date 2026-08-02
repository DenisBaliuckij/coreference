import sys
import types
from dataclasses import dataclass

from pipeline.resolvers.spacy_neural_adapter import SpacyNeuralAdapter
from pipeline.types import CorefDocument, MentionSpan, Token


@dataclass
class _FakeSpan:
    start_char: int
    end_char: int
    text: str


class _FakeSpacyDoc:
    def __init__(self, spans_dict):
        self.spans = spans_dict


def _fake_nlp(text):
    # cluster: "John" (0-4) and "He" (14-16) corefer
    return _FakeSpacyDoc({
        "coref_clusters_1": [_FakeSpan(0, 4, "John"), _FakeSpan(14, 16, "He")],
    })


def _fake_resolve_and_substitute(text, mark=False):
    resolved = text[:14] + "John" + text[16:]
    return resolved, [], []


def _install_fake_module(monkeypatch):
    fake = types.ModuleType("anaphoraResolverSpacyNeural")
    fake.resolve_and_substitute = _fake_resolve_and_substitute
    fake._get_nlp = lambda: _fake_nlp
    monkeypatch.setitem(sys.modules, "anaphoraResolverSpacyNeural", fake)


def test_resolve_builds_clusters_from_native_coref_spans(monkeypatch):
    _install_fake_module(monkeypatch)

    text = "John arrived. He left."
    gold_tokens = [
        Token(0, "John", 0, 4, 0), Token(1, "arrived", 5, 12, 0), Token(2, ".", 12, 13, 0),
        Token(3, "He", 14, 16, 1), Token(4, "left", 17, 21, 1), Token(5, ".", 21, 22, 1),
    ]
    doc = CorefDocument(doc_id="d1", text=text, tokens=gold_tokens, clusters=[])

    adapter = SpacyNeuralAdapter()
    result = adapter.resolve(doc)

    assert result.resolved_text == "John arrived. John left."
    assert result.clusters == [[MentionSpan(0, 0), MentionSpan(3, 3)]]


def test_adapter_metadata():
    adapter = SpacyNeuralAdapter()
    assert adapter.name == "SpacyNeural"
    assert adapter.language_support == {"en"}
