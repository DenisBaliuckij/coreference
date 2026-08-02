import sys
import types
from dataclasses import dataclass
from typing import Optional

from pipeline.resolvers.lapin_liass_adapter import LapinLiassAdapter
from pipeline.types import CorefDocument, MentionSpan, Token


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
        # "John arrived. He left." -> spaCy tokens (word-level, char offsets into `text`)
        tokens = [
            _FakeSpacyToken(0, "John"), _FakeSpacyToken(5, "arrived"), _FakeSpacyToken(12, "."),
            _FakeSpacyToken(14, "He"), _FakeSpacyToken(17, "left"), _FakeSpacyToken(21, "."),
        ]
        doc = _FakeSpacyDoc(tokens)
        resolutions = [
            _FakeResolution(pronoun="He", pronoun_index=3, antecedent="John", antecedent_index=0, score=100.0)
        ]
        return {"text": text, "doc": doc, "resolutions": resolutions}


def _fake_build_substitutions(doc, resolutions, mark=False):
    subs = []
    for r in resolutions:
        if r.antecedent is None:
            continue
        tok = doc[r.pronoun_index]
        subs.append(types.SimpleNamespace(
            start=tok.idx, end=tok.idx + len(tok.text), replacement=r.antecedent,
        ))
    subs.sort(key=lambda s: s.start, reverse=True)
    return subs


def _fake_apply_substitutions(text, substitutions):
    out = text
    for s in substitutions:
        out = out[: s.start] + s.replacement + out[s.end :]
    return out


def _install_fake_module(monkeypatch):
    fake = types.ModuleType("anaphoraResolverLapinLiass")
    fake.BatchAnaphoraResolver = _FakeBatchAnaphoraResolver
    fake.build_substitutions = _fake_build_substitutions
    fake.apply_substitutions = _fake_apply_substitutions
    monkeypatch.setitem(sys.modules, "anaphoraResolverLapinLiass", fake)


def test_resolve_projects_clusters_onto_gold_tokens_and_substitutes_text(monkeypatch):
    _install_fake_module(monkeypatch)

    text = "John arrived. He left."
    gold_tokens = [
        Token(0, "John", 0, 4, 0), Token(1, "arrived", 5, 12, 0), Token(2, ".", 12, 13, 0),
        Token(3, "He", 14, 16, 1), Token(4, "left", 17, 21, 1), Token(5, ".", 21, 22, 1),
    ]
    doc = CorefDocument(doc_id="d1", text=text, tokens=gold_tokens, clusters=[])

    adapter = LapinLiassAdapter()
    result = adapter.resolve(doc)

    assert result.resolved_text == "John arrived. John left."
    assert result.clusters == [[MentionSpan(0, 0), MentionSpan(3, 3)]]


def test_adapter_metadata():
    adapter = LapinLiassAdapter()
    assert adapter.name == "LapinLiass"
    assert adapter.language_support == {"en"}
