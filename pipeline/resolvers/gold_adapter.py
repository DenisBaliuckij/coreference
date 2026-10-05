from __future__ import annotations

from ..corpus.oracle import build_oracle_text
from ..types import CorefDocument, ResolverOutput


class GoldAdapter:
    """Control: a "resolver" that returns the gold answer -- the oracle text and the gold clusters.

    Its coreference score must be 1.0 (a check of the scoring path), and its graph is the oracle
    text extracted a second time, so its oracle-vs-predicted graph scores are the noise floor of
    the graph backend: the best any resolver can reach when the backend itself is not perfectly
    repeatable (LLM extraction). Real resolvers are read against this ceiling, not against 1.0.
    """

    name = "Gold"
    language_support = "any"

    def __init__(self, oracle_mode: str = "all", language: str = "en"):
        self.oracle_mode = oracle_mode
        self.language = language

    def resolve(self, doc: CorefDocument) -> ResolverOutput:
        return ResolverOutput(resolved_text=build_oracle_text(doc, mode=self.oracle_mode, language=self.language), clusters=[list(c) for c in doc.clusters])
