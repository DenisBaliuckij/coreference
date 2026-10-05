from __future__ import annotations

from ..types import CorefDocument, ResolverOutput


class NoResolutionAdapter:
    """Lower-bound baseline: the text is passed on unchanged and no coreference is predicted.

    The graph built from it shows what the graph backend makes of the raw text, so the
    distance from it to the oracle graph is the whole coreference "error budget", and
    every real resolver can be placed between the two. ``clusters=None``: predicting no
    links is not a coreference system worth a CoNLL F1 of 0 in the tables.
    """

    name = "NoResolution"
    language_support = "any"

    def resolve(self, doc: CorefDocument) -> ResolverOutput:
        return ResolverOutput(resolved_text=doc.text, clusters=None)
