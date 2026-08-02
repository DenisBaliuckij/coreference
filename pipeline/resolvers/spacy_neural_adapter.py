from __future__ import annotations

from ..sys_path_setup import add_text_corpuses_processing_to_path
from ..types import CorefDocument, MentionSpan, ResolverOutput
from .base import project_char_span_to_gold_tokens


class SpacyNeuralAdapter:
    name = "SpacyNeural"
    language_support = {"en"}

    def resolve(self, doc: CorefDocument) -> ResolverOutput:
        add_text_corpuses_processing_to_path()
        from anaphoraResolverSpacyNeural import _get_nlp, resolve_and_substitute

        resolved_text, _, _ = resolve_and_substitute(doc.text)

        nlp = _get_nlp()
        spacy_doc = nlp(doc.text)
        spacy_clusters = [
            spans for key, spans in spacy_doc.spans.items()
            if key.startswith("coref_clusters")
        ]

        clusters: list[list[MentionSpan]] = []
        for cluster in spacy_clusters:
            if len(cluster) < 2:
                continue
            projected = []
            for mention_span in cluster:
                span = project_char_span_to_gold_tokens(
                    mention_span.start_char, mention_span.end_char, doc.tokens
                )
                if span is not None:
                    projected.append(span)
            if len(projected) >= 2:
                clusters.append(projected)

        return ResolverOutput(resolved_text=resolved_text, clusters=clusters)
