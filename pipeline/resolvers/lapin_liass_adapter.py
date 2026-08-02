from __future__ import annotations

from ..sys_path_setup import add_text_corpuses_processing_to_path
from ..types import CorefDocument, MentionSpan, ResolverOutput
from .base import UnionFind, project_char_span_to_gold_tokens


class LapinLiassAdapter:
    name = "LapinLiass"
    language_support = {"en"}

    def resolve(self, doc: CorefDocument) -> ResolverOutput:
        add_text_corpuses_processing_to_path()
        from anaphoraResolverLapinLiass import (
            BatchAnaphoraResolver,
            apply_substitutions,
            build_substitutions,
        )

        resolver = BatchAnaphoraResolver()
        result = resolver.resolve_document(doc.text)
        spacy_doc = result["doc"]
        resolutions = result["resolutions"]

        substitutions = build_substitutions(spacy_doc, resolutions, mark=False)
        resolved_text = apply_substitutions(doc.text, substitutions)

        uf = UnionFind()
        for r in resolutions:
            if r.antecedent_index is not None:
                uf.union(r.pronoun_index, r.antecedent_index)

        clusters: list[list[MentionSpan]] = []
        for group in uf.groups():
            if len(group) < 2:
                continue
            projected = []
            for token_idx in group:
                tok = spacy_doc[token_idx]
                span = project_char_span_to_gold_tokens(tok.idx, tok.idx + len(tok.text), doc.tokens)
                if span is not None:
                    projected.append(span)
            if len(projected) >= 2:
                # uf.groups() yields insertion order, not token-position order; sort for determinism
                projected.sort(key=lambda s: s.start_token)
                clusters.append(projected)

        return ResolverOutput(resolved_text=resolved_text, clusters=clusters)
