from __future__ import annotations

from pathlib import Path

from ..sys_path_setup import add_text_corpuses_processing_to_path
from ..types import CorefDocument, ResolverOutput


class LLMv2Adapter:
    name = "LLMv2"
    language_support = "any"

    def __init__(self, llm_client, language: str, base_dir: Path | None = None):
        self.llm_client = llm_client
        self.language = language
        dags_dir = add_text_corpuses_processing_to_path()
        self.base_dir = base_dir or (dags_dir / "llm_v2")

    def resolve(self, doc: CorefDocument) -> ResolverOutput:
        add_text_corpuses_processing_to_path()
        from llm_v2.config_schema import CoreferenceConfig
        from llm_v2.stages.coreference import resolve_coreferences
        from llm_v2.stages.preprocessing import preprocess

        sentences = preprocess(doc.text, language=self.language)
        config = CoreferenceConfig(
            enabled=True,
            prompt_file=f"prompts/coreference_{self.language}.txt",
        )
        resolved_text, _ = resolve_coreferences(
            sentences, self.llm_client, config, base_dir=self.base_dir
        )
        # No mention-span output from this resolver (it only rewrites text) --
        # clusters=None means coreference F1 is not computable for this
        # adapter; only the graph-level oracle-ablation applies. See design
        # spec Section 2.
        return ResolverOutput(resolved_text=resolved_text, clusters=None)
