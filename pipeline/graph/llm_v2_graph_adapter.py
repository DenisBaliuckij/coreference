from __future__ import annotations

from pathlib import Path

from ..sys_path_setup import add_text_corpuses_processing_to_path


class LLMv2GraphAdapter:
    name = "LLMv2"
    backend_name = "LLMv2"
    language_support = "any"

    def __init__(self, llm_client, embedder, language: str, base_dir: Path | None = None):
        self.llm_client = llm_client
        self.embedder = embedder
        self.language = language
        dags_dir = add_text_corpuses_processing_to_path()
        self.base_dir = base_dir or (dags_dir / "llm_v2")

    def build(self, text: str) -> dict:
        add_text_corpuses_processing_to_path()
        from llm_v2.config_schema import (
            DeduplicationConfig,
            ExtractionConfig,
            NormalizationConfig,
            PipelineConfig,
        )
        from llm_v2.stages.chunking import build_chunks
        from llm_v2.stages.deduplication import deduplicate_triplets
        from llm_v2.stages.extraction import extract_triplets
        from llm_v2.stages.graph_assembly import assemble_graph
        from llm_v2.stages.normalization import normalize_triplets
        from llm_v2.stages.preprocessing import preprocess

        extraction_config = ExtractionConfig(prompt_file=f"prompts/extraction_{self.language}.txt")
        normalization_config = NormalizationConfig(language=self.language)
        deduplication_config = DeduplicationConfig()
        pipeline_config = PipelineConfig(
            extraction=extraction_config,
            normalization=normalization_config,
            deduplication=deduplication_config,
        )

        sentences = preprocess(text, language=self.language)
        chunks = build_chunks(sentences, extraction_config)
        raw_triplets = extract_triplets(chunks, self.llm_client, extraction_config, base_dir=self.base_dir)
        norm_triplets = normalize_triplets(raw_triplets, normalization_config)
        dedup_triplets = deduplicate_triplets(norm_triplets, self.embedder, deduplication_config)
        raw_graph = assemble_graph(dedup_triplets, chunks, text, pipeline_config)
        return raw_graph.model_dump()
