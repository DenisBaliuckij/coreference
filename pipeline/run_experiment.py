from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .corpus.corefud_loader import load_corefud_corpus
from .corpus.oracle import build_oracle_text
from .eval.coref_scoring import score_coreference
from .eval.graph_scoring import compute_graph_scores
from .graph.llm_v2_graph_adapter import LLMv2GraphAdapter
from .graph.rule_based_graph_adapter import RuleBasedGraphAdapter
from .report import build_results, render_html_report, save_html_report, save_results_json
from .resolvers.base import supports_language
from .resolvers.lapin_liass_adapter import LapinLiassAdapter
from .resolvers.llm_v2_adapter import LLMv2Adapter
from .resolvers.spacy_neural_adapter import SpacyNeuralAdapter


def _build_resolver(name: str, language: str, llm_client=None):
    if name == "LapinLiass":
        return LapinLiassAdapter()
    if name == "SpacyNeural":
        return SpacyNeuralAdapter()
    if name == "LLMv2":
        if llm_client is None:
            raise ValueError("LLMv2 resolver requires an llm_client (see --llm-model)")
        return LLMv2Adapter(llm_client=llm_client, language=language)
    raise ValueError(f"unknown resolver: {name}")


def _build_graph_backend(name: str, language: str, llm_client=None, embedder=None):
    if name == "RuleBased":
        return RuleBasedGraphAdapter()
    if name == "LLMv2":
        if llm_client is None or embedder is None:
            raise ValueError("LLMv2 graph backend requires an llm_client and embedder (see --llm-model)")
        return LLMv2GraphAdapter(llm_client=llm_client, embedder=embedder, language=language)
    raise ValueError(f"unknown graph backend: {name}")


def _average_coref(doc_entries: list[dict]) -> dict | None:
    values = [e["coreference_metrics"] for e in doc_entries if e["coreference_metrics"] is not None]
    if not values:
        return None
    return {"conll_f1": sum(v["conll_f1"] for v in values) / len(values)}


def _average_graph(doc_entries: list[dict]) -> dict:
    n = len(doc_entries)
    node_f1s = [e["graph_metrics"]["node_precision_recall_f1"]["f1"] for e in doc_entries]
    edge_f1s = [e["graph_metrics"]["edge_precision_recall_f1"]["f1"] for e in doc_entries]
    smatch_f1s = [e["graph_metrics"]["smatch"]["f1"] for e in doc_entries]
    oracle_dups = [e["graph_metrics"]["oracle_node_duplication_rate"] for e in doc_entries]
    predicted_dups = [e["graph_metrics"]["predicted_node_duplication_rate"] for e in doc_entries]
    return {
        "node_precision_recall_f1": {"f1": sum(node_f1s) / n},
        "edge_precision_recall_f1": {"f1": sum(edge_f1s) / n},
        "smatch": {"f1": sum(smatch_f1s) / n},
        # Central to the design spec's oracle-ablation "error budget" framing:
        # aggregated here (and rendered by report.py) so it is visible without
        # digging through the per-document entries in results.json.
        "oracle_node_duplication_rate": sum(oracle_dups) / n,
        "predicted_node_duplication_rate": sum(predicted_dups) / n,
    }


def run_experiment(
    corpus_path: Path,
    language: str,
    resolver_names: list[str],
    graph_backend_names: list[str],
    output_dir: Path,
    llm_client=None,
    embedder=None,
) -> Path:
    documents = load_corefud_corpus(corpus_path)

    resolvers, skipped_resolvers = [], []
    for resolver_name in resolver_names:
        resolver = _build_resolver(resolver_name, language, llm_client=llm_client)
        if supports_language(resolver, language):
            resolvers.append((resolver_name, resolver))
        else:
            skipped_resolvers.append(resolver_name)

    backends, skipped_backends = [], []
    for backend_name in graph_backend_names:
        graph_backend = _build_graph_backend(
            backend_name, language, llm_client=llm_client, embedder=embedder
        )
        if supports_language(graph_backend, language):
            backends.append((backend_name, graph_backend))
        else:
            skipped_backends.append(backend_name)

    if not resolvers or not backends:
        # A silently empty results.json plus an exit code of 0 is the worst
        # possible outcome for a research tool: fail loudly instead.
        raise RuntimeError(
            f"no resolver x graph-backend pairing supports language {language!r}: "
            f"resolvers filtered out: {skipped_resolvers or 'none'}; "
            f"graph backends filtered out: {skipped_backends or 'none'}. "
            f"Requested resolvers={resolver_names}, graph_backends={graph_backend_names}."
        )

    # build_oracle_text and resolver.resolve depend only on the document (and,
    # for the latter, the resolver) -- never on the graph backend. Computing
    # them inside the backend loop wasted R x B work and, worse, let a
    # non-deterministic resolver (LLMv2) hand a *different* resolved text to
    # each backend, breaking the apples-to-apples comparison the report implies.
    oracle_texts = [build_oracle_text(doc) for doc in documents]

    # Keyed by (backend_name, document position): the oracle graph depends on
    # the backend and the document only, so it is built once per pair and
    # reused across every resolver.
    oracle_graph_cache: dict[tuple[str, int], dict] = {}

    per_pairing = []
    for resolver_name, resolver in resolvers:
        # Exactly one resolve() call per (resolver, document).
        resolver_outputs = [resolver.resolve(doc) for doc in documents]

        for backend_name, graph_backend in backends:
            doc_entries = []
            for doc_index, doc in enumerate(documents):
                resolver_output = resolver_outputs[doc_index]

                cache_key = (backend_name, doc_index)
                if cache_key not in oracle_graph_cache:
                    oracle_graph_cache[cache_key] = graph_backend.build(oracle_texts[doc_index])
                oracle_graph = oracle_graph_cache[cache_key]

                # Genuinely depends on both resolver and backend: computed R x B.
                predicted_graph = graph_backend.build(resolver_output.resolved_text)

                coref_metrics = None
                if resolver_output.clusters is not None:
                    coref_metrics = score_coreference(doc.clusters, resolver_output.clusters)

                graph_metrics = compute_graph_scores(oracle_graph, predicted_graph, graph_backend.backend_name)

                doc_entries.append({
                    "doc_id": doc.doc_id,
                    "coreference_metrics": coref_metrics,
                    "graph_metrics": graph_metrics,
                })

            per_pairing.append({
                "resolver": resolver_name,
                "graph_backend": backend_name,
                "documents": doc_entries,
                "coreference_metrics": _average_coref(doc_entries),
                "graph_metrics": _average_graph(doc_entries),
            })

    output_dir = Path(output_dir)
    run_id = output_dir.name
    results = build_results(run_id, language, per_pairing)
    results_path = save_results_json(results, output_dir)
    html = render_html_report(results)
    save_html_report(html, output_dir)
    return results_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cascade-architecture experimental stand")
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--language", required=True)
    parser.add_argument("--resolvers", required=True, help="comma-separated resolver names")
    parser.add_argument("--graph-backends", required=True, help="comma-separated graph backend names")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--llm-model", default=None, help="HF model name; required for the LLMv2 resolver/backend")
    args = parser.parse_args(argv)

    resolver_names = args.resolvers.split(",")
    backend_names = args.graph_backends.split(",")

    llm_client = None
    embedder = None
    if "LLMv2" in resolver_names or "LLMv2" in backend_names:
        from .sys_path_setup import add_text_corpuses_processing_to_path

        add_text_corpuses_processing_to_path()
        from llm_v2.config_schema import EmbeddingConfig, LLMConfig
        from llm_v2.models.embedder import Embedder
        from llm_v2.models.llm_client import LLMClient

        llm_client = LLMClient(LLMConfig(model_name=args.llm_model or "Qwen/Qwen2-1.5B-Instruct"))
        embedder = Embedder(EmbeddingConfig())

    results_path = run_experiment(
        corpus_path=args.corpus,
        language=args.language,
        resolver_names=resolver_names,
        graph_backend_names=backend_names,
        output_dir=args.output,
        llm_client=llm_client,
        embedder=embedder,
    )
    print(f"Results written to {results_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
