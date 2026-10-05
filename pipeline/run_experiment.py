from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .corpus.corefud_loader import load_corefud_corpus
from .corpus.oracle import build_oracle_text
from .eval.corefud_scoring import MATCH_MODES, normalized_key, score_corpus, write_response
from .eval.graph_scoring import compute_graph_scores
from .graph.llm_v2_graph_adapter import LLMv2GraphAdapter
from .graph.rule_based_graph_adapter import RuleBasedGraphAdapter
from .report import build_results, render_html_report, save_html_report, save_results_json
from .resolvers.base import supports_language
from .types import ResolverOutput
from .resolvers.lapin_liass_adapter import LapinLiassAdapter
from .resolvers.llm_v2_adapter import LLMv2Adapter
from .resolvers.gold_adapter import GoldAdapter
from .resolvers.no_resolution_adapter import NoResolutionAdapter
from .resolvers.spacy_neural_adapter import SpacyNeuralAdapter


def _build_resolver(name: str, language: str, llm_client=None):
    if name == "NoResolution":
        return NoResolutionAdapter()
    if name == "Gold":
        return GoldAdapter()
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


def _read_key_text(corpus_path: Path) -> str:
    """The gold CoNLL-U exactly as the loader reads it (a directory = its *.conllu, sorted)."""
    corpus_path = Path(corpus_path)
    if corpus_path.is_dir():
        return "\n".join(p.read_text(encoding="utf-8").rstrip("\n") + "\n"
                         for p in sorted(corpus_path.glob("*.conllu")))
    return corpus_path.read_text(encoding="utf-8")


def _score_resolver(resolver_name, documents, resolver_outputs, key_path, key_text, coref_dir, match):
    """Corpus-level CorefUD score of one resolver, or None if it reports no clusters."""
    if all(o.clusters is None for o in resolver_outputs):
        return None
    predicted = {doc.doc_id: out.clusters for doc, out in zip(documents, resolver_outputs)}
    response_text, dropped = write_response(key_text, predicted)
    response_path = coref_dir / f"{resolver_name}.response.conllu"
    response_path.write_text(response_text, encoding="utf-8")
    metrics = score_corpus(key_path, response_path, match=match)
    metrics["dropped_cross_sentence_mentions"] = dropped
    return metrics


def _average_graph(doc_entries: list[dict]) -> dict:
    n = len(doc_entries)
    node_f1s = [e["graph_metrics"]["node_precision_recall_f1"]["f1"] for e in doc_entries]
    edge_f1s = [e["graph_metrics"]["edge_precision_recall_f1"]["f1"] for e in doc_entries]
    triple_f1s = [e["graph_metrics"]["triple_f1"]["f1"] for e in doc_entries]
    oracle_dups = [e["graph_metrics"]["oracle_node_duplication_rate"] for e in doc_entries]
    predicted_dups = [e["graph_metrics"]["predicted_node_duplication_rate"] for e in doc_entries]
    return {
        "node_precision_recall_f1": {"f1": sum(node_f1s) / n},
        "edge_precision_recall_f1": {"f1": sum(edge_f1s) / n},
        "triple_f1": {"f1": sum(triple_f1s) / n},
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
    coref_match: str = "exact",
    save_artifacts: bool = False,
    resume: bool = False,
) -> Path:
    """... ``save_artifacts`` keeps every resolved text and graph under ``<output>/artifacts/``
    (texts/<resolver or ORACLE>/<doc>.txt, graphs/<backend>/<resolver or ORACLE>/<doc>.json)
    for error analysis. ``resume`` (needs ``save_artifacts``) reuses the graphs and text-only
    resolver texts an interrupted run of the same output left, so only the rest is computed."""
    if resume and not save_artifacts:
        raise ValueError("resume needs save_artifacts: it continues from the saved artifacts")
    if coref_match not in MATCH_MODES:
        raise ValueError(f"coref_match must be one of {MATCH_MODES}, not {coref_match!r}")
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

    # Coreference is scored once per resolver over the whole corpus with the official CorefUD
    # scorer (micro-averaged, as in the CRAC shared tasks); the gold key is written once.
    output_dir = Path(output_dir)
    coref_dir = output_dir / "coref"
    coref_dir.mkdir(parents=True, exist_ok=True)
    key_text, dropped_gold = normalized_key(_read_key_text(corpus_path),
                                            {doc.doc_id: doc.clusters for doc in documents})
    key_path = coref_dir / "key.conllu"
    key_path.write_text(key_text, encoding="utf-8")

    def save(rel_path: str, content) -> None:
        if not save_artifacts:
            return
        path = output_dir / "artifacts" / rel_path.replace(":", "_")
        path.parent.mkdir(parents=True, exist_ok=True)
        text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, default=str)
        path.write_text(text, encoding="utf-8")

    def saved(rel_path: str):
        """With ``resume``, the artifact an earlier (interrupted) run of this output left, else None."""
        if not resume:
            return None
        path = output_dir / "artifacts" / rel_path.replace(":", "_")
        if not path.is_file():
            return None
        text = path.read_text(encoding="utf-8")
        return json.loads(text) if path.suffix == ".json" else text

    def build_graph(graph_backend, backend_name: str, who: str, doc_id: str, text: str) -> dict:
        rel = f"graphs/{backend_name}/{who}/{doc_id}.json"
        graph = saved(rel)
        if graph is None:
            graph = graph_backend.build(text)
            save(rel, graph)
        return graph

    for doc, oracle_text in zip(documents, oracle_texts):
        save(f"texts/ORACLE/{doc.doc_id}.txt", oracle_text)

    per_pairing = []
    for resolver_name, resolver in resolvers:
        # Exactly one resolve() call per (resolver, document). On resume, a text-only resolver's
        # saved text is reused (that is all it returns); resolvers with clusters run again, since
        # the clusters are not saved and those resolvers are cheap.
        resolver_outputs = []
        for doc in documents:
            text = saved(f"texts/{resolver_name}/{doc.doc_id}.txt") if not getattr(resolver, "returns_clusters", True) else None
            out = ResolverOutput(resolved_text=text, clusters=None) if text is not None else resolver.resolve(doc)
            save(f"texts/{resolver_name}/{doc.doc_id}.txt", out.resolved_text)
            resolver_outputs.append(out)
        coref_metrics = _score_resolver(resolver_name, documents, resolver_outputs, key_path,
                                        key_text, coref_dir, coref_match)
        if coref_metrics is not None:
            coref_metrics["dropped_gold_cross_sentence_mentions"] = dropped_gold

        for backend_name, graph_backend in backends:
            doc_entries = []
            for doc_index, doc in enumerate(documents):
                resolver_output = resolver_outputs[doc_index]

                cache_key = (backend_name, doc_index)
                if cache_key not in oracle_graph_cache:
                    oracle_graph_cache[cache_key] = build_graph(graph_backend, backend_name, "ORACLE",
                                                                doc.doc_id, oracle_texts[doc_index])
                oracle_graph = oracle_graph_cache[cache_key]

                # Genuinely depends on both resolver and backend: computed R x B.
                predicted_graph = build_graph(graph_backend, backend_name, resolver_name, doc.doc_id,
                                              resolver_output.resolved_text)

                graph_metrics = compute_graph_scores(oracle_graph, predicted_graph, graph_backend.backend_name)

                doc_entries.append({
                    "doc_id": doc.doc_id,
                    # coreference is scored at corpus level only (see the pairing entry)
                    "coreference_metrics": None,
                    "graph_metrics": graph_metrics,
                })

            per_pairing.append({
                "resolver": resolver_name,
                "graph_backend": backend_name,
                "documents": doc_entries,
                "coreference_metrics": coref_metrics,
                "graph_metrics": _average_graph(doc_entries),
            })

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
    parser.add_argument("--llm-model", default=None,
                        help="LLM for the LLMv2 resolver/backend: a Hugging Face model name loaded in-process "
                             "(default Qwen/Qwen2-1.5B-Instruct), or with --llm-endpoint the model name the "
                             "endpoint serves")
    parser.add_argument("--llm-endpoint", default=None,
                        help="OpenAI-compatible base URL (e.g. http://127.0.0.1:8081/v1): send LLMv2 prompts "
                             "to an already served model instead of loading one in-process")
    parser.add_argument("--llm-max-new-tokens", type=int, default=1024,
                        help="output token limit with --llm-endpoint (LLMv2's in-process default is 256)")
    parser.add_argument("--llm-temperature", type=float, default=0.0,
                        help="sampling temperature with --llm-endpoint (0 = greedy, reproducible)")
    parser.add_argument("--embedding-model", default=None,
                        help="sentence-transformers model for LLMv2 triple deduplication; default "
                             "all-mpnet-base-v2 (LLMv2's configured one) for English, "
                             "paraphrase-multilingual-mpnet-base-v2 for other languages")
    parser.add_argument("--save-artifacts", action="store_true",
                        help="keep every resolved text and graph under <output>/artifacts/")
    parser.add_argument("--resume", action="store_true",
                        help="with --save-artifacts: continue an interrupted run of the same --output, "
                             "reusing its saved graphs and LLM resolver texts")
    parser.add_argument("--coref-match", choices=MATCH_MODES, default="exact",
                        help="CorefUD scorer mention matching (default: exact; 'head' needs mention "
                             "heads in the gold data, e.g. CorefUD/GUM, not converted RuCoCo)")
    args = parser.parse_args(argv)

    resolver_names = args.resolvers.split(",")
    backend_names = args.graph_backends.split(",")

    llm_client = None
    embedder = None
    run_config = {"argv": argv if argv is not None else sys.argv[1:]}
    if "LLMv2" in resolver_names or "LLMv2" in backend_names:
        from .sys_path_setup import add_text_corpuses_processing_to_path

        add_text_corpuses_processing_to_path()
        from llm_v2.config_schema import EmbeddingConfig, LLMConfig
        from llm_v2.models.embedder import Embedder

        if args.llm_endpoint:
            from .llm.openai_client import OpenAICompatibleClient

            llm_client = OpenAICompatibleClient(args.llm_endpoint, args.llm_model or "default",
                                                max_new_tokens=args.llm_max_new_tokens,
                                                temperature=args.llm_temperature)
        else:
            from llm_v2.models.llm_client import LLMClient

            llm_client = LLMClient(LLMConfig(model_name=args.llm_model or "Qwen/Qwen2-1.5B-Instruct"))
        embedding_model = args.embedding_model or (
            "all-mpnet-base-v2" if args.language == "en" else "paraphrase-multilingual-mpnet-base-v2")
        embedder = Embedder(EmbeddingConfig(model_name=embedding_model))
        run_config["embedding_model"] = embedding_model

    results_path = run_experiment(
        corpus_path=args.corpus,
        language=args.language,
        resolver_names=resolver_names,
        graph_backend_names=backend_names,
        output_dir=args.output,
        llm_client=llm_client,
        embedder=embedder,
        coref_match=args.coref_match,
        save_artifacts=args.save_artifacts,
        resume=args.resume,
    )
    if hasattr(llm_client, "usage"):
        run_config["llm"] = llm_client.usage()
    (results_path.parent / "run_config.json").write_text(json.dumps(run_config, indent=1), encoding="utf-8")
    print(f"Results written to {results_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
