from __future__ import annotations

from coval.eval.evaluator import b_cubed, ceafe, evaluate_documents, muc

from ..types import MentionSpan


def _mention_key(span: MentionSpan) -> tuple[int, int]:
    return (span.start_token, span.end_token)


def _clusters_as_tuples(clusters: list[list[MentionSpan]]) -> list[tuple]:
    return [tuple(sorted({_mention_key(m) for m in cluster})) for cluster in clusters]


def _mention_to_other_cluster(own_clusters: list[tuple], other_clusters: list[tuple]) -> dict:
    other_by_mention: dict = {}
    for oc in other_clusters:
        for m in oc:
            other_by_mention[m] = oc
    mapping: dict = {}
    for cluster in own_clusters:
        for m in cluster:
            if m in other_by_mention:
                mapping[m] = other_by_mention[m]
    return mapping


def score_coreference(
    gold_clusters: list[list[MentionSpan]], sys_clusters: list[list[MentionSpan]]
) -> dict:
    """Score system clusters against gold clusters with MUC/B3/CEAFe/CoNLL F1.

    Gold (KEY) clusters with fewer than two mentions are dropped before
    scoring, per the standard CoNLL-2012 evaluation convention: a "cluster" of
    size 1 is an annotated mention with no coreference partner, and MUC/B3/CEAFe
    are undefined or misleading over singletons. The CorefUD loader keeps every
    annotated mention, so gold data routinely contains singleton clusters,
    while the resolver adapters structurally cannot emit them -- leaving them in
    the key deflates every metric (a resolver that perfectly recovers the one
    real coreference link in a document with three extra singleton mentions
    scored 0.657 instead of 1.0).

    System (RESPONSE) clusters are deliberately NOT filtered: a system that
    reports a singleton is a legitimate signal the existing machinery handles.
    """
    key_clusters = [c for c in _clusters_as_tuples(gold_clusters) if len(c) >= 2]
    sys_clusters_t = _clusters_as_tuples(sys_clusters)

    key_mention_sys_cluster = _mention_to_other_cluster(key_clusters, sys_clusters_t)
    sys_mention_key_cluster = _mention_to_other_cluster(sys_clusters_t, key_clusters)

    coref_info = (key_clusters, sys_clusters_t, key_mention_sys_cluster, sys_mention_key_cluster)
    doc_coref_infos = {"doc1": coref_info}

    results: dict = {}
    for name, metric in (("muc", muc), ("b_cubed", b_cubed), ("ceafe", ceafe)):
        recall, precision, f1_score = evaluate_documents(doc_coref_infos, metric)
        results[name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1_score),
        }

    results["conll_f1"] = (results["muc"]["f1"] + results["b_cubed"]["f1"] + results["ceafe"]["f1"]) / 3
    return results
