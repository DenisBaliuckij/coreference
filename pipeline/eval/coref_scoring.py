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
    key_clusters = _clusters_as_tuples(gold_clusters)
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
