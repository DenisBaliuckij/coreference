from __future__ import annotations

import smatch

from ..sys_path_setup import add_text_corpuses_processing_to_path
from ..graph.penman_convert import graph_to_amr_line


def _canonicalize(graph_dict: dict, backend: str):
    add_text_corpuses_processing_to_path()
    from graphMetrics import _to_networkx  # reused private helper, see design spec

    G = _to_networkx(graph_dict, backend)
    if backend == "RuleBased":
        label_of = {n: str(n) for n in G.nodes()}
    else:
        label_of = {n: G.nodes[n].get("label", str(n)) for n in G.nodes()}

    node_labels = set(label_of.values())
    edge_triples = set()
    for u, v, data in G.edges(data=True):
        a, b = sorted((label_of[u], label_of[v]))
        edge_triples.add((a, b, data.get("label", "")))
    return label_of, node_labels, edge_triples


def _prf(gold: set, pred: set) -> dict:
    tp = len(gold & pred)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_node_duplication_rate(graph_dict: dict, backend: str) -> float:
    """Fraction of nodes in this graph that share a label with another node
    in the same graph -- self-contained, not a comparison against another
    graph. Comparing oracle's rate to predicted's rate is what isolates the
    coreference-error contribution (oracle should be near zero by
    construction)."""
    add_text_corpuses_processing_to_path()
    from graphMetrics import _to_networkx

    G = _to_networkx(graph_dict, backend)
    if backend == "RuleBased":
        raw_labels = [str(n) for n in G.nodes()]
    else:
        raw_labels = [G.nodes[n].get("label", str(n)) for n in G.nodes()]

    if not raw_labels:
        return 0.0
    unique = len(set(raw_labels))
    return (len(raw_labels) - unique) / len(raw_labels)


def compute_smatch(oracle_graph: dict, predicted_graph: dict, backend: str) -> dict:
    _, oracle_nodes, oracle_edges = _canonicalize(oracle_graph, backend)
    _, pred_nodes, pred_edges = _canonicalize(predicted_graph, backend)

    oracle_nodes_dict = {f"o{i}": label for i, label in enumerate(sorted(oracle_nodes))}
    pred_nodes_dict = {f"p{i}": label for i, label in enumerate(sorted(pred_nodes))}
    oracle_label_to_id = {label: nid for nid, label in oracle_nodes_dict.items()}
    pred_label_to_id = {label: nid for nid, label in pred_nodes_dict.items()}

    oracle_edge_list = [
        (oracle_label_to_id[a], oracle_label_to_id[b], rel) for a, b, rel in oracle_edges
        if a in oracle_label_to_id and b in oracle_label_to_id
    ]
    pred_edge_list = [
        (pred_label_to_id[a], pred_label_to_id[b], rel) for a, b, rel in pred_edges
        if a in pred_label_to_id and b in pred_label_to_id
    ]

    oracle_line = graph_to_amr_line(oracle_nodes_dict, oracle_edge_list, "o")
    pred_line = graph_to_amr_line(pred_nodes_dict, pred_edge_list, "p")

    smatch.match_triple_dict.clear()  # module-level cache; must clear per independent comparison
    match_num, test_num, gold_num = smatch.get_amr_match(pred_line, oracle_line)
    precision, recall, f1 = smatch.compute_f(match_num, test_num, gold_num)
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_graph_scores(oracle_graph: dict, predicted_graph: dict, backend: str) -> dict:
    _, oracle_nodes, oracle_edges = _canonicalize(oracle_graph, backend)
    _, pred_nodes, pred_edges = _canonicalize(predicted_graph, backend)

    return {
        "node_precision_recall_f1": _prf(oracle_nodes, pred_nodes),
        "edge_precision_recall_f1": _prf(oracle_edges, pred_edges),
        "oracle_node_duplication_rate": compute_node_duplication_rate(oracle_graph, backend),
        "predicted_node_duplication_rate": compute_node_duplication_rate(predicted_graph, backend),
        "smatch": compute_smatch(oracle_graph, predicted_graph, backend),
    }
