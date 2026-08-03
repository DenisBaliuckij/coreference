from __future__ import annotations

import smatch

from ..graph.penman_convert import graph_to_amr_line


def _edge_endpoints(edge: dict, backend: str) -> tuple[str, str, str]:
    """(source, target, relation) in the backend-native dict's own direction."""
    if backend == "RuleBased":
        return str(edge["agent_1"]), str(edge["agent_2"]), str(edge.get("meaning", ""))
    return str(edge["source"]), str(edge["target"]), str(edge.get("label", ""))


def _raw_node_labels(graph_dict: dict, backend: str) -> list[str]:
    """Node labels straight out of the backend-native dict, duplicates intact.

    Deliberately does NOT route through graphMetrics._to_networkx: networkx
    silently collapses same-key nodes on add_node, which erased every duplicate
    before it could be counted.
    """
    if backend == "RuleBased":
        return [str(node) for node in graph_dict.get("nodes", [])]
    return [str(node["label"]) for node in graph_dict.get("nodes", [])]


def _canonicalize(graph_dict: dict, backend: str):
    """Reduce a backend-native graph dict to (label_of, node_labels, edge_triples).

    Reads the dict directly instead of going through graphMetrics._to_networkx:
    that helper builds an *undirected* nx.Graph, which -- together with the
    alphabetical endpoint sort this function used to apply -- made
    "john --hit--> mary" and "mary --hit--> john" indistinguishable. Edge
    triples now preserve the backend's own source -> target direction.
    """
    if backend == "RuleBased":
        label_of = {str(node): str(node) for node in graph_dict.get("nodes", [])}
    else:
        label_of = {str(node["id"]): str(node["label"]) for node in graph_dict.get("nodes", [])}

    edge_triples = set()
    for edge in graph_dict.get("edges", []):
        src, tgt, rel = _edge_endpoints(edge, backend)
        # Endpoints not declared in "nodes" still count, mirroring what
        # nx.add_edge used to do implicitly; their id doubles as their label.
        label_of.setdefault(src, src)
        label_of.setdefault(tgt, tgt)
        edge_triples.add((label_of[src], label_of[tgt], rel))

    node_labels = set(label_of.values())
    return label_of, node_labels, edge_triples


def _prf(gold: set, pred: set) -> dict:
    if not gold and not pred:
        # Both sides vacuous (e.g. a short document where the extractor found
        # nothing): that is agreement, not total failure. smatch already scores
        # empty-vs-empty as 1.0; all three metrics must say the same thing
        # before they get averaged into one pairing summary.
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
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
    construction).

    Note on the RuleBased backend: graphBuilder.merge_graph stores nodes as
    ``list(set(...))``, so a RuleBased graph dict produced by the real backend
    can never carry duplicate labels and its rate is structurally 0.0. The
    metric is meaningful for id/label backends (LLMv2), where two distinct node
    ids may share a label. This function still counts duplicates faithfully for
    either shape if the dict does contain them.
    """
    raw_labels = _raw_node_labels(graph_dict, backend)
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
