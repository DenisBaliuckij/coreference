from __future__ import annotations

import random

import smatch

# smatch caches the score of every node mapping its hill-climbing search evaluates, keyed by the
# whole mapping, in a module-level dict that is never trimmed: on graphs of a few hundred nodes it
# grows to many GB (a 450-node pair was OOM-killed at 12 GB). The cache only saves recomputation,
# so it is cleared whenever it passes _SMATCH_CACHE_LIMIT entries; scores are unchanged.
_SMATCH_CACHE_LIMIT = 50_000
if not getattr(smatch.compute_match, "_bounded", False):
    _smatch_compute_match = smatch.compute_match

    def _bounded_compute_match(mapping, weight_dict):
        if len(smatch.match_triple_dict) > _SMATCH_CACHE_LIMIT:
            smatch.match_triple_dict.clear()
        return _smatch_compute_match(mapping, weight_dict)

    _bounded_compute_match._bounded = True
    smatch.compute_match = _bounded_compute_match


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
    metric was meant for id/label backends (LLMv2), where two distinct node ids
    could share a label -- but the LLMv2 graph builder also merges nodes by
    normalised label (checked on the 2026-10-05 GUM run: 0 duplicates in every
    graph), so on both backends an unresolved anaphor appears as a pronoun-labelled
    node instead (see analysis.summarize.graph_sizes). This function still counts
    duplicates faithfully for either shape if the dict does contain them.
    """
    raw_labels = _raw_node_labels(graph_dict, backend)
    if not raw_labels:
        return 0.0
    unique = len(set(raw_labels))
    return (len(raw_labels) - unique) / len(raw_labels)


def compute_smatch(oracle_graph: dict, predicted_graph: dict, backend: str) -> dict:
    """Smatch with its own alignment search. NOT part of the default scores (see
    compute_triple_f1): memory grows about with the fourth power of the node count, so use it
    only on sentence-sized graphs (a few dozen nodes)."""
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

    # Smatch's alignment search runs on the graphs' own triples: one instance triple per node
    # (its label) and one relation triple per edge. The graphs used to be serialised to a single
    # AMR line hung under a synthetic root with numbered :has-entityN edges; those scaffolding
    # triples (2 per node) dominated the score and paired unrelated nodes that shared a sorted
    # position, and the search over them took >14 GB on a 450-node LLMv2 graph (OOM-killed run,
    # 2026-10-05). Node names must be <prefix><index in the instance list> for smatch's pool.
    def triples(nodes_dict, edge_list, prefix):
        index = {nid: f"{prefix}{i}" for i, nid in enumerate(nodes_dict)}
        instances = [("instance", index[nid], label.strip().lower()) for nid, label in nodes_dict.items()]
        relations = [(rel.strip().lower(), index[a], index[b]) for a, b, rel in edge_list]
        return instances, relations

    pred_inst, pred_rel = triples(pred_nodes_dict, pred_edge_list, "a")
    oracle_inst, oracle_rel = triples(oracle_nodes_dict, oracle_edge_list, "b")
    test_num, gold_num = len(pred_inst) + len(pred_rel), len(oracle_inst) + len(oracle_rel)
    if test_num == 0 or gold_num == 0:
        # two empty graphs agree (1.0, as node/edge P/R say); one empty side matches nothing
        same = test_num == gold_num
        return {"precision": float(same), "recall": float(same), "f1": float(same)}

    smatch.match_triple_dict.clear()  # module-level cache; must clear per independent comparison
    state = random.getstate()
    random.seed(0)  # smatch's hill-climbing restarts are random: fixed seed, reproducible scores
    try:
        _, match_num = smatch.get_best_match(pred_inst, [], pred_rel, oracle_inst, [], oracle_rel, "a", "b")
    finally:
        random.setstate(state)
    precision, recall, f1 = smatch.compute_f(match_num, test_num, gold_num)
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_triple_f1(oracle_graph: dict, predicted_graph: dict, backend: str) -> dict:
    """Smatch-style triple F1 under the label alignment: every node contributes one instance
    triple (its label) and every edge one relation triple, and a node of one graph is aligned
    with the node of the other graph that has the same canonical label.

    This is what Smatch computes when its node alignment is fixed to label identity. Smatch's own
    hill-climbing search over alignments is not used by default: on document-level graphs it does
    not scale (measured 2026-10-05: 3 GB and 58 s at 200 nodes, OOM at 450), and in graphs whose
    node identity is their canonical label the only alignments it can add pair nodes with
    different labels, i.e. different entities. Equal to (|nodes match| + |edges match|) over the
    triple counts of each side; deterministic and linear in graph size.
    """
    _, oracle_nodes, oracle_edges = _canonicalize(oracle_graph, backend)
    _, pred_nodes, pred_edges = _canonicalize(predicted_graph, backend)
    gold = {("node", n) for n in oracle_nodes} | {("edge",) + e for e in oracle_edges}
    pred = {("node", n) for n in pred_nodes} | {("edge",) + e for e in pred_edges}
    return _prf(gold, pred)


def compute_graph_scores(oracle_graph: dict, predicted_graph: dict, backend: str) -> dict:
    _, oracle_nodes, oracle_edges = _canonicalize(oracle_graph, backend)
    _, pred_nodes, pred_edges = _canonicalize(predicted_graph, backend)

    return {
        "node_precision_recall_f1": _prf(oracle_nodes, pred_nodes),
        "edge_precision_recall_f1": _prf(oracle_edges, pred_edges),
        "oracle_node_duplication_rate": compute_node_duplication_rate(oracle_graph, backend),
        "predicted_node_duplication_rate": compute_node_duplication_rate(predicted_graph, backend),
        "triple_f1": compute_triple_f1(oracle_graph, predicted_graph, backend),
    }
