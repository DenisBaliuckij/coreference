from __future__ import annotations

from collections import defaultdict


def _strip_penman_specials(s: str, space_replacement: str) -> str:
    # "/" separates a variable from its concept and ":" introduces a relation
    # in PENMAN notation; a label containing either breaks smatch's parser
    # exactly like an unescaped "(" or '"' would. Both typically separate
    # meaningful word tokens (e.g. "race/ethnicity"), so - like whitespace -
    # they become the separator rather than being dropped outright, which
    # would otherwise glue adjacent words into one unreadable token.
    for ch in " /:":
        s = s.replace(ch, space_replacement)
    for ch in "()\"":
        s = s.replace(ch, "")
    return s


def _sanitize_concept(label: str) -> str:
    s = _strip_penman_specials(label.strip().lower(), "_")
    return s or "concept"


def _sanitize_rel(rel: str) -> str:
    s = _strip_penman_specials(rel.strip().lower(), "-")
    if not s:
        s = "rel"
    if s.endswith("-of"):
        # smatch's AMR parser treats a trailing "-of" as "this relation is the
        # inverse of <rel without -of>" and silently flips triple direction.
        # Our KG relation labels aren't AMR roles, so escape to avoid that.
        s = s + "_"
    return s


def graph_to_amr_line(nodes: dict[str, str], edges: list[tuple[str, str, str]], var_prefix: str) -> str:
    """Serialize a (nodes, edges) graph into a single-line AMR/Penman string
    that smatch.get_amr_match() can parse.

    nodes: {node_id: label}
    edges: [(source_id, target_id, relation_label), ...]

    Every node is attached under one synthetic root via :has-entityN edges so
    disconnected components serialize correctly; each node is declared once
    and later references (including cycles) use bare-variable reentrancy.
    """
    node_ids = list(nodes.keys())
    if not node_ids:
        return f"({var_prefix}top / empty-graph)"

    var_of = {nid: f"{var_prefix}{i}" for i, nid in enumerate(node_ids)}
    outgoing: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for src, tgt, rel in edges:
        if src in var_of and tgt in var_of:
            outgoing[src].append((_sanitize_rel(rel), tgt))

    declared: set[str] = set()

    def render(node_id: str) -> str:
        var = var_of[node_id]
        if var in declared:
            return var
        declared.add(var)
        concept = _sanitize_concept(nodes[node_id])
        parts = [f"({var} / {concept}"]
        for rel, tgt in outgoing.get(node_id, []):
            parts.append(f":{rel} {render(tgt)}")
        parts.append(")")
        return " ".join(parts)

    children = " ".join(f":has-entity{i} {render(nid)}" for i, nid in enumerate(node_ids))
    return f"({var_prefix}top / graph-root {children})"
