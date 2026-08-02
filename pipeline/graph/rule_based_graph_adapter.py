from __future__ import annotations

from ..sys_path_setup import add_text_corpuses_processing_to_path


class RuleBasedGraphAdapter:
    name = "RuleBased"
    backend_name = "RuleBased"
    language_support = {"en"}

    def build(self, text: str) -> dict:
        add_text_corpuses_processing_to_path()
        from graphBuilder import extract_graph_edges, merge_graph

        edges = extract_graph_edges(text)
        graph = {"nodes": [], "edges": []}
        return merge_graph(graph, edges)
