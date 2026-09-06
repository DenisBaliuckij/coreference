from pipeline.graph.penman_convert import graph_to_amr_line

import amr  # from the `smatch` package; used only to validate our output parses


def test_empty_graph_produces_valid_line():
    line = graph_to_amr_line({}, [], "a")
    parsed = amr.AMR.parse_AMR_line(line)
    assert parsed is not None


def test_simple_graph_round_trips_through_amr_parser():
    nodes = {"n0": "Barack Obama", "n1": "Hawaii"}
    edges = [("n0", "n1", "born in")]
    line = graph_to_amr_line(nodes, edges, "a")
    parsed = amr.AMR.parse_AMR_line(line)
    parsed.rename_node("x")
    instance, _attr, relation = parsed.get_triples()
    concepts = {v for (_, _, v) in instance}
    assert "barack_obama" in concepts
    assert "hawaii" in concepts
    assert any(rel == "born-in" for (rel, _, _) in relation)


def test_cyclic_graph_does_not_infinite_loop_and_parses():
    nodes = {"x": "A", "y": "B"}
    edges = [("x", "y", "knows"), ("y", "x", "knows")]
    line = graph_to_amr_line(nodes, edges, "z")
    parsed = amr.AMR.parse_AMR_line(line)
    assert parsed is not None


def test_disconnected_components_all_declared_once():
    nodes = {"p": "A", "q": "B", "r": "C"}
    edges = [("p", "q", "rel1")]  # r is isolated
    line = graph_to_amr_line(nodes, edges, "d")
    parsed = amr.AMR.parse_AMR_line(line)
    parsed.rename_node("x")
    instance, _attr, _relation = parsed.get_triples()
    # 3 real nodes + 1 synthetic root = 4 instance triples, each node declared exactly once
    assert len(instance) == 4


def test_relation_ending_in_of_is_escaped():
    nodes = {"n0": "engine", "n1": "steel"}
    edges = [("n0", "n1", "made of")]
    line = graph_to_amr_line(nodes, edges, "a")
    parsed = amr.AMR.parse_AMR_line(line)
    parsed.rename_node("x")
    _instance, _attr, relation = parsed.get_triples()
    assert any(rel == "made-of_" for (rel, _, _) in relation)


def test_concept_label_containing_slash_is_stripped_not_left_to_break_parser():
    # A literal "/" in a node label (e.g. "race/ethnicity", common in real
    # corpora) collides with PENMAN's own var/concept separator and made
    # smatch.get_amr_match() return None on real documents before this was
    # sanitized alongside "(", ")", and '"'.
    nodes = {"n0": "race/ethnicity", "n1": "discrimination"}
    edges = [("n0", "n1", "linked to")]
    line = graph_to_amr_line(nodes, edges, "a")
    parsed = amr.AMR.parse_AMR_line(line)
    assert parsed is not None
    parsed.rename_node("x")
    instance, _attr, _relation = parsed.get_triples()
    concepts = {v for (_, _, v) in instance}
    assert "race_ethnicity" in concepts
    assert "raceethnicity" not in concepts


def test_relation_label_containing_colon_is_stripped_not_left_to_break_parser():
    nodes = {"n0": "a", "n1": "b"}
    edges = [("n0", "n1", "ratio: 2:1")]
    line = graph_to_amr_line(nodes, edges, "a")
    parsed = amr.AMR.parse_AMR_line(line)
    assert parsed is not None
