import json

from pipeline.report import (
    build_results,
    render_html_report,
    save_html_report,
    save_results_json,
)


def _sample_pairing():
    return [{
        "resolver": "LapinLiass",
        "graph_backend": "RuleBased",
        "coreference_metrics": {"conll_f1": 0.75},
        "graph_metrics": {
            "node_precision_recall_f1": {"f1": 0.8},
            "edge_precision_recall_f1": {"f1": 0.6},
            "smatch": {"f1": 0.7},
            "oracle_node_duplication_rate": 0.0,
            "predicted_node_duplication_rate": 0.25,
        },
    }, {
        "resolver": "LLMv2",
        "graph_backend": "LLMv2",
        "coreference_metrics": None,
        "graph_metrics": {
            "node_precision_recall_f1": {"f1": 0.9},
            "edge_precision_recall_f1": {"f1": 0.85},
            "smatch": {"f1": 0.88},
            "oracle_node_duplication_rate": 0.1,
            "predicted_node_duplication_rate": 0.4,
        },
    }]


def test_build_results_wraps_pairing_list_with_metadata():
    results = build_results("run1", "en", _sample_pairing())
    assert results["run_id"] == "run1"
    assert results["language"] == "en"
    assert len(results["results"]) == 2
    assert "generated_at" in results


def test_save_results_json_writes_valid_json(tmp_path):
    results = build_results("run1", "en", _sample_pairing())
    path = save_results_json(results, tmp_path)
    assert path.exists()
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["run_id"] == "run1"


def test_html_report_shows_na_for_missing_coreference_metrics():
    results = build_results("run1", "en", _sample_pairing())
    html = render_html_report(results)
    assert "LapinLiass" in html
    assert "LLMv2" in html
    assert "N/A" in html  # LLMv2 pairing has coreference_metrics=None
    assert "0.750" in html


def test_html_report_shows_node_duplication_rate_columns():
    """Finding 6b: the oracle-ablation "error budget" metric must be visible.

    It was computed per document and then dropped by the pairing-level
    aggregation, so a human reading the HTML never saw it.
    """
    results = build_results("run1", "en", _sample_pairing())
    html = render_html_report(results)
    assert "Oracle node dup. rate" in html
    assert "Predicted node dup. rate" in html
    assert "0.250" in html  # LapinLiass predicted duplication rate
    assert "0.400" in html  # LLMv2 predicted duplication rate


def test_html_report_renders_na_for_a_pairing_without_duplication_rates():
    pairing = [{
        "resolver": "Legacy",
        "graph_backend": "RuleBased",
        "coreference_metrics": {"conll_f1": 0.5},
        "graph_metrics": {"node_precision_recall_f1": {"f1": 0.5}},
    }]
    html = render_html_report(build_results("run1", "en", pairing))
    row = next(line for line in html.splitlines() if "Legacy" in line)
    assert row.count("<td>N/A</td>") == 4  # edge F1, smatch, both dup rates


def test_save_html_report_writes_file(tmp_path):
    results = build_results("run1", "en", _sample_pairing())
    html = render_html_report(results)
    path = save_html_report(html, tmp_path)
    assert path.exists()
    assert path.read_text(encoding="utf-8") == html
