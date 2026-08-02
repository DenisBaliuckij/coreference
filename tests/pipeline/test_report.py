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
        },
    }, {
        "resolver": "LLMv2",
        "graph_backend": "LLMv2",
        "coreference_metrics": None,
        "graph_metrics": {
            "node_precision_recall_f1": {"f1": 0.9},
            "edge_precision_recall_f1": {"f1": 0.85},
            "smatch": {"f1": 0.88},
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


def test_save_html_report_writes_file(tmp_path):
    results = build_results("run1", "en", _sample_pairing())
    html = render_html_report(results)
    path = save_html_report(html, tmp_path)
    assert path.exists()
    assert path.read_text(encoding="utf-8") == html
