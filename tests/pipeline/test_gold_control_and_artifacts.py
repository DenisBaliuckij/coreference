import json

import pytest

from pipeline.eval.corefud_scoring import scorer_available
from pipeline.resolvers.gold_adapter import GoldAdapter
from pipeline.run_experiment import run_experiment
from tests.pipeline.test_run_experiment_smoke import (
    _install_fake_graph_builder_module,
    _install_fake_graph_metrics_module,
    _write_two_doc_corpus,
)


def test_gold_resolver_returns_the_oracle_text_and_the_gold_clusters():
    from pipeline.corpus.corefud_loader import parse_conllu
    from tests.pipeline.test_run_experiment_smoke import TWO_DOC_CORPUS

    doc = parse_conllu(TWO_DOC_CORPUS)[0]
    out = GoldAdapter().resolve(doc)
    assert out.resolved_text == "John arrived. John left."
    assert out.clusters == doc.clusters


@pytest.mark.skipif(not scorer_available(), reason="official CorefUD scorer not installed")
def test_gold_control_scores_one_and_artifacts_are_kept(monkeypatch, tmp_path):
    _install_fake_graph_builder_module(monkeypatch)
    _install_fake_graph_metrics_module(monkeypatch)
    out_dir = tmp_path / "run"
    results = run_experiment(
        corpus_path=_write_two_doc_corpus(tmp_path),
        language="en",
        resolver_names=["Gold", "NoResolution"],
        graph_backend_names=["RuleBased"],
        output_dir=out_dir,
        save_artifacts=True,
    )
    by_resolver = {r["resolver"]: r for r in json.loads(results.read_text(encoding="utf-8"))["results"]}
    assert by_resolver["Gold"]["coreference_metrics"]["conll_f1"] == pytest.approx(1.0)
    assert by_resolver["NoResolution"]["coreference_metrics"] is None
    # a deterministic backend reproduces the oracle graph exactly from the gold text
    assert by_resolver["Gold"]["graph_metrics"]["edge_precision_recall_f1"]["f1"] == pytest.approx(1.0)

    art = out_dir / "artifacts"
    assert (art / "texts" / "ORACLE" / "docA.txt").read_text(encoding="utf-8") == "John arrived. John left."
    assert (art / "texts" / "NoResolution" / "docA.txt").read_text(encoding="utf-8") == "John arrived. He left."
    assert (art / "graphs" / "RuleBased" / "ORACLE" / "docB.json").exists()
    assert json.loads((art / "graphs" / "RuleBased" / "Gold" / "docA.json").read_text(encoding="utf-8"))["edges"]


def test_without_save_artifacts_nothing_extra_is_written(monkeypatch, tmp_path):
    _install_fake_graph_builder_module(monkeypatch)
    _install_fake_graph_metrics_module(monkeypatch)
    out_dir = tmp_path / "run"
    run_experiment(corpus_path=_write_two_doc_corpus(tmp_path), language="en",
                   resolver_names=["NoResolution"], graph_backend_names=["RuleBased"], output_dir=out_dir)
    assert not (out_dir / "artifacts").exists()
