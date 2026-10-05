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


def test_resume_reuses_saved_graphs_and_texts_and_builds_only_what_is_missing(monkeypatch, tmp_path):
    from tests.pipeline.test_run_experiment_smoke import _CountingBackend, _CountingResolver, _patch_factories

    _install_fake_graph_metrics_module(monkeypatch)
    corpus = _write_two_doc_corpus(tmp_path)
    out_dir = tmp_path / "run"

    first_backend = _CountingBackend("B")
    text_only = _CountingResolver("T")
    text_only.returns_clusters = False
    _patch_factories(monkeypatch, {"T": text_only}, {"B": first_backend})
    first = run_experiment(corpus_path=corpus, language="en", resolver_names=["T"], graph_backend_names=["B"],
                           output_dir=out_dir, save_artifacts=True)
    assert len(first_backend.built_texts) == 4  # 2 oracle + 2 predicted graphs

    # an interruption lost the predicted graph of docB
    (out_dir / "artifacts" / "graphs" / "B" / "T" / "docB.json").unlink()
    second_backend = _CountingBackend("B")
    second_resolver = _CountingResolver("T")
    second_resolver.returns_clusters = False
    _patch_factories(monkeypatch, {"T": second_resolver}, {"B": second_backend})
    second = run_experiment(corpus_path=corpus, language="en", resolver_names=["T"], graph_backend_names=["B"],
                            output_dir=out_dir, save_artifacts=True, resume=True)
    assert second_resolver.calls == []                     # saved texts reused
    assert len(second_backend.built_texts) == 1            # only the missing graph is built
    strip = lambda p: [r["graph_metrics"] for r in json.loads(p.read_text(encoding="utf-8"))["results"]]
    assert strip(first) == strip(second)                   # same results as the uninterrupted run


def test_resume_without_artifacts_is_refused(tmp_path):
    with pytest.raises(ValueError, match="save_artifacts"):
        run_experiment(corpus_path=_write_two_doc_corpus(tmp_path), language="en", resolver_names=["NoResolution"],
                       graph_backend_names=["RuleBased"], output_dir=tmp_path / "r", resume=True)
