import pytest

from pipeline.analysis.summarize import _renamed, _split_docs, bootstrap_ci, paired_bootstrap, text_diagnostics


def test_bootstrap_ci_brackets_the_mean_and_collapses_for_constant_values():
    lo, hi = bootstrap_ci([0.2, 0.4, 0.6, 0.8], n=500)
    assert lo <= 0.5 <= hi
    assert bootstrap_ci([0.3] * 10, n=200) == (0.3, 0.3)


def test_paired_bootstrap_detects_a_consistent_improvement():
    a = [0.5, 0.6, 0.7, 0.8, 0.9]
    b = [0.4, 0.5, 0.6, 0.7, 0.8]
    r = paired_bootstrap(a, b, n=500)
    assert r["mean_diff"] == pytest.approx(0.1)
    assert r["ci95"][0] > 0 and r["p_one_sided"] == 0.0


def test_documents_are_split_and_renamed_for_resampling():
    text = "# global.Entity = eid-etype-head-other\n# newdoc id = d1\n# sent_id = d1-1\n1\ta\n\n# newdoc id = d2\n# sent_id = d2-1\n1\tb\n"
    header, docs = _split_docs(text)
    assert header == ["# global.Entity = eid-etype-head-other"]
    assert list(docs) == ["d1", "d2"]
    renamed = _renamed(docs["d1"], "d1", "d1__b0")
    assert "# newdoc id = d1__b0" in renamed and "# sent_id = d1__b0-1" in renamed


def test_text_diagnostics_measures_pronouns_and_similarity(tmp_path):
    for sub, text in (("ORACLE", "John came. John left."), ("NoResolution", "John came. He left."),
                      ("LLMv2", "John came. John left.")):
        (tmp_path / "texts" / sub).mkdir(parents=True)
        (tmp_path / "texts" / sub / "d.txt").write_text(text, encoding="utf-8")
    t = text_diagnostics(tmp_path, "en")
    assert t["NoResolution"]["pronouns_per_1000_words"] == pytest.approx(250.0)
    assert t["LLMv2"]["pronouns_per_1000_words"] == 0.0
    assert t["LLMv2"]["char_similarity_to_oracle"] == pytest.approx(1.0)


def test_resampled_documents_are_separated_by_a_blank_line():
    from pipeline.analysis.summarize import _join_blocks

    text = _join_blocks(["# global.Entity = x"], ["# newdoc id = a\n1\tx", "# newdoc id = b\n1\ty\n"])
    assert text == "# global.Entity = x\n# newdoc id = a\n1\tx\n\n# newdoc id = b\n1\ty\n\n"
