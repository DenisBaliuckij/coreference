from pipeline.analysis.replacement_agreement import agreement, classify


def test_classify_separates_same_name_other_name_and_other_referent():
    assert classify("Antonín Dvořák", "Antonín Dvořák's") == "same_as_oracle"
    assert classify("Antonín Dvořák", "Dvořák") == "different_string_sharing_a_word"
    assert classify("Antonín Dvořák", "Smetana") == "different_string_no_shared_word"
    assert classify("Antonín Dvořák", None) == "no_edit_at_span"


def test_agreement_counts_single_word_oracle_replacements(tmp_path):
    texts = {"NoResolution": "Dvořák came . He left .", "ORACLE": "Dvořák came . Dvořák left .",
             "A": "Dvořák came . Dvořák left .", "B": "Dvořák came . He left ."}
    for who, t in texts.items():
        (tmp_path / who).mkdir()
        (tmp_path / who / "d.txt").write_text(t, encoding="utf-8")
    r = agreement(tmp_path, ["A", "B"])
    assert r["oracle_single_word_replacements"] == 1
    assert r["resolvers"]["A"]["same_as_oracle"] == 1 and r["resolvers"]["B"]["no_edit_at_span"] == 1
