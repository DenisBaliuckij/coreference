from pipeline.analysis.dropped_passages import dropped_words


def test_long_omissions_count_and_name_substitutions_do_not():
    original = ("He came . " + "one two three four five six seven eight nine ten eleven . " + "He left .").split()
    renamed = [("John" if w == "He" else w) for w in original]
    assert dropped_words(original, renamed) == 0
    shortened = ("John came . John left .").split()
    assert dropped_words(original, shortened) == 12
