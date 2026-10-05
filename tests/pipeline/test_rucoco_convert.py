import pytest

spacy = pytest.importorskip("spacy")

from pipeline.corpus.corefud_loader import parse_conllu
from pipeline.corpus.rucoco_convert import _nlp, convert_document

TEXT = "Компромисс возможен\n\nГлава комитета Александр Жуков выступает за компромисс. Он заявил об этом АФИ."
ZHUKOV = (TEXT.index("Александр"), TEXT.index("Жуков") + len("Жуков"))
HE = (TEXT.index("Он"), TEXT.index("Он") + 2)
KOMPROMISS_1 = (0, len("Компромисс"))
KOMPROMISS_2 = (TEXT.index("компромисс."), TEXT.index("компромисс.") + len("компромисс"))


@pytest.fixture(scope="module")
def nlp():
    return _nlp()


def _convert(nlp, entities, includes=None):
    return convert_document("doc1", {"text": TEXT, "entities": entities, "includes": includes or []}, nlp)


def test_mentions_map_to_the_words_they_cover(nlp):
    conllu, stats = _convert(nlp, [[list(ZHUKOV), list(HE)], [list(KOMPROMISS_1), list(KOMPROMISS_2)]])
    doc = parse_conllu(conllu)[0]
    words = [t.text for t in doc.tokens]
    as_text = sorted(sorted(" ".join(words[m.start_token: m.end_token + 1]) for m in c) for c in doc.clusters)
    assert as_text == [["Александр Жуков", "Он"], ["Компромисс", "компромисс"]]
    assert stats["entities"] == 2 and stats["mentions"] == 4 and stats["unmappable_mentions"] == 0


def test_text_is_reconstructed_and_paragraph_break_ends_a_sentence(nlp):
    conllu, _ = _convert(nlp, [])
    doc = parse_conllu(conllu)[0]
    assert doc.text == TEXT.replace("\n\n", " ")
    first_sentence = [t.text for t in doc.tokens if t.sent_index == 0]
    assert first_sentence == ["Компромисс", "возможен"]


def test_no_sentence_break_inside_a_mention(nlp):
    span = (TEXT.index("Жуков"), TEXT.index("Он") + 2)  # crosses the sentence end "компромисс."
    conllu, _ = _convert(nlp, [[list(span), list(HE)]])
    doc = parse_conllu(conllu)[0]
    m = doc.clusters[0][0]
    assert doc.tokens[m.start_token].sent_index == doc.tokens[m.end_token].sent_index


def test_split_antecedents_are_counted_not_written(nlp):
    conllu, stats = _convert(nlp, [[list(ZHUKOV)], [list(HE)]], includes=[[], [0]])
    assert stats["split_antecedents"] == 1
    assert "SplitAnte" not in conllu


def test_every_sentence_has_an_id_for_the_scorer(nlp):
    conllu, _ = _convert(nlp, [])
    sent_ids = [l for l in conllu.splitlines() if l.startswith("# sent_id")]
    assert len(sent_ids) == len({l for l in sent_ids}) >= 3
