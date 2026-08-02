# Cascade Experimental Stand Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone CLI pipeline (`coreference/pipeline/`) that loads a CorefUD gold corpus, runs a cascade of coreference-resolution + graph-construction (reusing `text-corpuses-processing`'s resolvers and `llm_v2` pipeline), and scores both coreference quality (CoNLL F1) and graph quality (oracle-ablation: oracle-resolved graph vs. predicted-resolved graph) via node/edge P/R, node-duplication rate, and a Smatch-style structural score.

**Architecture:** A `CorefDocument` (parsed from CoNLL-U `Entity=` bracket notation) flows through pluggable `ResolverAdapter`s (wrapping the sibling repo's `LapinLiass`/`SpacyNeural`/`llm_v2` coreference code) and pluggable `GraphBackendAdapter`s (wrapping `graphBuilder.py` and `llm_v2`'s extraction pipeline). Both are compared against an oracle path built directly from gold clusters, with no model involved. `text-corpuses-processing/dags` is added to `sys.path` at runtime; nothing there is copied or modified.

**Tech Stack:** Python 3.10+, pytest, `coval` (CoNLL coreference scorer, installed from `git+https://github.com/ns-moosavi/coval.git` — the PyPI package named `coval` is an unrelated project, do not use it), `smatch` (PyPI, AMR/graph matching), `networkx` (transitively required by the reused `graphMetrics.py`).

## Global Constraints

- Python 3.10+ (matches `text-corpuses-processing`'s stated floor).
- Do not modify any file under `text-corpuses-processing/` — reuse via `sys.path`, never copy/fork logic.
- All cross-repo imports of `text-corpuses-processing` modules happen **lazily inside functions/methods**, never at module top level, so tests can inject fakes via `sys.modules` without requiring the real spaCy/transformers models to be installed.
- `smatch.match_triple_dict.clear()` must be called immediately before every `smatch.get_amr_match(...)` call — it is a module-level memoization cache keyed by mapping-index tuples, not graph content, and leaks results across independent comparisons if not cleared (verified by hand during design).
- All metric dicts returned by scoring functions must contain plain Python `float`/`int`, not `numpy.float64` (cast explicitly) — `coval`'s `ceafe` returns numpy scalars, and `json.dumps` rejects them.
- Every `MentionSpan`/token coordinate used for coreference scoring is expressed in **gold CoNLL-U token indices**, never in a resolver's own tokenizer's indices — adapters project into gold coordinates before returning.

---

## File Structure

```
coreference/
  pipeline/
    __init__.py
    types.py                          # Token, MentionSpan, CorefDocument, ResolverOutput
    sys_path_setup.py                 # adds text-corpuses-processing/dags to sys.path
    config.py                         # (created empty in Task 1, used by Task 15)
    corpus/
      __init__.py
      corefud_loader.py                # parse_conllu, load_corefud_corpus
      oracle.py                        # build_oracle_text
    resolvers/
      __init__.py
      base.py                          # ResolverAdapter Protocol, UnionFind, span projection
      lapin_liass_adapter.py
      spacy_neural_adapter.py
      llm_v2_adapter.py
    graph/
      __init__.py
      base.py                          # GraphBackendAdapter Protocol
      rule_based_graph_adapter.py
      llm_v2_graph_adapter.py
      penman_convert.py                # graph_to_amr_line
    eval/
      __init__.py
      coref_scoring.py                 # score_coreference (coval-based)
      graph_scoring.py                 # compute_graph_scores, compute_smatch
    report.py
    run_experiment.py                  # CLI entry point
  data/
    corpora/
      sample_en_mini.conllu            # tiny hand-written fixture for the smoke test
  reports/                             # run outputs (gitignored)
  tests/
    pipeline/
      test_sys_path_setup.py
      test_types.py
      test_corefud_loader.py
      test_oracle.py
      test_resolvers_base.py
      test_lapin_liass_adapter.py
      test_spacy_neural_adapter.py
      test_llm_v2_adapter.py
      test_rule_based_graph_adapter.py
      test_penman_convert.py
      test_graph_scoring.py
      test_llm_v2_graph_adapter.py
      test_coref_scoring.py
      test_report.py
      test_run_experiment_smoke.py
  conftest.py
  requirements.txt
  .gitignore
  README.md
```

---

### Task 1: Project scaffolding

**Files:**
- Create: `coreference/pipeline/__init__.py`
- Create: `coreference/pipeline/corpus/__init__.py`
- Create: `coreference/pipeline/resolvers/__init__.py`
- Create: `coreference/pipeline/graph/__init__.py`
- Create: `coreference/pipeline/eval/__init__.py`
- Create: `coreference/pipeline/sys_path_setup.py`
- Create: `coreference/conftest.py`
- Create: `coreference/tests/__init__.py`
- Create: `coreference/tests/pipeline/__init__.py`
- Create: `coreference/requirements.txt`
- Create: `coreference/.gitignore`
- Create: `coreference/README.md`
- Test: `coreference/tests/pipeline/test_sys_path_setup.py`

**Interfaces:**
- Produces: `add_text_corpuses_processing_to_path() -> pathlib.Path` — every later task's adapters call this before importing anything from `text-corpuses-processing`.

- [ ] **Step 1: Create empty package `__init__.py` files**

```bash
mkdir -p pipeline/corpus pipeline/resolvers pipeline/graph pipeline/eval tests/pipeline data/corpora reports
touch pipeline/__init__.py pipeline/corpus/__init__.py pipeline/resolvers/__init__.py \
      pipeline/graph/__init__.py pipeline/eval/__init__.py tests/__init__.py tests/pipeline/__init__.py
```

- [ ] **Step 2: Write `conftest.py` at repo root**

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
```

- [ ] **Step 3: Write the failing test for `sys_path_setup.py`**

```python
# tests/pipeline/test_sys_path_setup.py
import sys
from pathlib import Path

from pipeline.sys_path_setup import add_text_corpuses_processing_to_path


def test_default_path_points_at_sibling_dags_dir():
    result = add_text_corpuses_processing_to_path()
    assert result == Path(__file__).resolve().parents[2] / "text-corpuses-processing" / "dags"
    assert str(result) in sys.path


def test_env_var_override(monkeypatch, tmp_path):
    monkeypatch.setenv("TEXT_CORPUSES_PROCESSING_DAGS", str(tmp_path))
    result = add_text_corpuses_processing_to_path()
    assert result == tmp_path
    assert str(tmp_path) in sys.path


def test_idempotent_does_not_duplicate_sys_path_entry():
    sys.path[:] = [p for p in sys.path if "text-corpuses-processing" not in p]
    add_text_corpuses_processing_to_path()
    count_after_first = sum(1 for p in sys.path if "text-corpuses-processing" in p)
    add_text_corpuses_processing_to_path()
    count_after_second = sum(1 for p in sys.path if "text-corpuses-processing" in p)
    assert count_after_first == 1
    assert count_after_second == 1
```

- [ ] **Step 4: Run test to verify it fails**

Run: `pytest tests/pipeline/test_sys_path_setup.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.sys_path_setup'`

- [ ] **Step 5: Write `pipeline/sys_path_setup.py`**

```python
from __future__ import annotations

import os
import sys
from pathlib import Path


def add_text_corpuses_processing_to_path() -> Path:
    """Add the sibling text-corpuses-processing repo's dags/ dir to sys.path.

    No existence check: if the directory is missing, later imports of the
    reused modules fail with the standard ModuleNotFoundError, which is more
    informative than a raise here and keeps this function safe to call
    unconditionally in tests that mock the reused modules via sys.modules.
    """
    override = os.environ.get("TEXT_CORPUSES_PROCESSING_DAGS")
    if override:
        dags_dir = Path(override)
    else:
        dags_dir = Path(__file__).resolve().parents[2] / "text-corpuses-processing" / "dags"

    dags_str = str(dags_dir)
    if dags_str not in sys.path:
        sys.path.insert(0, dags_str)
    return dags_dir
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/pipeline/test_sys_path_setup.py -v`
Expected: PASS (3 tests)

- [ ] **Step 7: Write `requirements.txt`, `.gitignore`, `README.md`**

`requirements.txt`:
```
networkx>=3.0
numpy>=1.21.0
scipy>=1.10.0
smatch>=1.0.4
coval @ git+https://github.com/ns-moosavi/coval.git
pytest>=7.0.0
```

`.gitignore`:
```
__pycache__/
*.pyc
.pytest_cache/
reports/
```

`README.md`:
```markdown
# Cascade experimental stand

Standalone pipeline: gold CoNLL-U corpus in, cascade-architecture coreference +
graph-construction results out. See
`docs/superpowers/specs/2026-08-02-cascade-experimental-stand-design.md` for
the design.

## Setup

    pip install -r requirements.txt

This pipeline reuses coreference/graph code from the sibling repo
`text-corpuses-processing` (assumed to be checked out at
`../text-corpuses-processing` relative to this repo; override with the
`TEXT_CORPUSES_PROCESSING_DAGS` env var). Follow that repo's own CLAUDE.md
prerequisites (spaCy models, nltk data) to run the LapinLiass/SpacyNeural/
LLMv2 resolvers and graph backends for real; unit tests here mock those
dependencies and don't require them installed.

## Run

    python -m pipeline.run_experiment \
      --corpus data/corpora/sample_en_mini.conllu --language en \
      --resolvers LapinLiass --graph-backends RuleBased \
      --output reports/run1

## Test

    pytest
```

- [ ] **Step 8: Install dependencies and commit**

```bash
pip install -r requirements.txt
git add pipeline tests conftest.py requirements.txt .gitignore README.md
git commit -m "feat: scaffold experimental-stand pipeline package"
```

---

### Task 2: Shared types

**Files:**
- Create: `pipeline/types.py`
- Test: `tests/pipeline/test_types.py`

**Interfaces:**
- Produces: `Token(index, text, start_char, end_char, sent_index, is_empty=False)`, `MentionSpan(start_token, end_token, is_zero=False)`, `CorefDocument(doc_id, text, tokens, clusters)`, `ResolverOutput(resolved_text, clusters)` — every later task imports these.

- [ ] **Step 1: Write the failing test**

```python
# tests/pipeline/test_types.py
from pipeline.types import CorefDocument, MentionSpan, ResolverOutput, Token


def test_mention_span_equality_and_defaults():
    a = MentionSpan(start_token=1, end_token=2)
    b = MentionSpan(start_token=1, end_token=2, is_zero=False)
    assert a == b
    assert a.is_zero is False


def test_coref_document_holds_tokens_and_clusters():
    tok = Token(index=0, text="John", start_char=0, end_char=4, sent_index=0)
    doc = CorefDocument(doc_id="d1", text="John", tokens=[tok], clusters=[])
    assert doc.tokens[0].text == "John"
    assert doc.clusters == []


def test_resolver_output_allows_none_clusters():
    out = ResolverOutput(resolved_text="John left.", clusters=None)
    assert out.clusters is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/pipeline/test_types.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.types'`

- [ ] **Step 3: Write `pipeline/types.py`**

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Token:
    index: int
    text: str
    start_char: int
    end_char: int
    sent_index: int
    is_empty: bool = False


@dataclass(frozen=True)
class MentionSpan:
    start_token: int
    end_token: int  # inclusive
    is_zero: bool = False


@dataclass
class CorefDocument:
    doc_id: str
    text: str
    tokens: list[Token]
    clusters: list[list[MentionSpan]]


@dataclass
class ResolverOutput:
    resolved_text: str
    clusters: list[list[MentionSpan]] | None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/pipeline/test_types.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add pipeline/types.py tests/pipeline/test_types.py
git commit -m "feat: add shared dataclasses for the experimental stand"
```

---

### Task 3: CorefUD loader

**Files:**
- Create: `pipeline/corpus/corefud_loader.py`
- Test: `tests/pipeline/test_corefud_loader.py`

**Interfaces:**
- Consumes: `Token`, `MentionSpan`, `CorefDocument` from `pipeline.types`.
- Produces: `parse_conllu(text: str) -> list[CorefDocument]`, `load_corefud_corpus(path: Path) -> list[CorefDocument]` — consumed by `pipeline/corpus/oracle.py` (Task 4) and `pipeline/run_experiment.py` (Task 15).

**Supported subset (documented limitation):** CoNLL-U with `Entity=` MISC bracket notation (`(eid...`, `eid)`, or combined `(eid...)`); one entity id per `Entity=` value (no same-token nested distinct entities beyond what `|`-separation handles); multiword-token range lines (`ID` containing `-`) are skipped, so corpora relying on them (e.g. Spanish/French contractions) will reconstruct text slightly incorrectly — acceptable for the English/Russian scope this pipeline targets today.

- [ ] **Step 1: Write the failing tests**

```python
# tests/pipeline/test_corefud_loader.py
from pipeline.corpus.corefud_loader import parse_conllu
from pipeline.types import MentionSpan


SINGLE_TOKEN_MENTIONS = """\
# newdoc id = doc1
# sent_id = doc1-1
1\tJohn\tJohn\tPROPN\t_\t_\t2\tnsubj\t_\tEntity=(e1-person-1-new-1-sgl-1)
2\tarrived\tarrive\tVERB\t_\t_\t0\troot\t_\tSpaceAfter=No
3\t.\t.\tPUNCT\t_\t_\t2\tpunct\t_\t_

# sent_id = doc1-2
1\t0\t0\tPRON\t_\t_\t2\tnsubj\t_\tEntity=(e1)
1.1\tPro\tpro\tPRON\t_\t_\t_\t_\t_\t_
2\tleft\tleave\tVERB\t_\t_\t0\troot\t_\tSpaceAfter=No
3\t.\t.\tPUNCT\t_\t_\t2\tpunct\t_\t_
"""


def test_reconstructs_text_with_space_after_no_respected():
    docs = parse_conllu(SINGLE_TOKEN_MENTIONS)
    assert len(docs) == 1
    assert docs[0].text == "John arrived. 0 left."


def test_sentence_indices_increment_per_blank_line():
    doc = parse_conllu(SINGLE_TOKEN_MENTIONS)[0]
    assert doc.tokens[0].sent_index == 0  # John
    assert doc.tokens[3].sent_index == 1  # 0 (second sentence)


def test_empty_node_marked_is_empty_and_not_counted_in_entity_span():
    doc = parse_conllu(SINGLE_TOKEN_MENTIONS)[0]
    empty_tokens = [t for t in doc.tokens if t.is_empty]
    assert len(empty_tokens) == 1
    assert empty_tokens[0].text == ""


def test_two_mentions_of_same_entity_form_one_cluster():
    doc = parse_conllu(SINGLE_TOKEN_MENTIONS)[0]
    assert doc.clusters == [[MentionSpan(0, 0, False), MentionSpan(3, 3, False)]]


MULTI_TOKEN_SPAN = """\
1\tthe\tthe\tDET\t_\t_\t3\tdet\t_\tEntity=(e2-org-3-new-1-sgl-1
2\tUnited\tUnited\tPROPN\t_\t_\t3\tcompound\t_\t_
3\tNations\tNations\tPROPN\t_\t_\t0\troot\t_\tEntity=e2)
"""


def test_multi_token_span_open_and_close_on_different_tokens():
    doc = parse_conllu(MULTI_TOKEN_SPAN)[0]
    assert doc.clusters == [[MentionSpan(0, 2, False)]]


OVERLAPPING_ENTITIES = """\
1\tthe\tthe\tDET\t_\t_\t4\tdet\t_\tEntity=(e3-org-1-new-1-sgl-1
2\tCity\tcity\tPROPN\t_\t_\t4\tcompound\t_\tEntity=(e4-loc-1-new-1-sgl-1)
3\tof\tof\tADP\t_\t_\t4\tcase\t_\t_
4\tLondon\tLondon\tPROPN\t_\t_\t0\troot\t_\tEntity=e3)
"""


def test_overlapping_entities_use_independent_stacks():
    doc = parse_conllu(OVERLAPPING_ENTITIES)[0]
    clusters_by_span = {c[0].start_token: c for c in doc.clusters}
    assert clusters_by_span[0] == [MentionSpan(0, 3, False)]
    assert clusters_by_span[1] == [MentionSpan(1, 1, False)]


def test_multiple_newdoc_blocks_produce_multiple_documents():
    text = (
        "# newdoc id = a\n"
        "1\tHi\thi\tINTJ\t_\t_\t0\troot\t_\t_\n"
        "\n"
        "# newdoc id = b\n"
        "1\tBye\tbye\tINTJ\t_\t_\t0\troot\t_\t_\n"
    )
    docs = parse_conllu(text)
    assert [d.doc_id for d in docs] == ["a", "b"]
    assert docs[0].text == "Hi"
    assert docs[1].text == "Bye"


def test_unclosed_entity_raises():
    text = "1\tJohn\tJohn\tPROPN\t_\t_\t0\troot\t_\tEntity=(e1-person-1\n"
    try:
        parse_conllu(text)
        assert False, "expected ValueError"
    except ValueError:
        pass
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/pipeline/test_corefud_loader.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.corpus.corefud_loader'`

- [ ] **Step 3: Write `pipeline/corpus/corefud_loader.py`**

```python
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from ..types import CorefDocument, MentionSpan, Token

_ENTITY_PIECE_RE = re.compile(r"^(?P<open>\()?(?P<eid>[A-Za-z0-9]+)(?:-[^()]*)?(?P<close>\))?$")


def _parse_entity_misc(misc: str) -> list[tuple[str, bool, bool]]:
    events: list[tuple[str, bool, bool]] = []
    if not misc or misc == "_":
        return events
    for field in misc.split("|"):
        if not field.startswith("Entity="):
            continue
        value = field[len("Entity="):]
        m = _ENTITY_PIECE_RE.match(value)
        if not m:
            continue
        events.append((m.group("eid"), bool(m.group("open")), bool(m.group("close"))))
    return events


def _misc_has_no_space_after(misc: str) -> bool:
    return misc != "_" and "SpaceAfter=No" in misc.split("|")


def parse_conllu(text: str) -> list[CorefDocument]:
    """Parse CoNLL-U with CorefUD Entity= bracket annotations into documents.

    Splits on '# newdoc' comment lines; if none are present, the whole input
    is treated as a single document named "doc1".
    """
    doc_blocks: list[list[str]] = []
    doc_ids: list[str] = []
    current: list[str] = []

    for line in text.splitlines():
        if line.startswith("# newdoc"):
            if current:
                doc_blocks.append(current)
                current = []
            m = re.search(r"newdoc(?:\s+id\s*=\s*(\S+))?", line)
            doc_id = m.group(1) if m and m.group(1) else f"doc{len(doc_blocks) + 1}"
            doc_ids.append(doc_id)
            continue
        current.append(line)

    if current:
        doc_blocks.append(current)
    if not doc_ids:
        doc_ids = ["doc1"]

    return [
        _parse_document_block(doc_id, block_lines)
        for doc_id, block_lines in zip(doc_ids, doc_blocks)
    ]


def _parse_document_block(doc_id: str, lines: list[str]) -> CorefDocument:
    tokens: list[Token] = []
    open_stack: dict[str, list[int]] = defaultdict(list)
    clusters_by_eid: dict[str, list[MentionSpan]] = defaultdict(list)

    text_parts: list[str] = []
    char_pos = 0
    sent_index = 0
    pending_space = False
    seen_token_in_sentence = False

    for line in lines:
        line = line.rstrip("\n")
        if line.startswith("#"):
            continue
        if not line.strip():
            if seen_token_in_sentence:
                sent_index += 1
                seen_token_in_sentence = False
            continue

        cols = line.split("\t")
        if len(cols) != 10:
            continue
        tok_id, form, _lemma, _upos, _xpos, _feats, _head, _deprel, _deps, misc = cols

        if "-" in tok_id:
            continue  # multiword-token range line: see module docstring limitation

        is_empty = "." in tok_id

        if pending_space:
            text_parts.append(" ")
            char_pos += 1
            pending_space = False

        idx = len(tokens)
        if is_empty:
            start_char = end_char = char_pos
        else:
            text_parts.append(form)
            start_char = char_pos
            char_pos += len(form)
            end_char = char_pos
            pending_space = not _misc_has_no_space_after(misc)

        seen_token_in_sentence = True
        tokens.append(Token(
            index=idx,
            text="" if is_empty else form,
            start_char=start_char,
            end_char=end_char,
            sent_index=sent_index,
            is_empty=is_empty,
        ))

        for eid, is_open, is_close in _parse_entity_misc(misc):
            if is_open:
                open_stack[eid].append(idx)
            if is_close:
                if not open_stack[eid]:
                    raise ValueError(f"close without open for entity {eid} in doc {doc_id}")
                start = open_stack[eid].pop()
                clusters_by_eid[eid].append(
                    MentionSpan(
                        start_token=start,
                        end_token=idx,
                        is_zero=(start == idx and tokens[start].is_empty),
                    )
                )

    for eid, stack in open_stack.items():
        if stack:
            raise ValueError(f"unclosed entity {eid} in doc {doc_id}")

    clusters = [spans for spans in clusters_by_eid.values() if spans]
    return CorefDocument(doc_id=doc_id, text="".join(text_parts), tokens=tokens, clusters=clusters)


def load_corefud_corpus(path: str | Path) -> list[CorefDocument]:
    path = Path(path)
    if path.is_dir():
        documents: list[CorefDocument] = []
        for file_path in sorted(path.glob("*.conllu")):
            documents.extend(parse_conllu(file_path.read_text(encoding="utf-8")))
        return documents
    return parse_conllu(path.read_text(encoding="utf-8"))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/pipeline/test_corefud_loader.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add pipeline/corpus/corefud_loader.py tests/pipeline/test_corefud_loader.py
git commit -m "feat: add CorefUD Entity= bracket-notation loader"
```

---

### Task 4: Oracle text builder

**Files:**
- Create: `pipeline/corpus/oracle.py`
- Test: `tests/pipeline/test_oracle.py`

**Interfaces:**
- Consumes: `CorefDocument`, `MentionSpan` from `pipeline.types`.
- Produces: `build_oracle_text(doc: CorefDocument) -> str` — consumed by `pipeline/run_experiment.py` (Task 15).

**Convention:** the cluster's first mention (document order) is treated as the head; every other mention in the cluster is replaced by the head's surface text. Zero/empty mentions are skipped (nothing to substitute in the surface text for a token with no surface form).

- [ ] **Step 1: Write the failing tests**

```python
# tests/pipeline/test_oracle.py
from pipeline.corpus.oracle import build_oracle_text
from pipeline.types import CorefDocument, MentionSpan, Token


def _doc(text, tokens, clusters):
    return CorefDocument(doc_id="d1", text=text, tokens=tokens, clusters=clusters)


def test_substitutes_non_head_mentions_with_head_surface_text():
    text = "John arrived. He left."
    tokens = [
        Token(0, "John", 0, 4, 0),
        Token(1, "arrived", 5, 12, 0),
        Token(2, ".", 12, 13, 0),
        Token(3, "He", 14, 16, 1),
        Token(4, "left", 17, 21, 1),
        Token(5, ".", 21, 22, 1),
    ]
    clusters = [[MentionSpan(0, 0), MentionSpan(3, 3)]]
    doc = _doc(text, tokens, clusters)
    assert build_oracle_text(doc) == "John arrived. John left."


def test_singleton_clusters_are_left_unchanged():
    text = "John left."
    tokens = [Token(0, "John", 0, 4, 0), Token(1, "left", 5, 9, 0), Token(2, ".", 9, 10, 0)]
    doc = _doc(text, tokens, clusters=[[MentionSpan(0, 0)]])
    assert build_oracle_text(doc) == "John left."


def test_zero_mentions_are_skipped_not_substituted_into_text():
    text = "John arrived.  left."
    tokens = [
        Token(0, "John", 0, 4, 0),
        Token(1, "arrived", 5, 12, 0),
        Token(2, ".", 12, 13, 0),
        Token(3, "", 14, 14, 1, is_empty=True),
        Token(4, "left", 15, 19, 1),
        Token(5, ".", 19, 20, 1),
    ]
    clusters = [[MentionSpan(0, 0), MentionSpan(3, 3, is_zero=True)]]
    doc = _doc(text, tokens, clusters)
    assert build_oracle_text(doc) == text  # unchanged: nothing to replace at a zero-width span


def test_multiple_clusters_applied_without_offset_corruption():
    text = "John met Mary. He greeted her."
    tokens = [
        Token(0, "John", 0, 4, 0), Token(1, "met", 5, 8, 0), Token(2, "Mary", 9, 13, 0), Token(3, ".", 13, 14, 0),
        Token(4, "He", 15, 17, 1), Token(5, "greeted", 18, 25, 1), Token(6, "her", 26, 29, 1), Token(7, ".", 29, 30, 1),
    ]
    clusters = [[MentionSpan(0, 0), MentionSpan(4, 4)], [MentionSpan(2, 2), MentionSpan(6, 6)]]
    doc = _doc(text, tokens, clusters)
    assert build_oracle_text(doc) == "John met Mary. John greeted Mary."
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/pipeline/test_oracle.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.corpus.oracle'`

- [ ] **Step 3: Write `pipeline/corpus/oracle.py`**

```python
from __future__ import annotations

from ..types import CorefDocument, MentionSpan


def _span_text(doc: CorefDocument, span: MentionSpan) -> str:
    start_char = doc.tokens[span.start_token].start_char
    end_char = doc.tokens[span.end_token].end_char
    return doc.text[start_char:end_char]


def build_oracle_text(doc: CorefDocument) -> str:
    """Replace every non-head mention in each gold cluster with the head's
    surface text. Head = the cluster's first mention in document order."""
    replacements: list[tuple[int, int, str]] = []

    for cluster in doc.clusters:
        if len(cluster) < 2:
            continue
        head_text = _span_text(doc, cluster[0])
        for mention in cluster[1:]:
            if mention.is_zero:
                continue
            start_char = doc.tokens[mention.start_token].start_char
            end_char = doc.tokens[mention.end_token].end_char
            replacements.append((start_char, end_char, head_text))

    replacements.sort(key=lambda r: r[0], reverse=True)
    out = doc.text
    for start, end, replacement in replacements:
        out = out[:start] + replacement + out[end:]
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/pipeline/test_oracle.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add pipeline/corpus/oracle.py tests/pipeline/test_oracle.py
git commit -m "feat: add oracle-resolved-text builder from gold clusters"
```

---

### Task 5: Resolver base — Protocol, span projection, union-find

**Files:**
- Create: `pipeline/resolvers/base.py`
- Test: `tests/pipeline/test_resolvers_base.py`

**Interfaces:**
- Consumes: `Token`, `MentionSpan` from `pipeline.types`.
- Produces: `ResolverAdapter` (Protocol with `name: str`, `language_support: set[str] | Literal["any"]`, `resolve(doc) -> ResolverOutput`), `supports_language(adapter, language: str) -> bool`, `project_char_span_to_gold_tokens(start_char, end_char, gold_tokens) -> MentionSpan | None`, `UnionFind` class with `union(a, b)` and `groups() -> list[list[int]]` — consumed by Tasks 6, 7, 8, and `run_experiment.py` (Task 15).

- [ ] **Step 1: Write the failing tests**

```python
# tests/pipeline/test_resolvers_base.py
from pipeline.resolvers.base import UnionFind, project_char_span_to_gold_tokens, supports_language
from pipeline.types import MentionSpan, Token


def test_project_char_span_picks_overlapping_gold_tokens():
    gold_tokens = [
        Token(0, "John", 0, 4, 0),
        Token(1, "arrived", 5, 12, 0),
    ]
    span = project_char_span_to_gold_tokens(0, 4, gold_tokens)
    assert span == MentionSpan(start_token=0, end_token=0)


def test_project_char_span_spans_multiple_gold_tokens():
    gold_tokens = [
        Token(0, "the", 0, 3, 0),
        Token(1, "United", 4, 10, 0),
        Token(2, "Nations", 11, 18, 0),
    ]
    span = project_char_span_to_gold_tokens(4, 18, gold_tokens)
    assert span == MentionSpan(start_token=1, end_token=2)


def test_project_char_span_returns_none_when_no_overlap():
    gold_tokens = [Token(0, "John", 0, 4, 0)]
    assert project_char_span_to_gold_tokens(10, 14, gold_tokens) is None


def test_project_char_span_skips_empty_tokens():
    gold_tokens = [
        Token(0, "John", 0, 4, 0),
        Token(1, "", 4, 4, 0, is_empty=True),
    ]
    span = project_char_span_to_gold_tokens(4, 4, gold_tokens)
    assert span is None


def test_union_find_groups_transitively_linked_indices():
    uf = UnionFind()
    uf.union(3, 7)
    uf.union(7, 12)
    uf.union(0, 1)
    groups = {frozenset(g) for g in uf.groups()}
    assert frozenset({3, 7, 12}) in groups
    assert frozenset({0, 1}) in groups


class _FakeAdapter:
    language_support = {"en"}


class _AnyLanguageAdapter:
    language_support = "any"


def test_supports_language_checks_set_membership():
    assert supports_language(_FakeAdapter(), "en") is True
    assert supports_language(_FakeAdapter(), "ru") is False


def test_supports_language_any_accepts_everything():
    assert supports_language(_AnyLanguageAdapter(), "ru") is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/pipeline/test_resolvers_base.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.resolvers.base'`

- [ ] **Step 3: Write `pipeline/resolvers/base.py`**

```python
from __future__ import annotations

from typing import Literal, Protocol

from ..types import CorefDocument, MentionSpan, ResolverOutput, Token


class ResolverAdapter(Protocol):
    name: str
    language_support: set[str] | Literal["any"]

    def resolve(self, doc: CorefDocument) -> ResolverOutput: ...


def supports_language(adapter, language: str) -> bool:
    return adapter.language_support == "any" or language in adapter.language_support


def project_char_span_to_gold_tokens(
    start_char: int, end_char: int, gold_tokens: list[Token]
) -> MentionSpan | None:
    overlapping = [
        tok.index for tok in gold_tokens
        if not tok.is_empty and tok.start_char < end_char and tok.end_char > start_char
    ]
    if not overlapping:
        return None
    return MentionSpan(start_token=min(overlapping), end_token=max(overlapping))


class UnionFind:
    def __init__(self) -> None:
        self._parent: dict[int, int] = {}

    def _find(self, x: int) -> int:
        self._parent.setdefault(x, x)
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self._find(a), self._find(b)
        if ra != rb:
            self._parent[ra] = rb

    def groups(self) -> list[list[int]]:
        buckets: dict[int, list[int]] = {}
        for x in self._parent:
            buckets.setdefault(self._find(x), []).append(x)
        return list(buckets.values())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/pipeline/test_resolvers_base.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add pipeline/resolvers/base.py tests/pipeline/test_resolvers_base.py
git commit -m "feat: add resolver adapter protocol, span projection, union-find"
```

---

### Task 6: LapinLiassAdapter

**Files:**
- Create: `pipeline/resolvers/lapin_liass_adapter.py`
- Test: `tests/pipeline/test_lapin_liass_adapter.py`

**Interfaces:**
- Consumes: `add_text_corpuses_processing_to_path` (Task 1), `project_char_span_to_gold_tokens`, `UnionFind` (Task 5), `CorefDocument`, `ResolverOutput`, `MentionSpan` (Task 2).
- Produces: `LapinLiassAdapter` class with `name = "LapinLiass"`, `language_support = {"en"}`, `resolve(doc) -> ResolverOutput` — consumed by `run_experiment.py` (Task 15).

**Testing approach:** `anaphoraResolverLapinLiass.py` calls `spacy.load("en_core_web_sm", ...)` at module import time, so real imports require that model installed. To keep unit tests independent of spaCy, inject a fake module into `sys.modules["anaphoraResolverLapinLiass"]` before calling `resolve()` — the adapter's lazy `from anaphoraResolverLapinLiass import ...` inside `resolve()` picks up the fake from `sys.modules` without touching disk.

- [ ] **Step 1: Write the failing test**

```python
# tests/pipeline/test_lapin_liass_adapter.py
import sys
import types
from dataclasses import dataclass
from typing import Optional

from pipeline.resolvers.lapin_liass_adapter import LapinLiassAdapter
from pipeline.types import CorefDocument, MentionSpan, Token


@dataclass
class _FakeSpacyToken:
    idx: int
    text: str


class _FakeSpacyDoc:
    def __init__(self, tokens):
        self._tokens = tokens

    def __getitem__(self, i):
        return self._tokens[i]


@dataclass
class _FakeResolution:
    pronoun: str
    pronoun_index: int
    antecedent: Optional[str]
    antecedent_index: Optional[int]
    score: float


class _FakeBatchAnaphoraResolver:
    def resolve_document(self, text):
        # "John arrived. He left." -> spaCy tokens (word-level, char offsets into `text`)
        tokens = [
            _FakeSpacyToken(0, "John"), _FakeSpacyToken(5, "arrived"), _FakeSpacyToken(12, "."),
            _FakeSpacyToken(14, "He"), _FakeSpacyToken(17, "left"), _FakeSpacyToken(21, "."),
        ]
        doc = _FakeSpacyDoc(tokens)
        resolutions = [
            _FakeResolution(pronoun="He", pronoun_index=3, antecedent="John", antecedent_index=0, score=100.0)
        ]
        return {"text": text, "doc": doc, "resolutions": resolutions}


def _fake_build_substitutions(doc, resolutions, mark=False):
    subs = []
    for r in resolutions:
        if r.antecedent is None:
            continue
        tok = doc[r.pronoun_index]
        subs.append(types.SimpleNamespace(
            start=tok.idx, end=tok.idx + len(tok.text), replacement=r.antecedent,
        ))
    subs.sort(key=lambda s: s.start, reverse=True)
    return subs


def _fake_apply_substitutions(text, substitutions):
    out = text
    for s in substitutions:
        out = out[: s.start] + s.replacement + out[s.end :]
    return out


def _install_fake_module(monkeypatch):
    fake = types.ModuleType("anaphoraResolverLapinLiass")
    fake.BatchAnaphoraResolver = _FakeBatchAnaphoraResolver
    fake.build_substitutions = _fake_build_substitutions
    fake.apply_substitutions = _fake_apply_substitutions
    monkeypatch.setitem(sys.modules, "anaphoraResolverLapinLiass", fake)


def test_resolve_projects_clusters_onto_gold_tokens_and_substitutes_text(monkeypatch):
    _install_fake_module(monkeypatch)

    text = "John arrived. He left."
    gold_tokens = [
        Token(0, "John", 0, 4, 0), Token(1, "arrived", 5, 12, 0), Token(2, ".", 12, 13, 0),
        Token(3, "He", 14, 16, 1), Token(4, "left", 17, 21, 1), Token(5, ".", 21, 22, 1),
    ]
    doc = CorefDocument(doc_id="d1", text=text, tokens=gold_tokens, clusters=[])

    adapter = LapinLiassAdapter()
    result = adapter.resolve(doc)

    assert result.resolved_text == "John arrived. John left."
    assert result.clusters == [[MentionSpan(0, 0), MentionSpan(3, 3)]]


def test_adapter_metadata():
    adapter = LapinLiassAdapter()
    assert adapter.name == "LapinLiass"
    assert adapter.language_support == {"en"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/pipeline/test_lapin_liass_adapter.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.resolvers.lapin_liass_adapter'`

- [ ] **Step 3: Write `pipeline/resolvers/lapin_liass_adapter.py`**

```python
from __future__ import annotations

from ..sys_path_setup import add_text_corpuses_processing_to_path
from ..types import CorefDocument, MentionSpan, ResolverOutput
from .base import UnionFind, project_char_span_to_gold_tokens


class LapinLiassAdapter:
    name = "LapinLiass"
    language_support = {"en"}

    def resolve(self, doc: CorefDocument) -> ResolverOutput:
        add_text_corpuses_processing_to_path()
        from anaphoraResolverLapinLiass import (
            BatchAnaphoraResolver,
            apply_substitutions,
            build_substitutions,
        )

        resolver = BatchAnaphoraResolver()
        result = resolver.resolve_document(doc.text)
        spacy_doc = result["doc"]
        resolutions = result["resolutions"]

        substitutions = build_substitutions(spacy_doc, resolutions, mark=False)
        resolved_text = apply_substitutions(doc.text, substitutions)

        uf = UnionFind()
        for r in resolutions:
            if r.antecedent_index is not None:
                uf.union(r.pronoun_index, r.antecedent_index)

        clusters: list[list[MentionSpan]] = []
        for group in uf.groups():
            if len(group) < 2:
                continue
            projected = []
            for token_idx in group:
                tok = spacy_doc[token_idx]
                span = project_char_span_to_gold_tokens(tok.idx, tok.idx + len(tok.text), doc.tokens)
                if span is not None:
                    projected.append(span)
            if len(projected) >= 2:
                clusters.append(projected)

        return ResolverOutput(resolved_text=resolved_text, clusters=clusters)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/pipeline/test_lapin_liass_adapter.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add pipeline/resolvers/lapin_liass_adapter.py tests/pipeline/test_lapin_liass_adapter.py
git commit -m "feat: add LapinLiass resolver adapter"
```

---

### Task 7: SpacyNeuralAdapter

**Files:**
- Create: `pipeline/resolvers/spacy_neural_adapter.py`
- Test: `tests/pipeline/test_spacy_neural_adapter.py`

**Interfaces:**
- Consumes: same as Task 6.
- Produces: `SpacyNeuralAdapter` class, `name = "SpacyNeural"`, `language_support = {"en"}`, `resolve(doc) -> ResolverOutput`.

- [ ] **Step 1: Write the failing test**

```python
# tests/pipeline/test_spacy_neural_adapter.py
import sys
import types
from dataclasses import dataclass

from pipeline.resolvers.spacy_neural_adapter import SpacyNeuralAdapter
from pipeline.types import CorefDocument, MentionSpan, Token


@dataclass
class _FakeSpan:
    start_char: int
    end_char: int
    text: str


class _FakeSpacyDoc:
    def __init__(self, spans_dict):
        self.spans = spans_dict


def _fake_nlp(text):
    # cluster: "John" (0-4) and "He" (14-16) corefer
    return _FakeSpacyDoc({
        "coref_clusters_1": [_FakeSpan(0, 4, "John"), _FakeSpan(14, 16, "He")],
    })


def _fake_resolve_and_substitute(text, mark=False):
    resolved = text[:14] + "John" + text[16:]
    return resolved, [], []


def _install_fake_module(monkeypatch):
    fake = types.ModuleType("anaphoraResolverSpacyNeural")
    fake.resolve_and_substitute = _fake_resolve_and_substitute
    fake._get_nlp = lambda: _fake_nlp
    monkeypatch.setitem(sys.modules, "anaphoraResolverSpacyNeural", fake)


def test_resolve_builds_clusters_from_native_coref_spans(monkeypatch):
    _install_fake_module(monkeypatch)

    text = "John arrived. He left."
    gold_tokens = [
        Token(0, "John", 0, 4, 0), Token(1, "arrived", 5, 12, 0), Token(2, ".", 12, 13, 0),
        Token(3, "He", 14, 16, 1), Token(4, "left", 17, 21, 1), Token(5, ".", 21, 22, 1),
    ]
    doc = CorefDocument(doc_id="d1", text=text, tokens=gold_tokens, clusters=[])

    adapter = SpacyNeuralAdapter()
    result = adapter.resolve(doc)

    assert result.resolved_text == "John arrived. John left."
    assert result.clusters == [[MentionSpan(0, 0), MentionSpan(3, 3)]]


def test_adapter_metadata():
    adapter = SpacyNeuralAdapter()
    assert adapter.name == "SpacyNeural"
    assert adapter.language_support == {"en"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/pipeline/test_spacy_neural_adapter.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.resolvers.spacy_neural_adapter'`

- [ ] **Step 3: Write `pipeline/resolvers/spacy_neural_adapter.py`**

```python
from __future__ import annotations

from ..sys_path_setup import add_text_corpuses_processing_to_path
from ..types import CorefDocument, MentionSpan, ResolverOutput
from .base import project_char_span_to_gold_tokens


class SpacyNeuralAdapter:
    name = "SpacyNeural"
    language_support = {"en"}

    def resolve(self, doc: CorefDocument) -> ResolverOutput:
        add_text_corpuses_processing_to_path()
        from anaphoraResolverSpacyNeural import _get_nlp, resolve_and_substitute

        resolved_text, _, _ = resolve_and_substitute(doc.text)

        nlp = _get_nlp()
        spacy_doc = nlp(doc.text)
        spacy_clusters = [
            spans for key, spans in spacy_doc.spans.items()
            if key.startswith("coref_clusters")
        ]

        clusters: list[list[MentionSpan]] = []
        for cluster in spacy_clusters:
            if len(cluster) < 2:
                continue
            projected = []
            for mention_span in cluster:
                span = project_char_span_to_gold_tokens(
                    mention_span.start_char, mention_span.end_char, doc.tokens
                )
                if span is not None:
                    projected.append(span)
            if len(projected) >= 2:
                clusters.append(projected)

        return ResolverOutput(resolved_text=resolved_text, clusters=clusters)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/pipeline/test_spacy_neural_adapter.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add pipeline/resolvers/spacy_neural_adapter.py tests/pipeline/test_spacy_neural_adapter.py
git commit -m "feat: add SpacyNeural resolver adapter"
```

---

### Task 8: LLMv2Adapter (coreference-only)

**Files:**
- Create: `pipeline/resolvers/llm_v2_adapter.py`
- Test: `tests/pipeline/test_llm_v2_adapter.py`

**Interfaces:**
- Consumes: `add_text_corpuses_processing_to_path` (Task 1), `CorefDocument`, `ResolverOutput` (Task 2).
- Produces: `LLMv2Adapter(llm_client, language, base_dir=None)` class, `name = "LLMv2"`, `language_support = "any"`, `resolve(doc) -> ResolverOutput` with `clusters` always `None` (documented limitation — no mention spans available from this resolver, per design spec Section 2).

**Testing approach:** inject fake `llm_v2.stages.preprocessing`, `llm_v2.stages.coreference`, and `llm_v2.config_schema` modules into `sys.modules` (these import `torch`/`transformers` transitively in the real repo; tests must not require those installed).

- [ ] **Step 1: Write the failing test**

```python
# tests/pipeline/test_llm_v2_adapter.py
import sys
import types

from pipeline.resolvers.llm_v2_adapter import LLMv2Adapter
from pipeline.types import CorefDocument, Token


class _FakeLLMClient:
    def generate(self, prompt):
        return "unused"


def _install_fake_modules(monkeypatch, captured):
    preprocessing_mod = types.ModuleType("llm_v2.stages.preprocessing")

    def _fake_preprocess(text, language="ru"):
        captured["preprocess_language"] = language
        return [types.SimpleNamespace(id=0, text=text)]

    preprocessing_mod.preprocess = _fake_preprocess

    coreference_mod = types.ModuleType("llm_v2.stages.coreference")

    def _fake_resolve_coreferences(sentences, llm, config, base_dir=None):
        captured["config_prompt_file"] = config.prompt_file
        captured["base_dir"] = base_dir
        return "John arrived. John left.", sentences

    coreference_mod.resolve_coreferences = _fake_resolve_coreferences

    config_schema_mod = types.ModuleType("llm_v2.config_schema")

    class _FakeCoreferenceConfig:
        def __init__(self, enabled=True, prompt_file=""):
            self.enabled = enabled
            self.prompt_file = prompt_file

    config_schema_mod.CoreferenceConfig = _FakeCoreferenceConfig

    monkeypatch.setitem(sys.modules, "llm_v2", types.ModuleType("llm_v2"))
    monkeypatch.setitem(sys.modules, "llm_v2.stages", types.ModuleType("llm_v2.stages"))
    monkeypatch.setitem(sys.modules, "llm_v2.stages.preprocessing", preprocessing_mod)
    monkeypatch.setitem(sys.modules, "llm_v2.stages.coreference", coreference_mod)
    monkeypatch.setitem(sys.modules, "llm_v2.config_schema", config_schema_mod)


def test_resolve_returns_resolved_text_and_none_clusters(monkeypatch):
    captured: dict = {}
    _install_fake_modules(monkeypatch, captured)

    doc = CorefDocument(
        doc_id="d1", text="John arrived. He left.",
        tokens=[Token(0, "John", 0, 4, 0)], clusters=[],
    )
    adapter = LLMv2Adapter(llm_client=_FakeLLMClient(), language="en")
    result = adapter.resolve(doc)

    assert result.resolved_text == "John arrived. John left."
    assert result.clusters is None
    assert captured["preprocess_language"] == "en"
    assert captured["config_prompt_file"] == "prompts/coreference_en.txt"


def test_adapter_metadata():
    adapter = LLMv2Adapter(llm_client=_FakeLLMClient(), language="en")
    assert adapter.name == "LLMv2"
    assert adapter.language_support == "any"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/pipeline/test_llm_v2_adapter.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.resolvers.llm_v2_adapter'`

- [ ] **Step 3: Write `pipeline/resolvers/llm_v2_adapter.py`**

```python
from __future__ import annotations

from pathlib import Path

from ..sys_path_setup import add_text_corpuses_processing_to_path
from ..types import CorefDocument, ResolverOutput


class LLMv2Adapter:
    name = "LLMv2"
    language_support = "any"

    def __init__(self, llm_client, language: str, base_dir: Path | None = None):
        self.llm_client = llm_client
        self.language = language
        dags_dir = add_text_corpuses_processing_to_path()
        self.base_dir = base_dir or (dags_dir / "llm_v2")

    def resolve(self, doc: CorefDocument) -> ResolverOutput:
        add_text_corpuses_processing_to_path()
        from llm_v2.config_schema import CoreferenceConfig
        from llm_v2.stages.coreference import resolve_coreferences
        from llm_v2.stages.preprocessing import preprocess

        sentences = preprocess(doc.text, language=self.language)
        config = CoreferenceConfig(
            enabled=True,
            prompt_file=f"prompts/coreference_{self.language}.txt",
        )
        resolved_text, _ = resolve_coreferences(
            sentences, self.llm_client, config, base_dir=self.base_dir
        )
        # No mention-span output from this resolver (it only rewrites text) --
        # clusters=None means coreference F1 is not computable for this
        # adapter; only the graph-level oracle-ablation applies. See design
        # spec Section 2.
        return ResolverOutput(resolved_text=resolved_text, clusters=None)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/pipeline/test_llm_v2_adapter.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add pipeline/resolvers/llm_v2_adapter.py tests/pipeline/test_llm_v2_adapter.py
git commit -m "feat: add LLMv2 coreference resolver adapter (text-only, no clusters)"
```

---

### Task 9: Graph base Protocol + RuleBasedGraphAdapter

**Files:**
- Create: `pipeline/graph/base.py`
- Create: `pipeline/graph/rule_based_graph_adapter.py`
- Test: `tests/pipeline/test_rule_based_graph_adapter.py`

**Interfaces:**
- Produces: `GraphBackendAdapter` Protocol (`name: str`, `backend_name: str`, `language_support`, `build(text: str) -> dict`); `RuleBasedGraphAdapter` class, `name = backend_name = "RuleBased"`, `language_support = {"en"}` — consumed by `run_experiment.py` (Task 15) and `graph_scoring.py` (Task 11, via `backend_name`).

- [ ] **Step 1: Write the failing test**

```python
# tests/pipeline/test_rule_based_graph_adapter.py
import sys
import types

from pipeline.graph.rule_based_graph_adapter import RuleBasedGraphAdapter


def _install_fake_module(monkeypatch):
    fake = types.ModuleType("graphBuilder")

    def _fake_extract_graph_edges(text):
        return [("john", "arrive", "")] if "John" in text else []

    def _fake_merge_graph(graph, new_edges):
        nodes = set(graph["nodes"])
        for a1, a2, meaning in new_edges:
            nodes.add(a1)
            nodes.add(a2)
            graph["edges"].append({"agent_1": a1, "agent_2": a2, "meaning": meaning, "weight": 1})
        graph["nodes"] = list(nodes)
        return graph

    fake.extract_graph_edges = _fake_extract_graph_edges
    fake.merge_graph = _fake_merge_graph
    monkeypatch.setitem(sys.modules, "graphBuilder", fake)


def test_build_returns_nodes_and_edges_shape(monkeypatch):
    _install_fake_module(monkeypatch)
    adapter = RuleBasedGraphAdapter()
    graph = adapter.build("John arrived.")
    assert set(graph["nodes"]) == {"john", "arrive"}
    assert graph["edges"] == [{"agent_1": "john", "agent_2": "arrive", "meaning": "", "weight": 1}]


def test_adapter_metadata():
    adapter = RuleBasedGraphAdapter()
    assert adapter.name == "RuleBased"
    assert adapter.backend_name == "RuleBased"
    assert adapter.language_support == {"en"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/pipeline/test_rule_based_graph_adapter.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.graph.rule_based_graph_adapter'`

- [ ] **Step 3: Write `pipeline/graph/base.py`**

```python
from __future__ import annotations

from typing import Literal, Protocol


class GraphBackendAdapter(Protocol):
    name: str
    backend_name: str
    language_support: set[str] | Literal["any"]

    def build(self, text: str) -> dict: ...
```

- [ ] **Step 4: Write `pipeline/graph/rule_based_graph_adapter.py`**

```python
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/pipeline/test_rule_based_graph_adapter.py -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Commit**

```bash
git add pipeline/graph/base.py pipeline/graph/rule_based_graph_adapter.py tests/pipeline/test_rule_based_graph_adapter.py
git commit -m "feat: add graph backend protocol and rule-based graph adapter"
```

---

### Task 10: `penman_convert.py` (graph → AMR-line for Smatch)

**Files:**
- Create: `pipeline/graph/penman_convert.py`
- Test: `tests/pipeline/test_penman_convert.py`

**Interfaces:**
- Produces: `graph_to_amr_line(nodes: dict[str, str], edges: list[tuple[str, str, str]], var_prefix: str) -> str` — consumed by `graph_scoring.py` (Task 11).

**Design note (verified by hand):** every node is wrapped under one synthetic root (`{var_prefix}top / graph-root`) with a `:has-entityN` edge to each node, so disconnected components and cycles serialize correctly. Each node is declared (`var / concept`) exactly once on first encounter; later references (including cycles) emit the bare variable name (AMR reentrancy) instead of redeclaring. Relation labels ending in `-of` are escaped (`-of_`) because `smatch`'s AMR parser treats trailing `-of` as a reversed-relation marker, which would silently flip triple direction for unrelated KG relations like "made-of" or "part-of".

- [ ] **Step 1: Write the failing tests**

```python
# tests/pipeline/test_penman_convert.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/pipeline/test_penman_convert.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.graph.penman_convert'`

- [ ] **Step 3: Write `pipeline/graph/penman_convert.py`**

```python
from __future__ import annotations

from collections import defaultdict


def _sanitize_concept(label: str) -> str:
    s = label.strip().lower().replace(" ", "_").replace("(", "").replace(")", "").replace('"', "")
    return s or "concept"


def _sanitize_rel(rel: str) -> str:
    s = rel.strip().lower().replace(" ", "-").replace("(", "").replace(")", "").replace('"', "")
    if not s:
        s = "rel"
    if s.endswith("-of"):
        # smatch's AMR parser treats a trailing "-of" as "this relation is the
        # inverse of <rel without -of>" and silently flips triple direction.
        # Our KG relation labels aren't AMR roles, so escape to avoid that.
        s = s + "_"
    return s


def graph_to_amr_line(nodes: dict[str, str], edges: list[tuple[str, str, str]], var_prefix: str) -> str:
    """Serialize a (nodes, edges) graph into a single-line AMR/Penman string
    that smatch.get_amr_match() can parse.

    nodes: {node_id: label}
    edges: [(source_id, target_id, relation_label), ...]

    Every node is attached under one synthetic root via :has-entityN edges so
    disconnected components serialize correctly; each node is declared once
    and later references (including cycles) use bare-variable reentrancy.
    """
    node_ids = list(nodes.keys())
    if not node_ids:
        return f"({var_prefix}top / empty-graph)"

    var_of = {nid: f"{var_prefix}{i}" for i, nid in enumerate(node_ids)}
    outgoing: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for src, tgt, rel in edges:
        if src in var_of and tgt in var_of:
            outgoing[src].append((_sanitize_rel(rel), tgt))

    declared: set[str] = set()

    def render(node_id: str) -> str:
        var = var_of[node_id]
        if var in declared:
            return var
        declared.add(var)
        concept = _sanitize_concept(nodes[node_id])
        parts = [f"({var} / {concept}"]
        for rel, tgt in outgoing.get(node_id, []):
            parts.append(f":{rel} {render(tgt)}")
        parts.append(")")
        return " ".join(parts)

    children = " ".join(f":has-entity{i} {render(nid)}" for i, nid in enumerate(node_ids))
    return f"({var_prefix}top / graph-root {children})"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/pipeline/test_penman_convert.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add pipeline/graph/penman_convert.py tests/pipeline/test_penman_convert.py
git commit -m "feat: add graph-to-AMR-line converter for smatch scoring"
```

---

### Task 11: `graph_scoring.py` (node/edge P/R, duplication rate, Smatch)

**Files:**
- Create: `pipeline/eval/graph_scoring.py`
- Test: `tests/pipeline/test_graph_scoring.py`

**Interfaces:**
- Consumes: `add_text_corpuses_processing_to_path` (Task 1), `graph_to_amr_line` (Task 10). Reuses `graphMetrics._to_networkx` from `text-corpuses-processing` (a private helper, intentionally reused rather than duplicated — see module docstring).
- Produces: `compute_graph_scores(oracle_graph: dict, predicted_graph: dict, backend: str) -> dict`, `compute_node_duplication_rate(graph_dict: dict, backend: str) -> float` — consumed by `run_experiment.py` (Task 15).

**Testing approach:** inject a fake `graphMetrics` module into `sys.modules` exposing a minimal `_to_networkx` so tests don't require `networkx`'s real behavior beyond what we control — actually, since `networkx` itself is a direct dependency of this repo (`requirements.txt`, Task 1), tests use the **real** `graphMetrics._to_networkx` reused as-is (it's pure Python + networkx, no heavy model dependency), by injecting a fake `sys.modules["graphMetrics"]` only if `text-corpuses-processing` isn't checked out in the test environment. To keep tests simple and not depend on `text-corpuses-processing` being present, write a small local reimplementation of the two-line `_to_networkx` dispatch inline in the fake module used by tests (matching the reused module's known shape from the design spec), verifying our own scoring logic in isolation from the reused module's own correctness (which is that module's existing test suite's job, not ours).

- [ ] **Step 1: Write the failing tests**

```python
# tests/pipeline/test_graph_scoring.py
import sys
import types

import networkx as nx

from pipeline.eval.graph_scoring import compute_graph_scores, compute_node_duplication_rate


def _install_fake_graph_metrics(monkeypatch):
    fake = types.ModuleType("graphMetrics")

    def _to_networkx(graph_dict, backend):
        G = nx.Graph()
        if backend == "RuleBased":
            for node in graph_dict.get("nodes", []):
                G.add_node(node)
            for edge in graph_dict.get("edges", []):
                G.add_edge(edge["agent_1"], edge["agent_2"], weight=edge.get("weight", 1), label=edge.get("meaning", ""))
        else:
            for node in graph_dict.get("nodes", []):
                G.add_node(node["id"], label=node["label"])
            for edge in graph_dict.get("edges", []):
                G.add_edge(edge["source"], edge["target"], weight=edge.get("weight", 1), label=edge.get("label", ""))
        return G

    fake._to_networkx = _to_networkx
    monkeypatch.setitem(sys.modules, "graphMetrics", fake)


def _rule_based_graph(nodes, edges):
    return {"nodes": nodes, "edges": [{"agent_1": a, "agent_2": b, "meaning": m, "weight": 1} for a, b, m in edges]}


def test_identical_graphs_score_perfect_on_every_metric(monkeypatch):
    _install_fake_graph_metrics(monkeypatch)
    g = _rule_based_graph(["obama", "hawaii"], [("obama", "hawaii", "born in")])
    scores = compute_graph_scores(g, g, backend="RuleBased")
    assert scores["node_precision_recall_f1"]["f1"] == 1.0
    assert scores["edge_precision_recall_f1"]["f1"] == 1.0
    assert scores["smatch"]["f1"] == 1.0


def test_predicted_missing_a_node_reduces_recall_not_precision(monkeypatch):
    _install_fake_graph_metrics(monkeypatch)
    oracle = _rule_based_graph(["obama", "hawaii"], [("obama", "hawaii", "born in")])
    predicted = _rule_based_graph(["obama"], [])
    scores = compute_graph_scores(oracle, predicted, backend="RuleBased")
    assert scores["node_precision_recall_f1"]["precision"] == 1.0
    assert scores["node_precision_recall_f1"]["recall"] == 0.5
    assert 0.0 < scores["smatch"]["f1"] < 1.0


def test_node_duplication_rate_counts_repeated_labels(monkeypatch):
    _install_fake_graph_metrics(monkeypatch)
    graph_with_dupes = _rule_based_graph(["obama", "obama", "hawaii"], [])
    # note: RuleBased backend nodes are a flat string list in the real schema,
    # but networkx dedupes identical string nodes on add_node -- duplication
    # is only observable in ID-based backends. Test duplication with an
    # ID-based (LLMv2-shaped) graph instead:
    llm_graph = {
        "nodes": [{"id": "n0", "label": "obama"}, {"id": "n1", "label": "obama"}, {"id": "n2", "label": "hawaii"}],
        "edges": [],
    }
    rate = compute_node_duplication_rate(llm_graph, backend="LLMv2")
    assert rate == 1 / 3


def test_node_duplication_rate_zero_for_empty_graph(monkeypatch):
    _install_fake_graph_metrics(monkeypatch)
    empty_graph = {"nodes": [], "edges": []}
    assert compute_node_duplication_rate(empty_graph, backend="RuleBased") == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/pipeline/test_graph_scoring.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.eval.graph_scoring'`

- [ ] **Step 3: Write `pipeline/eval/graph_scoring.py`**

```python
from __future__ import annotations

import smatch

from ..sys_path_setup import add_text_corpuses_processing_to_path
from .penman_convert import graph_to_amr_line


def _canonicalize(graph_dict: dict, backend: str):
    add_text_corpuses_processing_to_path()
    from graphMetrics import _to_networkx  # reused private helper, see design spec

    G = _to_networkx(graph_dict, backend)
    if backend == "RuleBased":
        label_of = {n: str(n) for n in G.nodes()}
    else:
        label_of = {n: G.nodes[n].get("label", str(n)) for n in G.nodes()}

    node_labels = set(label_of.values())
    edge_triples = set()
    for u, v, data in G.edges(data=True):
        a, b = sorted((label_of[u], label_of[v]))
        edge_triples.add((a, b, data.get("label", "")))
    return label_of, node_labels, edge_triples


def _prf(gold: set, pred: set) -> dict:
    tp = len(gold & pred)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_node_duplication_rate(graph_dict: dict, backend: str) -> float:
    """Fraction of nodes in this graph that share a label with another node
    in the same graph -- self-contained, not a comparison against another
    graph. Comparing oracle's rate to predicted's rate is what isolates the
    coreference-error contribution (oracle should be near zero by
    construction)."""
    add_text_corpuses_processing_to_path()
    from graphMetrics import _to_networkx

    G = _to_networkx(graph_dict, backend)
    if backend == "RuleBased":
        raw_labels = [str(n) for n in G.nodes()]
    else:
        raw_labels = [G.nodes[n].get("label", str(n)) for n in G.nodes()]

    if not raw_labels:
        return 0.0
    unique = len(set(raw_labels))
    return (len(raw_labels) - unique) / len(raw_labels)


def compute_smatch(oracle_graph: dict, predicted_graph: dict, backend: str) -> dict:
    _, oracle_nodes, oracle_edges = _canonicalize(oracle_graph, backend)
    _, pred_nodes, pred_edges = _canonicalize(predicted_graph, backend)

    oracle_nodes_dict = {f"o{i}": label for i, label in enumerate(sorted(oracle_nodes))}
    pred_nodes_dict = {f"p{i}": label for i, label in enumerate(sorted(pred_nodes))}
    oracle_label_to_id = {label: nid for nid, label in oracle_nodes_dict.items()}
    pred_label_to_id = {label: nid for nid, label in pred_nodes_dict.items()}

    oracle_edge_list = [
        (oracle_label_to_id[a], oracle_label_to_id[b], rel) for a, b, rel in oracle_edges
        if a in oracle_label_to_id and b in oracle_label_to_id
    ]
    pred_edge_list = [
        (pred_label_to_id[a], pred_label_to_id[b], rel) for a, b, rel in pred_edges
        if a in pred_label_to_id and b in pred_label_to_id
    ]

    oracle_line = graph_to_amr_line(oracle_nodes_dict, oracle_edge_list, "o")
    pred_line = graph_to_amr_line(pred_nodes_dict, pred_edge_list, "p")

    smatch.match_triple_dict.clear()  # module-level cache; must clear per independent comparison
    match_num, test_num, gold_num = smatch.get_amr_match(pred_line, oracle_line)
    precision, recall, f1 = smatch.compute_f(match_num, test_num, gold_num)
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_graph_scores(oracle_graph: dict, predicted_graph: dict, backend: str) -> dict:
    _, oracle_nodes, oracle_edges = _canonicalize(oracle_graph, backend)
    _, pred_nodes, pred_edges = _canonicalize(predicted_graph, backend)

    return {
        "node_precision_recall_f1": _prf(oracle_nodes, pred_nodes),
        "edge_precision_recall_f1": _prf(oracle_edges, pred_edges),
        "oracle_node_duplication_rate": compute_node_duplication_rate(oracle_graph, backend),
        "predicted_node_duplication_rate": compute_node_duplication_rate(predicted_graph, backend),
        "smatch": compute_smatch(oracle_graph, predicted_graph, backend),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/pipeline/test_graph_scoring.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add pipeline/eval/graph_scoring.py tests/pipeline/test_graph_scoring.py
git commit -m "feat: add graph scoring (node/edge P-R, duplication rate, smatch)"
```

---

### Task 12: LLMv2GraphAdapter

**Files:**
- Create: `pipeline/graph/llm_v2_graph_adapter.py`
- Test: `tests/pipeline/test_llm_v2_graph_adapter.py`

**Interfaces:**
- Consumes: `add_text_corpuses_processing_to_path` (Task 1).
- Produces: `LLMv2GraphAdapter(llm_client, embedder, language, base_dir=None)`, `name = backend_name = "LLMv2"`, `language_support = "any"`, `build(text) -> dict` (shape: `{"meta", "chunks", "nodes": [{"id","label",...}], "edges": [{"id","source","target","label","weight",...}]}`, matching `RawGraph.model_dump()` from the reused `llm_v2` pipeline — this is also exactly the shape `graphMetrics._to_networkx` expects for non-"RuleBased" backends).

**Testing approach:** inject fake `llm_v2.stages.{preprocessing,chunking,extraction,normalization,deduplication,graph_assembly}` and `llm_v2.config_schema` modules — the real ones transitively import `torch`/`transformers`/`sentence-transformers`.

- [ ] **Step 1: Write the failing test**

```python
# tests/pipeline/test_llm_v2_graph_adapter.py
import sys
import types

from pipeline.graph.llm_v2_graph_adapter import LLMv2GraphAdapter


class _FakeLLMClient:
    def generate(self, prompt):
        return "obama | born in | hawaii"


class _FakeEmbedder:
    pass


def _install_fake_modules(monkeypatch, captured):
    def _mod(name):
        m = types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, m)
        return m

    _mod("llm_v2")
    _mod("llm_v2.stages")

    preprocessing = _mod("llm_v2.stages.preprocessing")
    preprocessing.preprocess = lambda text, language="ru": [types.SimpleNamespace(id=0, text=text)]

    chunking = _mod("llm_v2.stages.chunking")
    chunking.build_chunks = lambda sentences, config: [
        types.SimpleNamespace(id="chunk_0", text=sentences[0].text, sentence_ids=[0])
    ]

    extraction = _mod("llm_v2.stages.extraction")

    def _extract_triplets(chunks, llm, config, base_dir=None):
        captured["extraction_base_dir"] = base_dir
        return [types.SimpleNamespace(subject="obama", relation="born in", object="hawaii", chunk_id="chunk_0")]

    extraction.extract_triplets = _extract_triplets

    normalization = _mod("llm_v2.stages.normalization")
    normalization.normalize_triplets = lambda triplets, config: [
        types.SimpleNamespace(
            subject=t.subject, relation=t.relation, object=t.object, chunk_id=t.chunk_id,
            norm_subject=t.subject, norm_relation=t.relation, norm_object=t.object,
        )
        for t in triplets
    ]

    deduplication = _mod("llm_v2.stages.deduplication")
    deduplication.deduplicate_triplets = lambda triplets, embedder, config: list(triplets)

    graph_assembly = _mod("llm_v2.stages.graph_assembly")

    class _FakeRawGraph:
        def __init__(self, nodes, edges):
            self._nodes = nodes
            self._edges = edges

        def model_dump(self):
            return {"meta": {}, "chunks": [], "nodes": self._nodes, "edges": self._edges}

    def _assemble_graph(triplets, chunks, source_text, config):
        nodes = []
        edges = []
        seen = {}
        for t in triplets:
            for label in (t.norm_subject, t.norm_object):
                if label not in seen:
                    seen[label] = f"n{len(nodes)}"
                    nodes.append({"id": seen[label], "label": label})
            edges.append({
                "id": f"e{len(edges)}", "source": seen[t.norm_subject], "target": seen[t.norm_object],
                "label": t.norm_relation, "weight": 1,
            })
        return _FakeRawGraph(nodes, edges)

    graph_assembly.assemble_graph = _assemble_graph

    config_schema = _mod("llm_v2.config_schema")
    for cls_name in ("ExtractionConfig", "NormalizationConfig", "DeduplicationConfig", "PipelineConfig"):
        def _make_init(name):
            def _init(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
                if name == "ExtractionConfig" and "prompt_file" not in kwargs:
                    self.prompt_file = ""
                if name == "NormalizationConfig" and "language" not in kwargs:
                    self.language = "en"
            return _init
        setattr(config_schema, cls_name, type(cls_name, (), {"__init__": _make_init(cls_name)}))


def test_build_returns_raw_graph_dict_shape(monkeypatch):
    captured: dict = {}
    _install_fake_modules(monkeypatch, captured)

    adapter = LLMv2GraphAdapter(llm_client=_FakeLLMClient(), embedder=_FakeEmbedder(), language="en")
    graph = adapter.build("Obama was born in Hawaii.")

    assert graph["nodes"] == [{"id": "n0", "label": "obama"}, {"id": "n1", "label": "hawaii"}]
    assert graph["edges"] == [{"id": "e0", "source": "n0", "target": "n1", "label": "born in", "weight": 1}]


def test_adapter_metadata():
    adapter = LLMv2GraphAdapter(llm_client=_FakeLLMClient(), embedder=_FakeEmbedder(), language="en")
    assert adapter.name == "LLMv2"
    assert adapter.backend_name == "LLMv2"
    assert adapter.language_support == "any"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/pipeline/test_llm_v2_graph_adapter.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.graph.llm_v2_graph_adapter'`

- [ ] **Step 3: Write `pipeline/graph/llm_v2_graph_adapter.py`**

```python
from __future__ import annotations

from pathlib import Path

from ..sys_path_setup import add_text_corpuses_processing_to_path


class LLMv2GraphAdapter:
    name = "LLMv2"
    backend_name = "LLMv2"
    language_support = "any"

    def __init__(self, llm_client, embedder, language: str, base_dir: Path | None = None):
        self.llm_client = llm_client
        self.embedder = embedder
        self.language = language
        dags_dir = add_text_corpuses_processing_to_path()
        self.base_dir = base_dir or (dags_dir / "llm_v2")

    def build(self, text: str) -> dict:
        add_text_corpuses_processing_to_path()
        from llm_v2.config_schema import (
            DeduplicationConfig,
            ExtractionConfig,
            NormalizationConfig,
            PipelineConfig,
        )
        from llm_v2.stages.chunking import build_chunks
        from llm_v2.stages.deduplication import deduplicate_triplets
        from llm_v2.stages.extraction import extract_triplets
        from llm_v2.stages.graph_assembly import assemble_graph
        from llm_v2.stages.normalization import normalize_triplets
        from llm_v2.stages.preprocessing import preprocess

        extraction_config = ExtractionConfig(prompt_file=f"prompts/extraction_{self.language}.txt")
        normalization_config = NormalizationConfig(language=self.language)
        deduplication_config = DeduplicationConfig()
        pipeline_config = PipelineConfig(
            extraction=extraction_config,
            normalization=normalization_config,
            deduplication=deduplication_config,
        )

        sentences = preprocess(text, language=self.language)
        chunks = build_chunks(sentences, extraction_config)
        raw_triplets = extract_triplets(chunks, self.llm_client, extraction_config, base_dir=self.base_dir)
        norm_triplets = normalize_triplets(raw_triplets, normalization_config)
        dedup_triplets = deduplicate_triplets(norm_triplets, self.embedder, deduplication_config)
        raw_graph = assemble_graph(dedup_triplets, chunks, text, pipeline_config)
        return raw_graph.model_dump()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/pipeline/test_llm_v2_graph_adapter.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add pipeline/graph/llm_v2_graph_adapter.py tests/pipeline/test_llm_v2_graph_adapter.py
git commit -m "feat: add LLMv2 graph backend adapter"
```

---

### Task 13: `coref_scoring.py` (CoNLL F1 via coval)

**Files:**
- Create: `pipeline/eval/coref_scoring.py`
- Test: `tests/pipeline/test_coref_scoring.py`

**Interfaces:**
- Consumes: `MentionSpan` from `pipeline.types`. Depends directly on the `coval` package (installed in Task 1's `requirements.txt` from `git+https://github.com/ns-moosavi/coval.git` — **not** the unrelated PyPI package of the same name).
- Produces: `score_coreference(gold_clusters: list[list[MentionSpan]], sys_clusters: list[list[MentionSpan]]) -> dict` with keys `muc`, `b_cubed`, `ceafe` (each `{"precision", "recall", "f1"}`) and `conll_f1` (float) — consumed by `run_experiment.py` (Task 15).

- [ ] **Step 1: Write the failing tests**

```python
# tests/pipeline/test_coref_scoring.py
from pipeline.eval.coref_scoring import score_coreference
from pipeline.types import MentionSpan


def _cluster(*pairs):
    return [MentionSpan(a, b) for a, b in pairs]


def test_perfect_match_scores_one_on_every_metric():
    gold = [_cluster((0, 0), (5, 5), (10, 10))]
    sysc = [_cluster((0, 0), (5, 5), (10, 10))]
    result = score_coreference(gold, sysc)
    assert result["muc"]["f1"] == 1.0
    assert result["b_cubed"]["f1"] == 1.0
    assert result["ceafe"]["f1"] == 1.0
    assert result["conll_f1"] == 1.0


def test_partial_match_between_zero_and_one():
    gold = [_cluster((0, 0), (5, 5), (10, 10))]
    sysc = [_cluster((0, 0), (5, 5))]  # missed the third mention
    result = score_coreference(gold, sysc)
    assert 0.0 < result["conll_f1"] < 1.0


def test_empty_system_output_scores_zero():
    gold = [_cluster((0, 0), (5, 5))]
    result = score_coreference(gold, [])
    assert result["conll_f1"] == 0.0


def test_all_returned_values_are_plain_python_floats():
    gold = [_cluster((0, 0), (5, 5))]
    sysc = [_cluster((0, 0), (5, 5))]
    result = score_coreference(gold, sysc)
    for metric_name in ("muc", "b_cubed", "ceafe"):
        for key in ("precision", "recall", "f1"):
            assert isinstance(result[metric_name][key], float)
    assert isinstance(result["conll_f1"], float)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/pipeline/test_coref_scoring.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.eval.coref_scoring'`

- [ ] **Step 3: Write `pipeline/eval/coref_scoring.py`**

```python
from __future__ import annotations

from coval.eval.evaluator import b_cubed, ceafe, evaluate_documents, muc

from ..types import MentionSpan


def _mention_key(span: MentionSpan) -> tuple[int, int]:
    return (span.start_token, span.end_token)


def _clusters_as_tuples(clusters: list[list[MentionSpan]]) -> list[tuple]:
    return [tuple(sorted({_mention_key(m) for m in cluster})) for cluster in clusters]


def _mention_to_other_cluster(own_clusters: list[tuple], other_clusters: list[tuple]) -> dict:
    other_by_mention: dict = {}
    for oc in other_clusters:
        for m in oc:
            other_by_mention[m] = oc
    mapping: dict = {}
    for cluster in own_clusters:
        for m in cluster:
            if m in other_by_mention:
                mapping[m] = other_by_mention[m]
    return mapping


def score_coreference(
    gold_clusters: list[list[MentionSpan]], sys_clusters: list[list[MentionSpan]]
) -> dict:
    key_clusters = _clusters_as_tuples(gold_clusters)
    sys_clusters_t = _clusters_as_tuples(sys_clusters)

    key_mention_sys_cluster = _mention_to_other_cluster(key_clusters, sys_clusters_t)
    sys_mention_key_cluster = _mention_to_other_cluster(sys_clusters_t, key_clusters)

    coref_info = (key_clusters, sys_clusters_t, key_mention_sys_cluster, sys_mention_key_cluster)
    doc_coref_infos = {"doc1": coref_info}

    results: dict = {}
    for name, metric in (("muc", muc), ("b_cubed", b_cubed), ("ceafe", ceafe)):
        recall, precision, f1_score = evaluate_documents(doc_coref_infos, metric)
        results[name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1_score),
        }

    results["conll_f1"] = (results["muc"]["f1"] + results["b_cubed"]["f1"] + results["ceafe"]["f1"]) / 3
    return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/pipeline/test_coref_scoring.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add pipeline/eval/coref_scoring.py tests/pipeline/test_coref_scoring.py
git commit -m "feat: add CoNLL F1 coreference scoring via coval"
```

---

### Task 14: `report.py`

**Files:**
- Create: `pipeline/report.py`
- Test: `tests/pipeline/test_report.py`

**Interfaces:**
- Produces: `build_results(run_id: str, language: str, per_pairing: list[dict]) -> dict`, `save_results_json(results: dict, output_dir: Path) -> Path`, `render_html_report(results: dict) -> str`, `save_html_report(html: str, output_dir: Path) -> Path` — consumed by `run_experiment.py` (Task 15).

- [ ] **Step 1: Write the failing tests**

```python
# tests/pipeline/test_report.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/pipeline/test_report.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.report'`

- [ ] **Step 3: Write `pipeline/report.py`**

```python
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def build_results(run_id: str, language: str, per_pairing: list[dict]) -> dict:
    return {
        "run_id": run_id,
        "language": language,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results": per_pairing,
    }


def save_results_json(results: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "results.json"
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _fmt(value) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_html_report(results: dict) -> str:
    rows = []
    for entry in results["results"]:
        coref = entry.get("coreference_metrics")
        graph = entry.get("graph_metrics") or {}
        conll_f1 = coref["conll_f1"] if coref else None
        node_f1 = graph.get("node_precision_recall_f1", {}).get("f1")
        edge_f1 = graph.get("edge_precision_recall_f1", {}).get("f1")
        smatch_f1 = graph.get("smatch", {}).get("f1")
        rows.append(
            f"<tr><td>{entry['resolver']}</td><td>{entry['graph_backend']}</td>"
            f"<td>{_fmt(conll_f1)}</td><td>{_fmt(node_f1)}</td>"
            f"<td>{_fmt(edge_f1)}</td><td>{_fmt(smatch_f1)}</td></tr>"
        )
    table_rows = "\n".join(rows)
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Experiment run {results['run_id']}</title></head>
<body>
<h1>Cascade experimental stand -- run {results['run_id']}</h1>
<p>Language: {results['language']} | Generated: {results['generated_at']}</p>
<table border="1" cellpadding="4" cellspacing="0">
<thead><tr><th>Resolver</th><th>Graph backend</th><th>CoNLL F1</th>
<th>Node F1</th><th>Edge F1</th><th>Smatch F1</th></tr></thead>
<tbody>
{table_rows}
</tbody>
</table>
</body></html>"""


def save_html_report(html: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "report.html"
    path.write_text(html, encoding="utf-8")
    return path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/pipeline/test_report.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add pipeline/report.py tests/pipeline/test_report.py
git commit -m "feat: add results.json and HTML report generation"
```

---

### Task 15: `run_experiment.py` CLI + smoke test

**Files:**
- Create: `pipeline/run_experiment.py`
- Create: `data/corpora/sample_en_mini.conllu`
- Test: `tests/pipeline/test_run_experiment_smoke.py`

**Interfaces:**
- Consumes every module from Tasks 2-14.
- Produces: `run_experiment(corpus_path, language, resolver_names, graph_backend_names, output_dir, llm_client=None, embedder=None) -> Path` (the orchestration function, independently testable with fakes) and `main(argv=None) -> int` (the CLI entry point).

**Smoke test scope:** exercises `LapinLiassAdapter` + `RuleBasedGraphAdapter` only, with `text-corpuses-processing`'s real `anaphoraResolverLapinLiass`/`graphBuilder` modules mocked via `sys.modules` injection exactly as in Tasks 6 and 9 — this keeps the smoke test runnable without spaCy models installed, while still exercising the full `run_experiment()` orchestration path (loader → oracle → resolver → graph backend → scoring → report) end-to-end for real.

- [ ] **Step 1: Create the tiny CoNLL-U fixture corpus**

`data/corpora/sample_en_mini.conllu`:
```
# newdoc id = sample1
# sent_id = sample1-1
1	John	John	PROPN	_	_	2	nsubj	_	Entity=(e1-person-1-new-1-sgl-1)
2	arrived	arrive	VERB	_	_	0	root	_	SpaceAfter=No
3	.	.	PUNCT	_	_	2	punct	_	_

# sent_id = sample1-2
1	He	he	PRON	_	_	2	nsubj	_	Entity=(e1)
2	left	leave	VERB	_	_	0	root	_	SpaceAfter=No
3	.	.	PUNCT	_	_	2	punct	_	_
```

- [ ] **Step 2: Write the failing smoke test**

```python
# tests/pipeline/test_run_experiment_smoke.py
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pipeline.run_experiment import run_experiment

FIXTURE_CORPUS = Path(__file__).resolve().parents[2] / "data" / "corpora" / "sample_en_mini.conllu"


@dataclass
class _FakeSpacyToken:
    idx: int
    text: str


class _FakeSpacyDoc:
    def __init__(self, tokens):
        self._tokens = tokens

    def __getitem__(self, i):
        return self._tokens[i]


@dataclass
class _FakeResolution:
    pronoun: str
    pronoun_index: int
    antecedent: Optional[str]
    antecedent_index: Optional[int]
    score: float


class _FakeBatchAnaphoraResolver:
    def resolve_document(self, text):
        tokens = [
            _FakeSpacyToken(0, "John"), _FakeSpacyToken(5, "arrived"), _FakeSpacyToken(12, "."),
            _FakeSpacyToken(14, "He"), _FakeSpacyToken(17, "left"), _FakeSpacyToken(21, "."),
        ]
        doc = _FakeSpacyDoc(tokens)
        resolutions = [
            _FakeResolution("He", 3, "John", 0, 100.0)
        ]
        return {"text": text, "doc": doc, "resolutions": resolutions}


def _fake_build_substitutions(doc, resolutions, mark=False):
    subs = []
    for r in resolutions:
        if r.antecedent is None:
            continue
        tok = doc[r.pronoun_index]
        subs.append(types.SimpleNamespace(start=tok.idx, end=tok.idx + len(tok.text), replacement=r.antecedent))
    subs.sort(key=lambda s: s.start, reverse=True)
    return subs


def _fake_apply_substitutions(text, substitutions):
    out = text
    for s in substitutions:
        out = out[: s.start] + s.replacement + out[s.end :]
    return out


def _install_fake_resolver_module(monkeypatch):
    fake = types.ModuleType("anaphoraResolverLapinLiass")
    fake.BatchAnaphoraResolver = _FakeBatchAnaphoraResolver
    fake.build_substitutions = _fake_build_substitutions
    fake.apply_substitutions = _fake_apply_substitutions
    monkeypatch.setitem(sys.modules, "anaphoraResolverLapinLiass", fake)


def _install_fake_graph_builder_module(monkeypatch):
    fake = types.ModuleType("graphBuilder")

    def _extract_graph_edges(text):
        edges = [("john", "arrive", "")]
        if "left" in text.lower():
            edges.append(("john", "leave", ""))
        return edges

    def _merge_graph(graph, new_edges):
        nodes = set(graph["nodes"])
        for a1, a2, meaning in new_edges:
            nodes.add(a1)
            nodes.add(a2)
            graph["edges"].append({"agent_1": a1, "agent_2": a2, "meaning": meaning, "weight": 1})
        graph["nodes"] = list(nodes)
        return graph

    fake.extract_graph_edges = _extract_graph_edges
    fake.merge_graph = _merge_graph
    monkeypatch.setitem(sys.modules, "graphBuilder", fake)


def _install_fake_graph_metrics_module(monkeypatch):
    import networkx as nx

    fake = types.ModuleType("graphMetrics")

    def _to_networkx(graph_dict, backend):
        G = nx.Graph()
        for node in graph_dict.get("nodes", []):
            G.add_node(node)
        for edge in graph_dict.get("edges", []):
            G.add_edge(edge["agent_1"], edge["agent_2"], weight=edge.get("weight", 1), label=edge.get("meaning", ""))
        return G

    fake._to_networkx = _to_networkx
    monkeypatch.setitem(sys.modules, "graphMetrics", fake)


def test_run_experiment_end_to_end_produces_results_json(monkeypatch, tmp_path):
    _install_fake_resolver_module(monkeypatch)
    _install_fake_graph_builder_module(monkeypatch)
    _install_fake_graph_metrics_module(monkeypatch)

    output_dir = tmp_path / "run1"
    results_path = run_experiment(
        corpus_path=FIXTURE_CORPUS,
        language="en",
        resolver_names=["LapinLiass"],
        graph_backend_names=["RuleBased"],
        output_dir=output_dir,
    )

    assert results_path.exists()
    assert (output_dir / "report.html").exists()

    import json
    data = json.loads(results_path.read_text(encoding="utf-8"))
    assert data["language"] == "en"
    assert len(data["results"]) == 1
    pairing = data["results"][0]
    assert pairing["resolver"] == "LapinLiass"
    assert pairing["graph_backend"] == "RuleBased"
    assert pairing["coreference_metrics"]["conll_f1"] == 1.0  # LapinLiass resolves He->John, matching gold e1
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/pipeline/test_run_experiment_smoke.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.run_experiment'`

- [ ] **Step 4: Write `pipeline/run_experiment.py`**

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .corpus.corefud_loader import load_corefud_corpus
from .corpus.oracle import build_oracle_text
from .eval.coref_scoring import score_coreference
from .eval.graph_scoring import compute_graph_scores
from .graph.llm_v2_graph_adapter import LLMv2GraphAdapter
from .graph.rule_based_graph_adapter import RuleBasedGraphAdapter
from .report import build_results, render_html_report, save_html_report, save_results_json
from .resolvers.base import supports_language
from .resolvers.lapin_liass_adapter import LapinLiassAdapter
from .resolvers.llm_v2_adapter import LLMv2Adapter
from .resolvers.spacy_neural_adapter import SpacyNeuralAdapter


def _build_resolver(name: str, language: str, llm_client=None):
    if name == "LapinLiass":
        return LapinLiassAdapter()
    if name == "SpacyNeural":
        return SpacyNeuralAdapter()
    if name == "LLMv2":
        if llm_client is None:
            raise ValueError("LLMv2 resolver requires an llm_client (see --llm-model)")
        return LLMv2Adapter(llm_client=llm_client, language=language)
    raise ValueError(f"unknown resolver: {name}")


def _build_graph_backend(name: str, language: str, llm_client=None, embedder=None):
    if name == "RuleBased":
        return RuleBasedGraphAdapter()
    if name == "LLMv2":
        if llm_client is None or embedder is None:
            raise ValueError("LLMv2 graph backend requires an llm_client and embedder (see --llm-model)")
        return LLMv2GraphAdapter(llm_client=llm_client, embedder=embedder, language=language)
    raise ValueError(f"unknown graph backend: {name}")


def _average_coref(doc_entries: list[dict]) -> dict | None:
    values = [e["coreference_metrics"] for e in doc_entries if e["coreference_metrics"] is not None]
    if not values:
        return None
    return {"conll_f1": sum(v["conll_f1"] for v in values) / len(values)}


def _average_graph(doc_entries: list[dict]) -> dict:
    n = len(doc_entries)
    node_f1s = [e["graph_metrics"]["node_precision_recall_f1"]["f1"] for e in doc_entries]
    edge_f1s = [e["graph_metrics"]["edge_precision_recall_f1"]["f1"] for e in doc_entries]
    smatch_f1s = [e["graph_metrics"]["smatch"]["f1"] for e in doc_entries]
    return {
        "node_precision_recall_f1": {"f1": sum(node_f1s) / n},
        "edge_precision_recall_f1": {"f1": sum(edge_f1s) / n},
        "smatch": {"f1": sum(smatch_f1s) / n},
    }


def run_experiment(
    corpus_path: Path,
    language: str,
    resolver_names: list[str],
    graph_backend_names: list[str],
    output_dir: Path,
    llm_client=None,
    embedder=None,
) -> Path:
    documents = load_corefud_corpus(corpus_path)

    per_pairing = []
    for resolver_name in resolver_names:
        resolver = _build_resolver(resolver_name, language, llm_client=llm_client)
        if not supports_language(resolver, language):
            continue

        for backend_name in graph_backend_names:
            graph_backend = _build_graph_backend(backend_name, language, llm_client=llm_client, embedder=embedder)
            if not supports_language(graph_backend, language):
                continue

            doc_entries = []
            for doc in documents:
                oracle_text = build_oracle_text(doc)
                resolver_output = resolver.resolve(doc)

                oracle_graph = graph_backend.build(oracle_text)
                predicted_graph = graph_backend.build(resolver_output.resolved_text)

                coref_metrics = None
                if resolver_output.clusters is not None:
                    coref_metrics = score_coreference(doc.clusters, resolver_output.clusters)

                graph_metrics = compute_graph_scores(oracle_graph, predicted_graph, graph_backend.backend_name)

                doc_entries.append({
                    "doc_id": doc.doc_id,
                    "coreference_metrics": coref_metrics,
                    "graph_metrics": graph_metrics,
                })

            per_pairing.append({
                "resolver": resolver_name,
                "graph_backend": backend_name,
                "documents": doc_entries,
                "coreference_metrics": _average_coref(doc_entries),
                "graph_metrics": _average_graph(doc_entries),
            })

    output_dir = Path(output_dir)
    run_id = output_dir.name
    results = build_results(run_id, language, per_pairing)
    results_path = save_results_json(results, output_dir)
    html = render_html_report(results)
    save_html_report(html, output_dir)
    return results_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cascade-architecture experimental stand")
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--language", required=True)
    parser.add_argument("--resolvers", required=True, help="comma-separated resolver names")
    parser.add_argument("--graph-backends", required=True, help="comma-separated graph backend names")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--llm-model", default=None, help="HF model name; required for the LLMv2 resolver/backend")
    args = parser.parse_args(argv)

    resolver_names = args.resolvers.split(",")
    backend_names = args.graph_backends.split(",")

    llm_client = None
    embedder = None
    if "LLMv2" in resolver_names or "LLMv2" in backend_names:
        from .sys_path_setup import add_text_corpuses_processing_to_path

        add_text_corpuses_processing_to_path()
        from llm_v2.config_schema import EmbeddingConfig, LLMConfig
        from llm_v2.models.embedder import Embedder
        from llm_v2.models.llm_client import LLMClient

        llm_client = LLMClient(LLMConfig(model_name=args.llm_model or "Qwen/Qwen2-1.5B-Instruct"))
        embedder = Embedder(EmbeddingConfig())

    results_path = run_experiment(
        corpus_path=args.corpus,
        language=args.language,
        resolver_names=resolver_names,
        graph_backend_names=backend_names,
        output_dir=args.output,
        llm_client=llm_client,
        embedder=embedder,
    )
    print(f"Results written to {results_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/pipeline/test_run_experiment_smoke.py -v`
Expected: PASS (1 test)

- [ ] **Step 6: Run the full test suite**

Run: `pytest -v`
Expected: PASS (all tests from Tasks 1-15)

- [ ] **Step 7: Commit**

```bash
git add pipeline/run_experiment.py data/corpora/sample_en_mini.conllu tests/pipeline/test_run_experiment_smoke.py
git commit -m "feat: add run_experiment CLI orchestrating the cascade pipeline end-to-end"
```

---

## Self-Review Notes

- **Spec coverage:** CorefUD loading (Task 3), oracle-ablation text (Task 4), resolver adapters for LapinLiass/SpacyNeural/LLMv2 (Tasks 6-8), graph backend adapters for RuleBased/LLMv2 (Tasks 9, 12), coreference F1 (Task 13), graph P/R + duplication + Smatch (Task 11), report output (Task 14), CLI (Task 15) — every component in the design spec has a task.
- **Type consistency verified:** `ResolverOutput.clusters: list[list[MentionSpan]] | None` (Task 2) is produced identically by Tasks 6/7/8 and consumed identically in Task 15's `if resolver_output.clusters is not None` branch. `GraphBackendAdapter.backend_name` (Task 9) is set by every graph adapter and consumed by `compute_graph_scores(..., backend)` (Task 11) and `run_experiment`'s call site (Task 15) — same string values (`"RuleBased"`, `"LLMv2"`) used throughout.
- **No placeholders:** every step contains real, hand-verified code (the CoNLL-U parser, the coval integration, and the smatch/AMR conversion were each interactively tested against the real installed libraries during design, not written from memory).
