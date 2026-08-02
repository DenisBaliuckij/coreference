# Cascade-Architecture Experimental Stand — Design

## Context

This repository holds the working materials for a PhD thesis: *"Каскадные и совместные
архитектуры интеграции разрешения кореференции в построение семантического графа:
сравнительное мультиязычное исследование"* (see `Thesis/dissertation_plan.docx` and
`Thesis/dissertation_intro_skeleton.docx`). The thesis plan defines 8 research tasks and
3 hypotheses (H1: joint > cascade on error propagation; H2: typologically modulated;
H3: resource threshold), comparing **cascade** (coreference resolution → graph
construction) against a **joint** architecture, across typologically diverse languages,
with statistical significance testing and a downstream QA validation.

That full program is too large for one pipeline. This spec covers only **thesis tasks
2–3**: build a unified, language-agnostic experimental stand and a reproducible
cascade-architecture baseline, by reusing the coreference-resolution and
graph-construction code already implemented in the sibling repository
`text-corpuses-processing`. The **joint architecture (task 4)**, multi-language batch
comparison, statistical testing (H1–H3), and downstream QA validation are explicitly
out of scope and will be separate follow-on specs.

## What already exists in `text-corpuses-processing`

- `dags/anaphoraResolverLapinLiass.py` — rule-based English pronoun resolver
  (spaCy `en_core_web_sm`, salience scoring). Returns resolved text +
  `List[Resolution]` (pronoun index → antecedent index pairs).
- `dags/anaphoraResolverSpacyNeural.py` — English neural coreference
  (`en_coreference_web_trf`). Returns resolved text + `List[Resolution]`, internally
  has real coreference clusters (`doc.spans["coref_clusters_*"]`) before flattening.
- `dags/anaphoraResolver.py` — dispatcher between the two, by `resolver_name`.
- `dags/graphBuilder.py` — rule-based English graph extraction (spaCy `en_core_web_lg`
  dependency parsing + WordNet lemmatization → subject/relation/object triples).
- `dags/graphMetrics.py` — structural graph metrics (density, degree stats, hubs,
  components) for all three existing graph backends.
- `dags/llm_v2/` — the only **language-configurable** component. A local-LLM pipeline
  (Qwen2-1.5B-Instruct, runs on CPU, no API key) with distinct stages:
  `preprocessing → coreference (stages/coreference.py) → chunking → extraction
  (stages/extraction.py) → normalization → deduplication → graph_assembly →
  clustering`. Coreference and extraction are driven by prompt template files
  (`prompts/coreference_{lang}.txt`, `prompts/extraction_{lang}.txt`); `en` and `ru`
  exist today. Its coreference stage **rewrites text** (replaces pronouns/demonstratives
  with full entity names) but does not emit mention spans or clusters.

None of the above already computes coreference F1 against gold, or compares two graphs
against each other. That evaluation layer is what this spec adds.

## Scope

Build a standalone CLI pipeline in this repo (`coreference/pipeline/`) that:

1. Loads a gold-annotated corpus in **CorefUD CoNLL-U format** (language-agnostic;
   directly usable for RuCoCo, CorefUD English, and any other CorefUD-format treebank;
   supports zero-anaphora mentions).
2. Derives an **oracle-resolved text**: substitute every gold mention with its cluster's
   head-mention surface form, using only gold annotations (no model).
3. Runs one or more **coreference resolver adapters** to produce a **predicted-resolved
   text** for the same document.
4. Runs a **graph backend adapter** on both the oracle-resolved and predicted-resolved
   text, producing an oracle graph and a predicted graph.
5. Scores:
   - **Coreference quality**: predicted clusters vs. gold clusters, CoNLL F1
     (MUC/B³/CEAFe) via a CorefUD-aware scorer — only for resolvers that expose
     mention-level clusters.
   - **Graph quality (oracle-ablation)**: oracle graph vs. predicted graph — node/edge
     precision-recall, node-duplication rate, and a Smatch-style structural score. The
     gap between oracle and predicted is the "error budget" attributable to coreference
     mistakes, per the thesis's experiment design (Part III).
6. Emits `results.json` (per resolver × per graph backend, metrics or explicit `N/A`)
   and a static HTML summary report, under `reports/<run-id>/`.

Reuse mechanism: `text-corpuses-processing/dags` is added to `sys.path` at process
start (mirroring how Airflow already treats that folder as the DAG import root); its
modules are imported directly, not copied or forked.

## Components

### `pipeline/corpus/corefud_loader.py`

Parses a CoNLL-U file (or directory of files) with `Entity=` coreference annotations
into a `CorefDocument`: raw text (reconstructed respecting `SpaceAfter=No`), token list
with character offsets, sentence boundaries, and gold clusters (list of mention token
spans, including empty/zero mentions per CorefUD convention).

### `pipeline/corpus/oracle.py`

`build_oracle_text(doc: CorefDocument) -> str` — for each gold cluster, replace every
non-head mention's span with the head mention's surface text (same substitution
strategy the existing `LapinLiass`/`SpacyNeural` adapters use), applied in
reverse-offset order to keep spans valid.

### `pipeline/resolvers/` — `ResolverAdapter` interface

```python
class ResolverOutput:
    resolved_text: str
    clusters: list[list[MentionSpan]] | None  # None = not scorable for coref F1

class ResolverAdapter(Protocol):
    language_support: set[str] | Literal["any"]
    def resolve(self, doc: CorefDocument) -> ResolverOutput: ...
```

- `LapinLiassAdapter`, `SpacyNeuralAdapter` (`language_support = {"en"}`): call the
  existing resolver on `doc` raw text, then project the resolver's spaCy-token spans
  back onto the gold CoNLL-U tokens by character-offset overlap, so `clusters` are
  expressed in gold token coordinates (required for the CorefUD scorer to compare
  like-for-like). `SpacyNeuralAdapter` uses the resolver's native coreference clusters
  directly; `LapinLiassAdapter` chains its pairwise `Resolution` antecedent links via
  union-find into clusters.
- `LLMv2Adapter` (`language_support = "any"`, in practice `{"en", "ru"}` until more
  prompt files are added): calls `llm_v2.stages.coreference.resolve_coreferences`.
  Returns `clusters=None` — its output is only usable for the graph-level
  oracle-ablation, not for coreference F1. This asymmetry is surfaced in the report,
  not hidden.

Adding a language to `LLMv2Adapter` = add `prompts/coreference_<lang>.txt` and
`prompts/extraction_<lang>.txt`; no code change.

### `pipeline/graph/` — `GraphBackendAdapter` interface

```python
class GraphBackendAdapter(Protocol):
    language_support: set[str] | Literal["any"]
    def build(self, text: str) -> dict: ...  # graph_dict, backend-native shape
```

- `LLMv2GraphAdapter` (`language_support = "any"`, `{"en", "ru"}` today): wraps
  `llm_v2.pipeline` stages `chunking → extraction → normalization → deduplication →
  graph_assembly → clustering`, given already-resolved text (coreference stage
  skipped/bypassed since resolution already happened).
- `RuleBasedGraphAdapter` (`language_support = {"en"}`, optional/bonus): wraps
  `graphBuilder.extract_graph_edges` + `merge_graph`, for English-only runs that want
  to compare against the existing rule-based backend too.

### `pipeline/eval/coref_scoring.py`

Wraps a CorefUD-aware coreference scorer (handles zero-mention alignment, unlike the
plain CoNLL-2012 `coval` scorer) to compute MUC/B³/CEAFe/CoNLL-F1 between a resolver's
projected gold-token-coordinate clusters and the gold clusters. Returns `None`
(reported as `N/A`) when the resolver's `clusters` is `None`.

### `pipeline/eval/graph_scoring.py`

- Node/edge precision-recall and node-duplication rate: set comparison over
  normalized node labels / edge triples between oracle graph and predicted graph.
- Smatch-style score: since our graphs are plain node/edge JSON (not AMR), a small
  `graph/penman_convert.py` maps each graph backend's native shape into a triple-set
  representation the Smatch matching algorithm (best-alignment search over node
  correspondences) can score; exact conversion approach (via the `smatch` package vs.
  a minimal from-scratch implementation of the matching algorithm) is an implementation
  detail decided during coding, not a design blocker.

### `pipeline/report.py`, `pipeline/run_experiment.py`

`run_experiment.py` is the CLI entry point:

```
python -m pipeline.run_experiment \
  --corpus data/corpora/en_sample.conllu --language en \
  --resolvers LapinLiass,SpacyNeural,LLMv2 \
  --graph-backends LLMv2,RuleBased \
  --output reports/run1
```

For each requested resolver × graph-backend combination (filtered by
`language_support` for the given `--language`), runs the full flow (oracle text →
oracle graph; predicted text → predicted graph; score both axes where computable) and
writes `results.json` + an HTML table under `reports/run1/`.

## Directory layout

```
coreference/
  pipeline/
    corpus/
      corefud_loader.py
      oracle.py
    resolvers/
      base.py
      lapin_liass_adapter.py
      spacy_neural_adapter.py
      llm_v2_adapter.py
    graph/
      base.py
      llm_v2_graph_adapter.py
      rule_based_graph_adapter.py
      penman_convert.py
    eval/
      coref_scoring.py
      graph_scoring.py
    report.py
    run_experiment.py
    config.py
  data/
    corpora/            # small CoNLL-U samples for smoke-testing
  reports/               # run outputs (gitignored)
  requirements.txt
  README.md
```

## Testing

- Unit tests per adapter using small synthetic `CorefDocument` fixtures (a few
  sentences, 1–2 clusters), independent of any real spaCy/LLM model where possible
  (mock the underlying resolver/LLM call, test only the adapter's span-projection and
  cluster-building logic).
- `corefud_loader.py`: round-trip test — parse a hand-written small CoNLL-U fixture
  with known clusters (including a zero mention) and assert the resulting
  `CorefDocument` matches expectations.
- `oracle.py`: given known gold clusters, assert the exact substituted text.
- `graph_scoring.py`: unit tests on small hand-built graph pairs with known
  overlap, checking P/R and duplication-rate arithmetic directly (no model calls).
- One smoke-integration test: run the full CLI on a tiny (2–3 sentence) CoNLL-U
  fixture with `LapinLiassAdapter` + `RuleBasedGraphAdapter` (the only pairing with no
  external model download required beyond what `text-corpuses-processing` already
  needs), asserting `results.json` is produced with the expected keys.

## Out of scope (future specs)

- Joint architecture (thesis task 4).
- Multi-language batch runs / cross-language comparison tables.
- Statistical significance testing (bootstrap, permutation tests, Holm correction) for
  H1–H3.
- Learning-curve experiments (H3, data-fraction training).
- Error analysis by anaphora type (pronominal / nominal / zero).
- Downstream QA validation task.
- New coreference resolvers or graph-extraction methods for languages beyond what
  `text-corpuses-processing` already supports (en, ru) — only the plumbing to add them
  easily is in scope.
