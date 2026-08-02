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
