#!/bin/sh
# Run the coreference stand in a throw-away container. Results go to ./reports/<name>.
# Corpus input: ./corpora (read-only); the DocumentCatalog text layer is mounted read-only
# at /catalog_text for corpus exports.
#   ./run.sh tests
#   ./run.sh smoke
#   ./run.sh python -m pipeline.run_experiment --corpus /corpora/x.conllu --language en ...
# Containers use the host network so that --llm-endpoint http://127.0.0.1:8081/v1 reaches the
# host's llama-server (it listens on 127.0.0.1 only).
cd "$(dirname "$0")"
mkdir -p reports corpora
LIMITS="--memory 12g --memory-swap 12g"  # an OOM stays inside the container, not on the host
MOUNTS="-v $PWD/reports:/reports -v $PWD/corpora:/corpora:ro -v /mnt/text-corpus/catalog_work/text:/catalog_text:ro"
case "$1" in
  tests) exec docker run --rm --name coref-experiment-tests coref-experiment:latest ;;
  smoke) exec docker run --rm --user 1000:1000 $LIMITS --name coref-experiment-smoke $MOUNTS coref-experiment:latest \
           python -m pipeline.run_experiment --corpus data/corpora/sample_en_mini.conllu --language en \
           --resolvers LapinLiass --graph-backends RuleBased --output /reports/smoke_en ;;
  *) exec docker run --rm --user 1000:1000 --network host $LIMITS --name "coref-experiment-$$" $MOUNTS coref-experiment:latest "$@" ;;
esac
