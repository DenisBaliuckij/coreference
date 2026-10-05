# Coreference experiment stand (cascade architecture, thesis tasks 2-3) in one image.
# Code: coreference @ cascade-experimental-stand, text-corpuses-processing/dags @ main,
# copied from `git archive` exports (commits recorded in /stand/BUILD_INFO). Build context: a
# directory holding coreference/ (this repo) and text-corpuses-processing/ (at least dags/).
# Layout matches the stand's default sibling path: /stand/.. /text-corpuses-processing/dags
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# the stand's own requirements + what the reused English resolver/graph modules need:
# spaCy models, NLTK WordNet
COPY coreference/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt "spacy>=3.8,<3.9" nltk \
    && python -m spacy download en_core_web_sm \
    && python -m spacy download en_core_web_lg \
    && python -m nltk.downloader -d /usr/local/share/nltk_data wordnet omw-1.4
# nltk 3.10 cannot read WordNet from the downloaded zips: unpack them
RUN cd /usr/local/share/nltk_data/corpora \
    && python -m zipfile -e wordnet.zip . && python -m zipfile -e omw-1.4.zip .

# official CorefUD scorer (CRAC shared tasks), pinned; used via COREFUD_SCORER
ARG COREFUD_SCORER_COMMIT=4fd7b0e0c661aeeff88bc60c19ef507b84d1b590
RUN git clone -q https://github.com/ufal/corefud-scorer.git /opt/corefud-scorer \
    && git -C /opt/corefud-scorer checkout -q $COREFUD_SCORER_COMMIT \
    && pip install --no-cache-dir -r /opt/corefud-scorer/requirements.txt
ENV COREFUD_SCORER=/opt/corefud-scorer/corefud-scorer.py

# LLMv2 resolver/graph backend (text-corpuses-processing/dags/llm_v2): CPU-only torch (the LLM
# itself is reached over --llm-endpoint; only the sentence embedder runs here) and the
# embedders used for triple deduplication, downloaded at build time into HF_HOME
ENV HF_HOME=/opt/hf
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir "transformers>=4.35" "sentence-transformers>=2.2" "pydantic>=2" pyyaml \
       pymorphy3 razdel scikit-learn hdbscan tqdm \
    && python -c "from sentence_transformers import SentenceTransformer as S; \
[S(m) for m in ('all-mpnet-base-v2', 'paraphrase-multilingual-mpnet-base-v2')]" \
    && chmod -R a+rX /opt/hf
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

COPY coreference /stand
COPY text-corpuses-processing/dags /text-corpuses-processing/dags
ARG BUILD_INFO=unknown
RUN echo "$BUILD_INFO" > /stand/BUILD_INFO

WORKDIR /stand
ENV PYTHONUNBUFFERED=1 HOME=/tmp
CMD ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider"]
