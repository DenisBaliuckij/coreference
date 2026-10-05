import urllib.error

import pytest

from pipeline.llm.openai_client import OpenAICompatibleClient
from pipeline.resolvers.no_resolution_adapter import NoResolutionAdapter
from pipeline.types import CorefDocument


def _reply(content="Иван ушёл.", prompt_tokens=10, completion_tokens=4):
    return {"choices": [{"message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}}


def test_generate_sends_one_greedy_chat_request_without_thinking(monkeypatch):
    client = OpenAICompatibleClient("http://llm:8081/v1/", "qwen3.8:27b-mtp-128k")
    sent = []
    monkeypatch.setattr(client, "_post", lambda body: sent.append(body) or _reply())

    assert client.generate("Перепиши: Он ушёл.") == "Иван ушёл."
    assert client.url == "http://llm:8081/v1/chat/completions"
    body = sent[0]
    assert body["messages"] == [{"role": "user", "content": "Перепиши: Он ушёл."}]
    assert body["temperature"] == 0.0 and body["max_tokens"] == 1024
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert client.usage()["calls"] == 1 and client.usage()["completion_tokens"] == 4


def test_generate_retries_transient_errors(monkeypatch):
    client = OpenAICompatibleClient("http://llm/v1", "m", retries=3)
    monkeypatch.setattr("pipeline.llm.openai_client.time.sleep", lambda s: None)
    replies = iter([urllib.error.URLError("refused"), _reply("ok")])

    def post(body):
        r = next(replies)
        if isinstance(r, Exception):
            raise r
        return r

    monkeypatch.setattr(client, "_post", post)
    assert client.generate("x") == "ok"


def test_generate_fails_loudly_after_the_last_retry(monkeypatch):
    client = OpenAICompatibleClient("http://llm/v1", "m", retries=2)
    monkeypatch.setattr("pipeline.llm.openai_client.time.sleep", lambda s: None)

    def post(body):
        raise urllib.error.URLError("down")

    monkeypatch.setattr(client, "_post", post)
    with pytest.raises(RuntimeError, match="failed 2 times"):
        client.generate("x")


def test_empty_content_is_an_empty_string_not_none(monkeypatch):
    client = OpenAICompatibleClient("http://llm/v1", "m")
    monkeypatch.setattr(client, "_post", lambda body: _reply(content=None))
    assert client.generate("x") == ""


def test_no_resolution_passes_the_text_through_and_predicts_nothing():
    doc = CorefDocument(doc_id="d", text="John came. He left.", tokens=[], clusters=[])
    out = NoResolutionAdapter().resolve(doc)
    assert out.resolved_text == "John came. He left."
    assert out.clusters is None
    assert NoResolutionAdapter.language_support == "any"
