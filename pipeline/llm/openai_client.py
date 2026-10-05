"""LLM client for an OpenAI-compatible chat endpoint (llama-server, vLLM, Ollama's /v1, ...).

Drop-in replacement for ``llm_v2.models.llm_client.LLMClient``: the LLMv2 resolver and graph
backend only call ``generate(prompt) -> str``. The bundled client loads a Hugging Face model
into the experiment process (Qwen2-1.5B-Instruct on CPU by default); this one sends the same
prompts to a model that is already served, e.g. the host's qwen3.8 on llama-server.

Settings used by the experiment and why:
- ``temperature`` 0 (greedy) so that runs are reproducible; the bundled client samples at 0.3.
- ``max_new_tokens`` 1024: the bundled 256 cuts off rewrites of a 5-sentence window
  (Russian especially) and longer JSON triple lists, which would corrupt the very text and
  graphs being compared.
- thinking disabled (``chat_template_kwargs.enable_thinking = false``): the prompts ask for a
  direct rewrite or a JSON list, and a reasoning model would otherwise spend its budget
  thinking.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request


class OpenAICompatibleClient:
    def __init__(self, base_url: str, model: str, max_new_tokens: int = 1024, temperature: float = 0.0,
                 top_p: float = 0.9, timeout: float = 600, retries: int = 3, enable_thinking: bool = False):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout
        self.retries = retries
        self.enable_thinking = enable_thinking
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def _body(self, prompt: str) -> dict:
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }

    def _post(self, body: dict) -> dict:
        request = urllib.request.Request(self.url, data=json.dumps(body).encode("utf-8"),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def generate(self, prompt: str) -> str:
        last_error = None
        for attempt in range(self.retries):
            try:
                data = self._post(self._body(prompt))
                usage = data.get("usage") or {}
                self.calls += 1
                self.prompt_tokens += usage.get("prompt_tokens", 0)
                self.completion_tokens += usage.get("completion_tokens", 0)
                return data["choices"][0]["message"].get("content") or ""
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
                last_error = exc
                time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"LLM endpoint {self.url} failed {self.retries} times: {last_error}")

    def usage(self) -> dict:
        return {"endpoint": self.url, "model": self.model, "calls": self.calls,
                "prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
                "temperature": self.temperature, "max_new_tokens": self.max_new_tokens}
