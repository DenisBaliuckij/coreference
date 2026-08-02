from __future__ import annotations

from typing import Literal, Protocol


class GraphBackendAdapter(Protocol):
    name: str
    backend_name: str
    language_support: set[str] | Literal["any"]

    def build(self, text: str) -> dict: ...
