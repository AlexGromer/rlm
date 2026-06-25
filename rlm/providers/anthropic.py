"""Anthropic (Claude) provider — Messages API."""
from __future__ import annotations

import os
from typing import Any

from .base import Message, Provider


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(
        self,
        model: str = "claude-opus-4-8",
        api_key: str | None = None,
        max_tokens: int = 4096,
        **client_kwargs: Any,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client_kwargs = client_kwargs
        self._client = None  # lazy

    def _ensure_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as e:  # pragma: no cover
                raise RuntimeError("pip install anthropic") from e
            self._client = anthropic.Anthropic(api_key=self._api_key, **self._client_kwargs)
        return self._client

    def complete(self, messages: list[Message], **kwargs: Any) -> str:
        client = self._ensure_client()
        system, rest = self.split_system(messages)
        resp = client.messages.create(
            model=kwargs.pop("model", self.model),
            max_tokens=kwargs.pop("max_tokens", self.max_tokens),
            system=system or "",
            messages=[{"role": m["role"], "content": m["content"]} for m in rest],
            **kwargs,
        )
        return "".join(block.text for block in resp.content if getattr(block, "type", None) == "text")
