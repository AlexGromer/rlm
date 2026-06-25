"""OpenAI (ChatGPT) provider — Chat Completions API.

Also serves as the base for any OpenAI-compatible endpoint (e.g. DeepSeek) by
overriding ``base_url`` / ``env_key`` / default ``model``.
"""
from __future__ import annotations

import os
from typing import Any

from .base import Message, Provider


class OpenAIProvider(Provider):
    name = "openai"
    default_model = "gpt-5"
    env_key = "OPENAI_API_KEY"
    base_url: str | None = None

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        **client_kwargs: Any,
    ) -> None:
        self.model = model or self.default_model
        self._api_key = api_key or os.environ.get(self.env_key)
        self._base_url = base_url or self.base_url
        self._client_kwargs = client_kwargs
        self._client = None  # lazy

    def _ensure_client(self):
        if self._client is None:
            try:
                import openai
            except ImportError as e:  # pragma: no cover
                raise RuntimeError("pip install openai") from e
            kw = dict(self._client_kwargs)
            if self._base_url:
                kw["base_url"] = self._base_url
            self._client = openai.OpenAI(api_key=self._api_key, **kw)
        return self._client

    def complete(self, messages: list[Message], **kwargs: Any) -> str:
        client = self._ensure_client()
        resp = client.chat.completions.create(
            model=kwargs.pop("model", self.model),
            messages=[{"role": m["role"], "content": m["content"]} for m in messages],
            **kwargs,
        )
        return resp.choices[0].message.content or ""
