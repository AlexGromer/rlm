"""Provider registry / factory."""
from __future__ import annotations

from typing import Any

from .base import Provider


def get_provider(name: str, **kwargs: Any) -> Provider:
    """Construct a provider by name.

    Supported: ``anthropic``, ``openai``, ``deepseek``, ``claude_code``, ``mock``.
    SDK-backed providers import their SDK lazily, so this module imports cleanly
    even when a given SDK is not installed.
    """
    name = name.lower()
    if name == "anthropic":
        from .anthropic import AnthropicProvider
        return AnthropicProvider(**kwargs)
    if name in ("openai", "chatgpt", "gpt"):
        from .openai import OpenAIProvider
        return OpenAIProvider(**kwargs)
    if name == "deepseek":
        from .deepseek import DeepSeekProvider
        return DeepSeekProvider(**kwargs)
    if name in ("claude_code", "claude-code", "cc"):
        from .claude_code import ClaudeCodeProvider
        return ClaudeCodeProvider(**kwargs)
    if name == "mock":
        from .mock import MockProvider
        return MockProvider(**kwargs)
    raise ValueError(f"unknown provider: {name!r}")


__all__ = ["Provider", "get_provider"]
