"""Provider abstraction — one interface, any model.

A Provider turns a chat history into a single completion string. This is the
*only* model-specific surface in the library: the RLM core (Algorithm 1) and the
REPL are entirely provider-agnostic, so the same recursive scaffold runs on
Claude (Anthropic), ChatGPT (OpenAI), DeepSeek, or the local Claude Code CLI just
by swapping the Provider.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

Message = dict[str, str]  # {"role": "system"|"user"|"assistant", "content": str}


class Provider(ABC):
    """Minimal LLM interface used by the RLM scaffold."""

    #: human-readable provider name (for logs/metadata)
    name: str = "provider"

    @abstractmethod
    def complete(self, messages: list[Message], **kwargs: Any) -> str:
        """Return the model's text completion for the given chat history."""

    @staticmethod
    def split_system(messages: list[Message]) -> tuple[str | None, list[Message]]:
        """Split a leading system message from the rest (APIs differ on this)."""
        if messages and messages[0]["role"] == "system":
            return messages[0]["content"], messages[1:]
        return None, list(messages)

    @staticmethod
    def flatten(messages: list[Message]) -> str:
        """Flatten a chat history to a single prompt string (for CLI-style models)."""
        parts = []
        for m in messages:
            parts.append(f"[{m['role'].upper()}]\n{m['content']}")
        return "\n\n".join(parts)
