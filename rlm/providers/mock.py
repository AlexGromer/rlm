"""Deterministic provider for tests and offline demos — no API, no cost.

Two modes:
- scripted: hand it a list of canned completions; it returns them in order
  (used to drive the core loop deterministically in unit tests).
- echo-callable: hand it a function ``fn(messages) -> str`` for custom behavior.
"""
from __future__ import annotations

from typing import Any, Callable

from .base import Message, Provider


class MockProvider(Provider):
    name = "mock"

    def __init__(
        self,
        script: list[str] | None = None,
        fn: Callable[[list[Message]], str] | None = None,
    ) -> None:
        if (script is None) == (fn is None):
            raise ValueError("provide exactly one of `script` or `fn`")
        self._script = list(script) if script is not None else None
        self._fn = fn
        self.calls: list[list[Message]] = []

    def complete(self, messages: list[Message], **kwargs: Any) -> str:
        self.calls.append(list(messages))
        if self._fn is not None:
            return self._fn(messages)
        if not self._script:
            # Safety net: if a test under-scripts, terminate cleanly.
            return "FINAL(mock script exhausted)"
        return self._script.pop(0)
