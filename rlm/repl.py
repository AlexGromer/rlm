"""The persistent REPL environment that backs an RLM.

The prompt P is stored as the ``context`` variable here — never in the root
model's context window. The root model emits Python code (in ```repl fences),
which is executed in this namespace; only truncated stdout flows back to the
model. ``llm_query`` / ``rlm_query`` are exposed as callables so the model can
*programmatically* (in loops) invoke sub-models over slices of ``context`` —
this "symbolic recursion" is the key capability of Algorithm 1 in the paper.

SECURITY: this executes model-generated Python in-process. The namespace is
isolated from the host module globals, but it is NOT a security sandbox (code can
import os, open files, etc.). For untrusted use, run the REPL in a subprocess /
container / restricted interpreter. See README "Security".
"""
from __future__ import annotations

import io
import re
import traceback
from contextlib import redirect_stdout
from typing import Any, Callable

_REPL_BLOCK_RE = re.compile(r"```repl[ \t]*\r?\n(.*?)```", re.DOTALL)


class Repl:
    """A persistent Python namespace seeded with ``context`` and query helpers."""

    def __init__(
        self,
        context: Any,
        llm_query: Callable[[str], str],
        rlm_query: Callable[[Any, str], str],
    ) -> None:
        # A fresh namespace per RLM instance; __builtins__ is the real one so the
        # model can `import re`, use print(), etc. (the paper runs a real REPL).
        self.namespace: dict[str, Any] = {
            "__name__": "__rlm_repl__",
            "context": context,
            "llm_query": llm_query,
            "rlm_query": rlm_query,
        }

    def execute(self, code: str) -> str:
        """Exec ``code`` in the persistent namespace; return captured stdout.

        On exception, return the formatted traceback instead — the root model
        uses this execution feedback to repair its next code block.
        """
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                exec(compile(code, "<rlm-repl>", "exec"), self.namespace)
        except Exception:  # noqa: BLE001 - feedback to the model, not a crash
            buf.write("\n[REPL ERROR]\n" + traceback.format_exc())
        return buf.getvalue()

    def has_var(self, name: str) -> bool:
        return name in self.namespace

    def get_var(self, name: str) -> Any:
        return self.namespace[name]


def extract_repl_blocks(text: str) -> list[str]:
    """Return the bodies of all ```repl ... ``` code fences in ``text``."""
    return [m.group(1) for m in _REPL_BLOCK_RE.finditer(text)]


def strip_repl_blocks(text: str) -> str:
    """Remove ```repl fences so FINAL markers inside code examples don't match."""
    return _REPL_BLOCK_RE.sub("", text)


def extract_final(text: str) -> tuple[str, str] | None:
    """Find a terminal FINAL_VAR(name) or FINAL(answer) marker outside code.

    Returns ``("var", name)`` or ``("text", answer)``, or ``None`` if absent.
    FINAL_VAR is checked first (it is the more specific marker). Parentheses are
    matched with balancing so the answer/variable may itself contain parens.
    """
    scan = strip_repl_blocks(text)
    for marker, kind in (("FINAL_VAR(", "var"), ("FINAL(", "text")):
        idx = scan.rfind(marker)
        if idx == -1:
            continue
        start = idx + len(marker)
        payload = _balanced(scan, start)
        if payload is None:
            if kind != "text":
                continue
            # Unbalanced parens in a free-text answer: take the rest of the
            # response and drop a single trailing ')' if present.
            payload = scan[start:].rstrip()
            if payload.endswith(")"):
                payload = payload[:-1]
        payload = payload.strip()
        if kind == "var" and len(payload) >= 2 and payload[0] in "'\"" and payload[-1] == payload[0]:
            payload = payload[1:-1]  # strip exactly one balanced quote pair
        return kind, payload
    return None


def _balanced(s: str, start: int) -> str | None:
    """Return substring from ``start`` up to the matching close paren.

    String-literal aware: parentheses inside '...' or "..." are not counted, so
    an answer containing code/quotes (e.g. ``score("a(b")``) balances correctly.
    """
    depth = 1
    in_str: str | None = None
    i = start
    while i < len(s):
        c = s[i]
        if in_str is not None:
            if c == "\\" and in_str == '"':
                i += 2
                continue
            if c == in_str:
                in_str = None
        elif c in ('"', "'"):
            in_str = c
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return s[start:i]
        i += 1
    return None
