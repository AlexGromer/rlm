"""Constant-size metadata helpers.

The whole point of an RLM (Recursive Language Models, Zhang/Kraska/Khattab, MIT
CSAIL, arXiv:2512.24601) is that the *root* model never sees the full prompt — it
sees only constant-size metadata about it (length, type, chunk sizes) and a
truncated view of any stdout it prints. The full prompt lives as the ``context``
variable inside the REPL (see ``rlm.repl``).
"""
from __future__ import annotations

from typing import Any


def describe_context(context: Any) -> dict[str, Any]:
    """Return constant-size metadata about the context object.

    ``context`` is either a ``str`` (one long document) or a ``list[str]``
    (a list of chunks/documents). The returned dict feeds the root system
    prompt placeholders: ``context_type``, ``context_total_length``,
    ``context_lengths``.
    """
    if isinstance(context, str):
        return {
            "context_type": "string",
            "context_total_length": len(context),
            "context_lengths": [len(context)],
        }
    if isinstance(context, (list, tuple)):
        lengths = [len(str(c)) for c in context]
        return {
            "context_type": f"list of {len(context)} chunks",
            "context_total_length": sum(lengths),
            "context_lengths": _summarize_lengths(lengths),
        }
    s = str(context)
    return {
        "context_type": type(context).__name__,
        "context_total_length": len(s),
        "context_lengths": [len(s)],
    }


def _summarize_lengths(lengths: list[int], max_shown: int = 20) -> list[int | str]:
    """Keep the chunk-length list itself constant-size for huge inputs."""
    if len(lengths) <= max_shown:
        return list(lengths)
    head: list[int | str] = list(lengths[: max_shown - 1])
    head.append(f"... (+{len(lengths) - len(head)} more chunks)")
    return head


def truncate_stdout(text: str, max_chars: int = 4000) -> str:
    """Truncate REPL stdout to a constant-size view for the root model.

    Keeps a head and a tail (semantics often live at both ends) plus an explicit
    marker of how much was elided, so the root model knows it is seeing a partial
    view and should use ``llm_query`` on variables instead of ``print``-ing them
    wholesale.
    """
    if text is None:
        return ""
    if len(text) <= max_chars:
        return text
    head = max_chars * 2 // 3
    tail = max_chars - head
    elided = len(text) - head - max(tail, 0)
    marker = (
        f"\n\n... [stdout truncated: {elided} chars elided of {len(text)} total; "
        "use llm_query on the underlying variable instead of printing it] ...\n\n"
    )
    if tail <= 0:  # e.g. max_chars == 0 → head == 0, emit only the marker
        return text[:head] + marker
    return text[:head] + marker + text[-tail:]
