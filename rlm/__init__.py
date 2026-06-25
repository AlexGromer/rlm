"""rlm — Recursive Language Models (reference implementation).

Paper: "Recursive Language Models", Zhang, Kraska, Khattab (MIT CSAIL),
arXiv:2512.24601. Reference code by the authors: github.com/alexzhang13/rlm.

Quick start:
    from rlm import RLM, get_provider
    model = RLM(get_provider("anthropic", model="claude-opus-4-8"))
    answer = model.run(huge_prompt, query="What is the magic number?")
"""
from __future__ import annotations

from .core import RLM
from .providers import get_provider

__all__ = ["RLM", "get_provider"]
__version__ = "0.1.0"
