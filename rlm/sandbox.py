"""In-process restricted sandbox for the RLM REPL (see ``repl.py``).

The REPL execs model-generated Python. By default (``sandbox=True``) that exec is
hardened against the common in-process escape vectors while keeping the code the
RLM actually needs:

  * **No filesystem / network / process access** — ``open`` is removed and modules
    like ``os``, ``sys``, ``subprocess``, ``socket``, ``shutil``, ``pathlib``,
    ``ctypes``, ``pickle`` are not importable.
  * **Import allow-list** — only pure text/data stdlib modules (``re``, ``json``,
    ``collections`` …) can be imported.
  * **Dangerous builtins removed** — ``eval``, ``exec``, ``compile``,
    ``__import__``, ``input``, ``globals``, ``getattr`` … are gone.
  * **AST pre-pass** rejects dunder attribute/name access (``obj.__class__``,
    ``__builtins__`` …) — i.e. the ``().__class__.__subclasses__()`` builtins-escape
    gadget chain.

The injected ``context`` / ``llm_query`` / ``rlm_query`` and ordinary code
(slicing, comprehensions, ``re``/``json``/``collections``, ``print``) keep working.

**Defense-in-depth, NOT a kernel sandbox.** A determined adversary may still find a
gadget (e.g. ``str.format`` spec attribute access). For fully-untrusted input, ALSO
run the process under OS isolation (container / seccomp / unprivileged user, no
network). Use ``sandbox=False`` to opt back into the raw REPL for trusted content.
"""
from __future__ import annotations

import ast
import builtins as _b
from typing import Any, Callable

#: stdlib modules considered safe for text/data processing; all others are denied.
ALLOWED_MODULES = frozenset({
    "re", "json", "math", "statistics", "collections", "itertools", "functools",
    "string", "textwrap", "datetime", "decimal", "fractions", "heapq", "bisect",
    "difflib", "unicodedata", "base64", "hashlib", "hmac", "random", "csv", "io",
    "typing", "operator", "enum", "dataclasses", "numbers", "calendar", "zlib",
    "array", "copy", "pprint", "keyword", "html", "urllib.parse",
})

#: builtins exposed to model code (allow-list; everything else is withheld).
_SAFE_BUILTINS = frozenset({
    "abs", "all", "any", "ascii", "bin", "bool", "bytearray", "bytes", "callable",
    "chr", "complex", "dict", "divmod", "enumerate", "filter", "float", "format",
    "frozenset", "hasattr", "hash", "hex", "id", "int", "isinstance", "issubclass",
    "iter", "len", "list", "map", "max", "min", "next", "oct", "ord", "pow",
    "print", "range", "repr", "reversed", "round", "set", "slice", "sorted", "str",
    "sum", "tuple", "type", "zip",
    # exception types so try/except works
    "BaseException", "Exception", "ArithmeticError", "AssertionError",
    "AttributeError", "EOFError", "FloatingPointError", "IndexError", "KeyError",
    "LookupError", "MemoryError", "NameError", "NotImplementedError",
    "OverflowError", "RecursionError", "RuntimeError", "StopIteration",
    "TypeError", "UnicodeDecodeError", "UnicodeEncodeError", "UnicodeError",
    "ValueError", "ZeroDivisionError",
    # constants
    "True", "False", "None", "NotImplemented", "Ellipsis",
})

#: names that must never appear in model code (escape vectors), even by assignment.
_DENIED_NAMES = frozenset({
    "eval", "exec", "compile", "open", "__import__", "input", "breakpoint",
    "globals", "locals", "vars", "dir", "memoryview", "help", "exit", "quit",
    "getattr", "setattr", "delattr", "__builtins__", "__loader__", "__spec__",
    "__class__", "__base__", "__bases__", "__mro__", "__subclasses__",
    "__globals__", "__code__", "__closure__", "__dict__", "__getattribute__",
})


class SandboxViolation(Exception):
    """Raised (before execution) when code uses a forbidden construct."""


def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".")[0]
    if level != 0 or (name not in ALLOWED_MODULES and root not in ALLOWED_MODULES):
        raise SandboxViolation(f"import of '{name}' is not allowed in the RLM sandbox")
    return _b.__import__(name, globals, locals, fromlist, level)


def build_builtins() -> dict[str, Any]:
    out = {k: getattr(_b, k) for k in _SAFE_BUILTINS if hasattr(_b, k)}
    out["__import__"] = _guarded_import
    out["__build_class__"] = _b.__build_class__  # keep `class` statements working
    return out


def _is_dunder(s: str) -> bool:
    return len(s) > 4 and s.startswith("__") and s.endswith("__")


class _Auditor(ast.NodeVisitor):
    def visit_Attribute(self, node: ast.Attribute) -> None:
        if _is_dunder(node.attr):
            raise SandboxViolation(f"dunder attribute access '.{node.attr}' is not allowed")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in _DENIED_NAMES or _is_dunder(node.id):
            raise SandboxViolation(f"use of '{node.id}' is not allowed in the RLM sandbox")
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for a in node.names:
            root = a.name.split(".")[0]
            if a.name not in ALLOWED_MODULES and root not in ALLOWED_MODULES:
                raise SandboxViolation(f"import of '{a.name}' is not allowed in the RLM sandbox")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        root = (node.module or "").split(".")[0]
        if node.level != 0 or (node.module not in ALLOWED_MODULES and root not in ALLOWED_MODULES):
            raise SandboxViolation(f"import from '{node.module}' is not allowed in the RLM sandbox")
        self.generic_visit(node)


def audit(code: str) -> None:
    """Raise ``SandboxViolation`` if ``code`` uses a forbidden construct (pre-exec)."""
    _Auditor().visit(ast.parse(code, "<rlm-repl>", "exec"))


def make_namespace(context: Any, llm_query: Callable, rlm_query: Callable) -> dict[str, Any]:
    """A restricted REPL namespace: safe builtins + the injected context/helpers."""
    return {
        "__name__": "__rlm_repl__",
        "__builtins__": build_builtins(),
        "context": context,
        "llm_query": llm_query,
        "rlm_query": rlm_query,
    }
