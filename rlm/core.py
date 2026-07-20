"""Recursive Language Model — Algorithm 1.

Implements the RLM scaffold from "Recursive Language Models" (Zhang, Kraska,
Khattab; MIT CSAIL; arXiv:2512.24601v3). The root model never receives the full
prompt; it receives only constant-size metadata and drives a persistent Python
REPL (where the prompt lives as ``context``), printing truncated views and
calling ``llm_query`` / ``rlm_query`` over slices until it emits a FINAL answer.

Algorithm 1 (paper):
    state <- InitREPL(prompt=P)
    state <- AddFunction(state, sub_RLM)
    hist  <- [Metadata(state)]
    while True:
        code        <- LLM(hist)
        state,stdout<- REPL(state, code)
        hist        <- hist || code || Metadata(stdout)
        if state[Final] is set: return state[Final]
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

from .metadata import describe_context, truncate_stdout
from .providers.base import Message, Provider
from .repl import Repl, extract_final, extract_repl_blocks

_PROMPT_PATH = Path(__file__).parent / "prompts" / "root_system.md"


def _load_default_system_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


class RLM:
    """A Recursive Language Model scaffold around one or more providers."""

    def __init__(
        self,
        provider: Provider,
        *,
        sub_provider: Provider | None = None,
        depth: int = 1,
        max_iters: int = 12,
        stdout_max_chars: int = 4000,
        system_prompt: str | None = None,
        complete_kwargs: dict[str, Any] | None = None,
        verbose: bool = False,
        sandbox: bool = True,
    ) -> None:
        if depth < 0:
            raise ValueError("depth must be >= 0")
        self.provider = provider
        #: model used for llm_query sub-calls (paper: root=GPT-5, sub=GPT-5-mini)
        self.sub_provider = sub_provider or provider
        self.depth = depth
        self.max_iters = max_iters
        self.stdout_max_chars = stdout_max_chars
        self.system_prompt = system_prompt or _load_default_system_prompt()
        self.complete_kwargs = complete_kwargs or {}
        self.verbose = verbose
        #: sandbox the REPL exec (restricted builtins + import allow-list + AST guard)
        self.sandbox = sandbox

    # -- public API ---------------------------------------------------------
    def run(self, prompt: Any, query: str | None = None) -> str:
        """Answer ``query`` over ``prompt`` (loaded into the REPL as ``context``).

        If ``query`` is None the query is assumed to be embedded in ``prompt``
        itself (the paper's setup, where P contains both the question and the
        long context).
        """
        return self._run(context=prompt, query=query)

    # -- sub-call helpers (exposed inside the REPL namespace) ---------------
    def _make_llm_query(self) -> Callable[[str], str]:
        sub = self.sub_provider
        kw = self.complete_kwargs

        def llm_query(prompt: str) -> str:
            return sub.complete([{"role": "user", "content": str(prompt)}], **kw)

        return llm_query

    def _make_rlm_query(self) -> Callable[[Any, str], str]:
        if self.depth >= 1:
            child_box: list[RLM | None] = [None]  # built lazily on first use

            def rlm_query(context: Any, query: str) -> str:
                if child_box[0] is None:
                    child_box[0] = RLM(
                        self.provider,
                        sub_provider=self.sub_provider,
                        depth=self.depth - 1,
                        max_iters=self.max_iters,
                        stdout_max_chars=self.stdout_max_chars,
                        system_prompt=self.system_prompt,
                        complete_kwargs=self.complete_kwargs,
                        verbose=self.verbose,
                        sandbox=self.sandbox,
                    )
                return child_box[0].run(context, query=query)

            return rlm_query

        # At max depth, rlm_query falls back to a single llm_query (per Appendix C).
        llm_query = self._make_llm_query()

        def rlm_query_fallback(context: Any, query: str) -> str:
            return llm_query(f"{query}\n\nContext:\n{context}")

        return rlm_query_fallback

    # -- the loop -----------------------------------------------------------
    def _run(self, context: Any, query: str | None) -> str:
        meta = describe_context(context)
        repl = Repl(context, self._make_llm_query(), self._make_rlm_query(), sandbox=self.sandbox)

        system = self.system_prompt.format(**meta)
        kickoff = (
            "The `context` variable is loaded in the REPL. "
            + (f"Your query is: {query}" if query else
               "Answer the query associated with the context.")
            + " Begin: inspect the context, then run ```repl code or give a FINAL answer."
        )
        messages: list[Message] = [
            {"role": "system", "content": system},
            {"role": "user", "content": kickoff},
        ]

        for i in range(self.max_iters):
            response = self.provider.complete(messages, **self.complete_kwargs)
            messages.append({"role": "assistant", "content": response})
            self._log(f"iter {i}: {response[:200]!r}")

            # Algorithm 1 order: execute REPL code first (so a turn may compute a
            # variable AND return it via FINAL_VAR in the same response), then
            # check for a terminal FINAL marker.
            blocks = extract_repl_blocks(response)
            if blocks:
                stdout = "\n".join(repl.execute(b) for b in blocks)
                view = truncate_stdout(stdout, self.stdout_max_chars) or "(no output)"
                messages.append({"role": "user", "content": f"REPL output:\n{view}"})

            final = extract_final(response)
            if final is not None:
                return self._resolve_final(final, repl)

            if not blocks:
                messages.append({
                    "role": "user",
                    "content": "No ```repl code block and no FINAL marker found. "
                    "Run code in a ```repl block, or finish with FINAL(...) / FINAL_VAR(...).",
                })

        # Iteration cap hit: force a final answer in one last turn.
        messages.append({
            "role": "user",
            "content": "Iteration limit reached. Provide your best answer now using "
            "FINAL(...) or FINAL_VAR(...).",
        })
        response = self.provider.complete(messages, **self.complete_kwargs)
        final = extract_final(response)
        if final is not None:
            return self._resolve_final(final, repl)
        return response.strip()

    def _resolve_final(self, final: tuple[str, str], repl: Repl) -> str:
        kind, payload = final
        if kind == "var":
            if repl.has_var(payload):
                return str(repl.get_var(payload))
            return f"[FINAL_VAR('{payload}') not found in REPL]"
        return payload

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[rlm depth={self.depth}] {msg}", file=sys.stderr)
