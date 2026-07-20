"""Claude Code provider — uses the local `claude -p` CLI as the model M.

This is the "RLM *for Claude Code*" path: the same recursive scaffold, but every
root/sub model call is a headless Claude Code invocation. Useful when you want
RLM behavior driven by your local Claude Code (its model, auth, MCP) rather than
a raw API key.

The flattened chat history is fed to `claude -p` over **stdin** (not argv) so that
large sub-call payloads don't hit ARG_MAX limits.
"""
from __future__ import annotations

import shutil
import subprocess
from typing import Any

from .base import Message, Provider


class ClaudeCodeProvider(Provider):
    name = "claude_code"

    # Built-in agentic tools disabled for RLM sub-calls. `claude -p` is an agent,
    # not a completion endpoint: left enabled it answers the RLM scaffold with a
    # tool_use turn (e.g. Bash) that returns NO assistant text, so the RLM loop
    # sees empty output and never converges. Disabling them forces a plain-text
    # ```repl / FINAL(...) response — the only thing the RLM core can parse.
    _NO_TOOLS = (
        "Bash Read Write Edit MultiEdit Glob Grep WebFetch WebSearch "
        "Task Agent TodoWrite NotebookEdit"
    ).split()

    def __init__(
        self,
        model: str | None = None,
        claude_bin: str = "claude",
        timeout: int = 600,
        extra_args: list[str] | None = None,
        strict_mcp: bool = True,
        no_tools: bool = True,
    ) -> None:
        self.model = model
        self.claude_bin = claude_bin
        self.timeout = timeout
        self.extra_args = extra_args or []
        # strict_mcp: the nested `claude -p` loads NO MCP servers (only --mcp-config,
        # of which there are none). Avoids recursion when RLM runs *inside* an MCP
        # server (an rlm-MCP nested session would otherwise re-expose rlm_query), and
        # is faster — sub-calls are text slices that don't need MCP.
        self.strict_mcp = strict_mcp
        # no_tools: disable built-in agentic tools so `claude -p` returns text, not a
        # tool_use turn (see _NO_TOOLS). RLM needs a completion, not an agent action.
        self.no_tools = no_tools

    def complete(self, messages: list[Message], **kwargs: Any) -> str:
        if shutil.which(self.claude_bin) is None:
            raise RuntimeError(
                f"`{self.claude_bin}` CLI not found on PATH — install Claude Code "
                "or pass a different provider."
            )
        unsupported = {k for k in kwargs if k != "timeout"}
        if unsupported:
            import warnings
            warnings.warn(
                f"ClaudeCodeProvider ignores kwargs: {sorted(unsupported)}", stacklevel=2
            )
        # Install the RLM scaffold as the *real* system prompt via --system-prompt,
        # instead of letting flatten() demote it to stdin user-text. Without this the
        # nested `claude -p` runs under the user's OWN system prompt (output-style /
        # CLAUDE.md / hooks) and rejects the RLM scaffold as prompt-injection. NB: do
        # NOT use --bare to strip hooks — it also disables the apiKeyHelper (→ "Not
        # logged in"); --system-prompt alone fully replaces the persona.
        system, convo = self.split_system(messages)
        prompt = self.flatten(convo)
        cmd = [self.claude_bin, "-p", "--output-format", "text"]
        if self.strict_mcp:
            cmd.append("--strict-mcp-config")
        if system:
            cmd += ["--system-prompt", system]
        if self.model:
            cmd += ["--model", self.model]
        cmd += self.extra_args
        # --disallowedTools takes nargs '+', so keep it LAST: its greedy match would
        # otherwise swallow following flags/values.
        if self.no_tools:
            cmd += ["--disallowedTools", *self._NO_TOOLS]
        proc = subprocess.run(
            cmd,
            input=prompt,
            text=True,
            capture_output=True,
            timeout=kwargs.pop("timeout", self.timeout),
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"claude -p failed (rc={proc.returncode}): {proc.stderr.strip()[:500]}"
            )
        return proc.stdout.strip()
