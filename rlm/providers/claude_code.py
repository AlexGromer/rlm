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

    def __init__(
        self,
        model: str | None = None,
        claude_bin: str = "claude",
        timeout: int = 600,
        extra_args: list[str] | None = None,
        strict_mcp: bool = True,
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
        prompt = self.flatten(messages)
        cmd = [self.claude_bin, "-p", "--output-format", "text"]
        if self.strict_mcp:
            cmd.append("--strict-mcp-config")
        cmd += self.extra_args
        if self.model:
            cmd += ["--model", self.model]
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
