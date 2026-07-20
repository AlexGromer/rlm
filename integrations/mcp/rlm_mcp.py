#!/usr/bin/env python3
"""Always-on RLM integration: a stdio MCP server exposing `rlm_query`.

Lets the main model offload "answer a question over an input too large to read
directly" to a Recursive Language Model — as a normal MCP tool call, no manual
/rlm skill. Add this server to your MCP config so it is always available; the
model reaches for it whenever it faces oversized context.

Design notes (see /opt/rlm/docs and the RLM-always-on backlog card):
  * The REPL runs SANDBOXED by default (rlm/sandbox.py) — this is a precondition
    for making RLM always-on: the tool execs model-generated Python.
  * Sub-calls default to a cheap **API** model (haiku), NOT nested `claude -p`, to
    avoid spawning nested Claude-Code processes (and rlm-MCP recursion).
  * Gate to genuinely-oversized inputs — RLM issues many sub-calls (cost/latency).

Pure stdlib: newline-delimited JSON-RPC 2.0 over stdin/stdout (no MCP SDK dep).

Env config:
  RLM_MCP_PROVIDER    root provider    (default 'anthropic'; 'claude_code', 'openai',
                                        'deepseek', 'mock' for tests)
  RLM_MCP_MODEL       root model       (default 'claude-sonnet-5')
  RLM_MCP_SUB_MODEL   sub-call model   (default 'claude-haiku-4-5-20251001')
  RLM_MCP_DEPTH       default depth    (default '1')
  RLM_MCP_MAX_ITERS   root-loop cap    (default '12')
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import traceback

# Make the `rlm` package importable when run as a standalone script.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "rlm-mcp", "version": "0.1.0"}

TOOLS = [
    {
        "name": "rlm_query",
        "description": (
            "Answer a question over an input TOO LARGE to read/reason over directly "
            "(a multi-megabyte log, a whole-codebase dump, a long transcript). A "
            "Recursive Language Model loads the input into a sandboxed Python REPL and "
            "recursively sub-queries the model over slices, instead of putting it all in "
            "context. Use ONLY for genuinely oversized inputs — it issues many sub-calls "
            "(slow/costly). For inputs that fit in context, read them directly instead."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The question to answer over the large input."},
                "file": {"type": "string", "description": "Path to the large input file (use this OR `input`)."},
                "input": {"type": "string", "description": "The large input text inline (use this OR `file`)."},
                "depth": {"type": "integer", "description": "Recursion depth (0=flat sub-calls, 1=recursive; default 1).", "default": 1},
            },
            "required": ["query"],
        },
    }
]


def _run_rlm_query(args: dict) -> str:
    from rlm import RLM, get_provider

    query = args.get("query")
    if not query:
        raise ValueError("`query` is required")
    if args.get("file"):
        context = pathlib.Path(args["file"]).read_text(encoding="utf-8", errors="replace")
    elif args.get("input") is not None:
        context = args["input"]
    else:
        raise ValueError("provide either `file` (path) or `input` (inline text)")

    prov_name = os.environ.get("RLM_MCP_PROVIDER", "anthropic")
    root_model = os.environ.get("RLM_MCP_MODEL", "claude-sonnet-5")
    sub_model = os.environ.get("RLM_MCP_SUB_MODEL", "claude-haiku-4-5-20251001")
    depth = int(args.get("depth", os.environ.get("RLM_MCP_DEPTH", "1")))
    max_iters = int(os.environ.get("RLM_MCP_MAX_ITERS", "12"))

    root = get_provider(prov_name, model=root_model)
    # Sub-calls on a cheap model of the same provider (NOT nested claude_code).
    sub = get_provider(prov_name, model=sub_model)
    model = RLM(root, sub_provider=sub, depth=depth, max_iters=max_iters, sandbox=True)
    return model.run(context, query=query)


# ── JSON-RPC / MCP plumbing ───────────────────────────────────────────────────
def _result(id_, result):
    return {"jsonrpc": "2.0", "id": id_, "result": result}


def _error(id_, code, message):
    return {"jsonrpc": "2.0", "id": id_, "error": {"code": code, "message": message}}


def handle(msg: dict):
    """Return a response dict, or None for notifications (no reply)."""
    method = msg.get("method")
    id_ = msg.get("id")

    if method == "initialize":
        client_ver = (msg.get("params") or {}).get("protocolVersion", PROTOCOL_VERSION)
        return _result(id_, {
            "protocolVersion": client_ver,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": SERVER_INFO,
        })
    if method in ("notifications/initialized", "initialized"):
        return None  # notification — no reply
    if method == "ping":
        return _result(id_, {})
    if method == "tools/list":
        return _result(id_, {"tools": TOOLS})
    if method == "tools/call":
        params = msg.get("params") or {}
        name = params.get("name")
        if name != "rlm_query":
            return _error(id_, -32602, f"unknown tool: {name}")
        try:
            answer = _run_rlm_query(params.get("arguments") or {})
            return _result(id_, {"content": [{"type": "text", "text": answer}]})
        except Exception as e:  # tool errors are returned as isError content, not crashes
            return _result(id_, {
                "content": [{"type": "text", "text": f"rlm_query failed: {e}\n{traceback.format_exc()[:800]}"}],
                "isError": True,
            })
    if id_ is not None:
        return _error(id_, -32601, f"method not found: {method}")
    return None  # unknown notification


def main() -> int:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        resp = handle(msg)
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
