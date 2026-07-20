# RLM MCP server — always-on `rlm_query`

`rlm_mcp.py` is a stdio MCP server exposing RLM as an **`rlm_query`** tool, so the
main model can offload *"answer a question over an input too large to read/reason
over directly"* to a Recursive Language Model automatically — no manual `/rlm`
skill. This is the "constantly-available" integration (vs the on-demand skill).

Pure stdlib (newline-delimited JSON-RPC 2.0 over stdio) — no MCP SDK dependency.

## Tool

`rlm_query(query, file?, input?, depth=1)` → the FINAL answer.
- `query` (required) — the question to answer over the large input.
- `file` **or** `input` — the oversized context (path, or inline text).
- `depth` — recursion depth (0 flat, 1 recursive; default 1).
- The tool description tells the model to use it **only** for genuinely-oversized
  inputs (RLM issues many sub-calls — slow/costly).

## Register (choose one) — takes effect after a **session restart**

**A. Always-on.** Add to `projects[<path>].mcpServers` in `~/.claude.json` (or, in
this framework, add `rlm` to the profile-mcp essential list in
`~/.claude/mcp-servers/profile-mcp/profiles.go` and rebuild):

```json
"rlm": {
  "command": "python3",
  "args": ["/opt/rlm/integrations/mcp/rlm_mcp.py"],
  "env": {
    "RLM_MCP_PROVIDER": "anthropic",
    "RLM_MCP_MODEL": "claude-sonnet-5",
    "RLM_MCP_SUB_MODEL": "claude-haiku-4-5-20251001",
    "ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY}"
  }
}
```

**B. On-demand.** Skip registration and use the `/rlm` skill or `rlm_cc.py` directly.

## Notes
- **Needs `ANTHROPIC_API_KEY`** with the default provider. Set
  `RLM_MCP_PROVIDER=claude_code` to drive calls through `claude -p` instead — heavier
  (nested headless Claude-Code processes) and a recursion hazard if this server is
  itself loaded in the nested session.
- Sub-calls default to a **cheap** model (haiku) to bound cost; the REPL is
  **sandboxed** (`rlm/sandbox.py`) by default — the security precondition for making
  RLM always-on.
- Tested in `tests/test_rlm_mcp.py` (protocol + tool dispatch, no API).
