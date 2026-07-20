"""Tests for the RLM MCP server (integrations/mcp/rlm_mcp.py)."""
import importlib.util
import pathlib

# Load the standalone MCP-server script as a module.
_P = pathlib.Path(__file__).resolve().parents[1] / "integrations" / "mcp" / "rlm_mcp.py"
_SPEC = importlib.util.spec_from_file_location("rlm_mcp", _P)
rlm_mcp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rlm_mcp)


def test_initialize():
    r = rlm_mcp.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                        "params": {"protocolVersion": "2024-11-05"}})
    assert r["result"]["serverInfo"]["name"] == "rlm-mcp"
    assert r["result"]["protocolVersion"] == "2024-11-05"
    assert "tools" in r["result"]["capabilities"]


def test_initialized_notification_gets_no_reply():
    assert rlm_mcp.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None


def test_ping():
    assert rlm_mcp.handle({"jsonrpc": "2.0", "id": 9, "method": "ping"})["result"] == {}


def test_tools_list():
    r = rlm_mcp.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    tools = r["result"]["tools"]
    assert [t["name"] for t in tools] == ["rlm_query"]
    assert "query" in tools[0]["inputSchema"]["required"]


def test_tools_call_runs_rlm(monkeypatch):
    from rlm.providers.mock import MockProvider
    import rlm as rlm_pkg
    # `_run_rlm_query` does `from rlm import get_provider` at call time → patch rlm.get_provider.
    monkeypatch.setattr(rlm_pkg, "get_provider",
                        lambda name, **kw: MockProvider(script=["FINAL(mocked answer)"]))
    r = rlm_mcp.handle({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                        "params": {"name": "rlm_query",
                                   "arguments": {"input": "big context", "query": "q"}}})
    assert not r["result"].get("isError")
    assert "mocked answer" in r["result"]["content"][0]["text"]


def test_tools_call_missing_input_is_error():
    r = rlm_mcp.handle({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                        "params": {"name": "rlm_query", "arguments": {"query": "q"}}})
    assert r["result"]["isError"]  # no `file` and no `input`


def test_unknown_tool_is_error():
    r = rlm_mcp.handle({"jsonrpc": "2.0", "id": 5, "method": "tools/call",
                        "params": {"name": "nope"}})
    assert "error" in r


def test_unknown_method_is_error():
    r = rlm_mcp.handle({"jsonrpc": "2.0", "id": 6, "method": "no/such"})
    assert r["error"]["code"] == -32601
