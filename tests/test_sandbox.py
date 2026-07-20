"""Sandbox tests for the RLM REPL (rlm/sandbox.py)."""
from rlm.repl import Repl


def _box(sandbox=True):
    return Repl(context="CTX", llm_query=lambda p: f"LLM[{p}]",
                rlm_query=lambda c, q: "RLM", sandbox=sandbox)


def test_normal_code_and_helpers_work():
    r = _box()
    assert r.execute("a = 21\nprint(a * 2)").strip() == "42"
    assert "CTX" in r.execute("print(context)")
    assert "LLM[q]" in r.execute("print(llm_query('q'))")
    assert "RLM" in r.execute("print(rlm_query('c', 'q'))")


def test_allowed_import_works():
    out = _box().execute("import re\nprint(re.findall(r'\\d+', 'a1b22'))")
    assert "['1', '22']" in out


def test_comprehension_and_slicing_work():
    out = _box().execute("print([w.upper() for w in context[:3]])")
    assert "SANDBOX BLOCKED" not in out and "REPL ERROR" not in out


def test_open_is_blocked():
    out = _box().execute("open('/etc/passwd')")
    assert "SANDBOX BLOCKED" in out and "open" in out


def test_denied_imports_blocked():
    assert "SANDBOX BLOCKED" in _box().execute("import os\nprint(os.getcwd())")
    assert "SANDBOX BLOCKED" in _box().execute("import subprocess")
    assert "SANDBOX BLOCKED" in _box().execute("from socket import socket")


def test_dunder_escape_gadgets_blocked():
    assert "SANDBOX BLOCKED" in _box().execute("print(().__class__.__bases__)")
    assert "SANDBOX BLOCKED" in _box().execute("[].__class__.__base__.__subclasses__()")
    assert "SANDBOX BLOCKED" in _box().execute("print(__builtins__)")


def test_eval_exec_compile_blocked():
    assert "SANDBOX BLOCKED" in _box().execute("eval('1+1')")
    assert "SANDBOX BLOCKED" in _box().execute("exec('x=1')")
    assert "SANDBOX BLOCKED" in _box().execute("compile('1', '<s>', 'eval')")
    assert "SANDBOX BLOCKED" in _box().execute("getattr(context, 'upper')")


def test_state_persists_across_execute():
    r = _box()
    r.execute("acc = []")
    r.execute("acc.append(1)")
    assert "[1]" in r.execute("print(acc)")


def test_runtime_error_is_fed_back_not_raised():
    # A normal (non-sandbox) exception still returns feedback, doesn't crash.
    assert "REPL ERROR" in _box().execute("1/0")


def test_sandbox_off_restores_raw_repl():
    out = _box(sandbox=False).execute("import os\nprint(type(os).__name__)")
    assert "SANDBOX BLOCKED" not in out
    assert "module" in out
