from rlm import RLM
from rlm.providers.mock import MockProvider


def test_direct_final():
    root = MockProvider(script=["FINAL(hello world)"])
    assert RLM(root, depth=0).run("P", query="Q") == "hello world"


def test_repl_then_final_var_with_sub_call():
    # root drives the REPL; sub answers llm_query deterministically
    root = MockProvider(script=[
        "```repl\nx = llm_query('what is the magic number?')\nprint(x)\n```",
        "FINAL_VAR(x)",
    ])
    sub = MockProvider(fn=lambda msgs: "42")
    out = RLM(root, sub_provider=sub, depth=0).run("big context", query="magic?")
    assert out == "42"
    assert len(sub.calls) == 1
    assert "magic number" in sub.calls[0][0]["content"]


def test_rlm_query_fallback_at_depth_zero():
    root = MockProvider(script=[
        "```repl\nz = rlm_query('CTX', 'QQ')\nprint(z)\n```",
        "FINAL_VAR(z)",
    ])
    sub = MockProvider(fn=lambda msgs: "fallback-answer")
    out = RLM(root, sub_provider=sub, depth=0).run("P", query="Q")
    assert out == "fallback-answer"
    # at depth 0 rlm_query falls back to a single llm_query carrying ctx+query
    payload = sub.calls[-1][0]["content"]
    assert "QQ" in payload and "CTX" in payload


def test_recursion_depth_one_spawns_child():
    # one shared fn provider: behaves as child when it sees the child's query
    def root_fn(messages):
        joined = "\n".join(m["content"] for m in messages)
        if "Your query is: CHILD_Q" in joined:
            return "FINAL(child-done)"
        n_assistant = sum(1 for m in messages if m["role"] == "assistant")
        if n_assistant == 0:
            return "```repl\nr = rlm_query('SUBCTX', 'CHILD_Q')\nprint(r)\n```"
        return "FINAL_VAR(r)"

    root = MockProvider(fn=root_fn)
    out = RLM(root, depth=1).run("P", query="PARENT_Q")
    assert out == "child-done"


def test_repl_block_and_final_var_same_turn():
    # H-1 regression: one response computes a var AND returns it via FINAL_VAR
    root = MockProvider(script=["```repl\nresult = llm_query('q')\n```\nFINAL_VAR(result)"])
    sub = MockProvider(fn=lambda msgs: "computed-answer")
    out = RLM(root, sub_provider=sub, depth=0).run("P", query="Q")
    assert out == "computed-answer"


def test_final_var_missing_returns_marker():
    root = MockProvider(script=["FINAL_VAR(nonexistent)"])
    out = RLM(root, depth=0).run("P", query="Q")
    assert "not found" in out


def test_iteration_cap_forces_termination():
    calls = {"n": 0}

    def never_finishes(messages):
        calls["n"] += 1
        return "```repl\nprint('still going')\n```"

    root = MockProvider(fn=never_finishes)
    out = RLM(root, depth=0, max_iters=2).run("P", query="Q")
    # max_iters loop calls (2) + 1 forced final turn
    assert calls["n"] == 3
    assert isinstance(out, str) and out  # best-effort, non-empty
