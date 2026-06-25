from rlm.repl import Repl, extract_final, extract_repl_blocks, strip_repl_blocks


def _repl():
    return Repl(context="CTX", llm_query=lambda p: f"LLM[{p}]", rlm_query=lambda c, q: "RLM")


def test_execute_captures_stdout_and_persists_state():
    r = _repl()
    assert r.execute("a = 21\nprint(a * 2)").strip() == "42"
    # namespace persists across executes
    assert r.execute("print(a + 1)").strip() == "22"


def test_context_and_helpers_available():
    r = _repl()
    out = r.execute("print(context); print(llm_query('q'))")
    assert "CTX" in out and "LLM[q]" in out


def test_execute_error_is_feedback_not_crash():
    r = _repl()
    out = r.execute("1/0")
    assert "REPL ERROR" in out and "ZeroDivisionError" in out


def test_extract_repl_blocks():
    text = "intro\n```repl\nprint(1)\n```\nmid\n```repl\nprint(2)\n```\n"
    assert extract_repl_blocks(text) == ["print(1)\n", "print(2)\n"]


def test_extract_final_text_and_var():
    assert extract_final("done FINAL(hello)") == ("text", "hello")
    assert extract_final("ok FINAL_VAR(answer)") == ("var", "answer")
    assert extract_final("nothing here") is None


def test_extract_final_balanced_parens():
    assert extract_final("FINAL(a (nested) b)") == ("text", "a (nested) b")


def test_final_inside_code_block_is_ignored():
    text = "```repl\nx = 'FINAL(not this)'\n```\nFINAL_VAR(x)"
    assert "FINAL(not this)" not in strip_repl_blocks(text)
    assert extract_final(text) == ("var", "x")


def test_extract_final_unbalanced_paren_fallback():
    # a free-text answer with an unclosed paren still terminates the loop
    assert extract_final("FINAL(use f(x for x>0)") == ("text", "use f(x for x>0")


def test_extract_final_paren_inside_string_literal():
    assert extract_final('FINAL(call score("a(b") now)') == ("text", 'call score("a(b") now')


def test_extract_final_var_quote_stripping():
    assert extract_final("FINAL_VAR('answer')") == ("var", "answer")
    assert extract_final('FINAL_VAR("answer")') == ("var", "answer")
    assert extract_final("FINAL_VAR(answer)") == ("var", "answer")
