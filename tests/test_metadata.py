from rlm.metadata import describe_context, truncate_stdout


def test_describe_string():
    m = describe_context("hello world")
    assert m["context_type"] == "string"
    assert m["context_total_length"] == 11
    assert m["context_lengths"] == [11]


def test_describe_list():
    m = describe_context(["abc", "de", "f"])
    assert m["context_type"] == "list of 3 chunks"
    assert m["context_total_length"] == 6
    assert m["context_lengths"] == [3, 2, 1]


def test_describe_list_summarized_when_huge():
    m = describe_context([str(i) for i in range(100)])
    # lengths list itself stays constant-size
    assert len(m["context_lengths"]) <= 20
    assert "more chunks" in str(m["context_lengths"][-1])


def test_truncate_short_passthrough():
    assert truncate_stdout("short", 4000) == "short"


def test_truncate_long():
    out = truncate_stdout("x" * 10000, 1000)
    assert len(out) < 10000
    assert "truncated" in out
    assert out.startswith("x")
    assert out.endswith("x")


def test_truncate_zero_returns_marker_only():
    # max_chars=0 must NOT return the full text (the tail==0 edge case)
    out = truncate_stdout("x" * 100, 0)
    assert "truncated" in out
    assert "x" not in out
