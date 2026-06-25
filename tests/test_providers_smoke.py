"""Live provider smoke tests — skipped unless the matching API key is present.

These make a real (tiny) API call, so they are opt-in via env vars and never run
in normal CI. They verify the provider adapters actually talk to their backends.
"""
import os

import pytest

from rlm import RLM, get_provider


def _smoke(provider_name, **kw):
    provider = get_provider(provider_name, **kw)
    model = RLM(provider, depth=0, max_iters=4)
    out = model.run(
        "The magic number is 4242. Ignore everything else.",
        query="What is the magic number? Answer with FINAL(...) immediately.",
    )
    assert "4242" in out


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="no ANTHROPIC_API_KEY")
def test_anthropic_smoke():
    _smoke("anthropic")


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="no OPENAI_API_KEY")
def test_openai_smoke():
    _smoke("openai")


@pytest.mark.skipif(not os.environ.get("DEEPSEEK_API_KEY"), reason="no DEEPSEEK_API_KEY")
def test_deepseek_smoke():
    _smoke("deepseek")
