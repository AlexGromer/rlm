# RLM — Recursive Language Models

A small, **model-agnostic** reference implementation of *Recursive Language
Models* (Zhang, Kraska, Khattab — MIT CSAIL, [arXiv:2512.24601](https://arxiv.org/abs/2512.24601);
authors' code: [github.com/alexzhang13/rlm](https://github.com/alexzhang13/rlm)).

An RLM processes prompts **larger than the model's context window** by treating
the prompt as an *environment* rather than feeding it into the model directly.
The prompt is stored as a `context` variable in a persistent Python REPL; the
root model sees only constant-size **metadata** about it, then writes code that
peeks, chunks, and **recursively calls itself** (`llm_query` / `rlm_query`) over
slices until it emits a `FINAL` answer.

This is Algorithm 1 of the paper, provider-agnostic: the same scaffold runs on
Claude (API), ChatGPT, DeepSeek, or your local **Claude Code** CLI — just swap
the provider.

## Install

```bash
cd /opt/rlm
pip install -e ".[all]"      # anthropic + openai SDKs; deepseek uses the openai SDK
# or no extras — the lib imports fine; install only the SDK you need.
```

## Use

```bash
# any OpenAI-compatible / Anthropic model (set the matching *_API_KEY env var)
echo "long document..." | rlm run --provider anthropic --query "summarize the argument"
rlm run --provider deepseek --file big.txt --query "find the off-by-one bug" --depth 1
rlm run --provider openai   --file logs.txt --query "what caused the 5xx spike?"

# RLM driven by the local Claude Code CLI (no API key needed)
python3 integrations/claude_code/rlm_cc.py --file huge.txt --query "what changed?"
```

```python
from rlm import RLM, get_provider

# root = strong model, sub-calls = cheaper model (as in the paper: GPT-5 / GPT-5-mini)
model = RLM(get_provider("anthropic", model="claude-opus-4-8"),
            sub_provider=get_provider("anthropic", model="claude-haiku-4-5-20251001"),
            depth=1)
print(model.run(huge_text, query="What is the magic number?"))
```

### Recursion depth
- `depth=0` — `rlm_query` falls back to a single `llm_query` (no nested loops).
- `depth=1` — sub-calls may themselves be RLMs (the paper's default sweet spot).
- `depth=N` — N levels of recursive RLM sub-calls; each level caps its iterations.

## Layout

| Path | Purpose |
|------|---------|
| `rlm/core.py` | `RLM` — the Algorithm-1 loop (root loop, history, depth, FINAL) |
| `rlm/repl.py` | persistent REPL: `context` var, `llm_query`/`rlm_query`, exec + parsing |
| `rlm/metadata.py` | constant-size context/stdout summaries |
| `rlm/providers/` | `anthropic`, `openai`, `deepseek`, `claude_code`, `mock` |
| `rlm/prompts/root_system.md` | the root system prompt (Appendix C of the paper) |
| `rlm/cli.py` | `rlm run ...` |
| `integrations/claude_code/` | skill + launcher using `claude -p` as the model |
| `tests/` | deterministic core tests (MockProvider — no API calls) |

## Security

The REPL **executes model-generated Python in-process**. The namespace is
isolated from this library's globals, but it is **not a security sandbox** — code
can import `os`, open files, make network calls, etc. Only run RLM on prompts
from sources you trust, or run the REPL inside a subprocess / container /
restricted interpreter. See `rlm/repl.py`.

## Status / not included

- Reference implementation, not production-hardened. No real sandbox (see above).
- Fine-tuning a natively-recursive model (RLM-Qwen3-8B, Appendix A of the paper)
  is **out of scope** — this repo is the inference scaffold only. The recipe is
  distilled for reference in [`docs/FINE_TUNING_RECIPE.md`](docs/FINE_TUNING_RECIPE.md).
