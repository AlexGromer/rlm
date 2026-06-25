---
name: rlm
description: Process a prompt/document far larger than the context window using a Recursive Language Model (RLM) — loads the input into a REPL variable and recursively sub-calls the model over slices. Use when an input is too big to read directly, or when a long-context task degrades with naive context-stuffing.
---

# RLM (Recursive Language Models) for Claude Code

Use this when the input is **too large to fit / reason over directly** (e.g. a
multi-megabyte log, a whole codebase dump, a long transcript) or when a
long-context task is degrading. Instead of stuffing everything into context, the
RLM scaffold loads the input into a Python REPL variable and lets the model write
code that peeks/chunks/recursively sub-queries itself over slices.

Reference: "Recursive Language Models" (Zhang, Kraska, Khattab, MIT CSAIL,
arXiv:2512.24601). Implementation: `/opt/rlm`.

## How to run

```bash
python3 /opt/rlm/integrations/claude_code/rlm_cc.py \
  --file <big_input_file> \
  --query "<the question to answer over it>" \
  --depth 1
```

- `--file` (or pipe via stdin) — the large context.
- `--query` — the question to answer over that context.
- `--depth` — recursion depth (0 = sub-calls are single LLM calls; 1 = sub-calls
  are themselves RLMs; default 1).
- The model M is the local `claude -p` CLI (uses your Claude Code model/auth).

## When NOT to use

- The input already fits comfortably in context → just read it directly.
- You need exact byte-level edits across the whole file → use Read/Edit, not RLM.

## Other models

The same core works with API providers (no Claude Code needed):

```bash
echo "$BIG" | rlm run --provider deepseek  --query "..." --depth 1   # DeepSeek
echo "$BIG" | rlm run --provider openai    --query "..." --depth 1   # ChatGPT
echo "$BIG" | rlm run --provider anthropic --query "..." --depth 1   # Claude API
```
