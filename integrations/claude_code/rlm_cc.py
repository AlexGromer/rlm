#!/usr/bin/env python3
"""RLM-for-Claude-Code launcher.

Runs the RLM scaffold using the local `claude -p` CLI as the model M, so you can
process a prompt far larger than the model's context window from inside Claude
Code: the big input is loaded into a REPL variable and Claude recursively
sub-calls itself over slices.

Usage:
    python3 rlm_cc.py --file huge.txt --query "What changed between v1 and v2?"
    cat huge.txt | python3 rlm_cc.py --query "summarize" --depth 1
"""
from __future__ import annotations

import argparse
import pathlib
import sys

# Make the sibling `rlm` package importable when run as a standalone script.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from rlm import RLM, get_provider  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="RLM over the local Claude Code CLI")
    ap.add_argument("--file", help="context/prompt file (else stdin)")
    ap.add_argument("--query", required=True, help="question to answer over the context")
    ap.add_argument("--model", default=None, help="claude --model override")
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--max-iters", type=int, default=12)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    context = (
        pathlib.Path(args.file).read_text(encoding="utf-8") if args.file else sys.stdin.read()
    )
    provider = get_provider("claude_code", model=args.model)
    model = RLM(provider, depth=args.depth, max_iters=args.max_iters, verbose=args.verbose)
    print(model.run(context, query=args.query))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
