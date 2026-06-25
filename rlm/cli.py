"""Command-line interface: `rlm run ...`.

Examples:
    echo "long prompt..." | rlm run --provider anthropic --query "summarize"
    rlm run --provider deepseek --file big.txt --query "find the bug" --depth 1
    rlm run --provider claude_code --file book.txt --query "who won?" --verbose
    rlm run --provider mock --script demo.json   # offline, no API
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

from . import RLM, get_provider


def _build_provider(name: str, model: str | None):
    kwargs = {}
    if model:
        kwargs["model"] = model
    if name == "mock":
        kwargs.pop("model", None)
    return get_provider(name, **kwargs)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rlm", description="Recursive Language Models")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="run an RLM over a prompt")
    run.add_argument("--provider", default="anthropic",
                     help="anthropic|openai|deepseek|claude_code|mock")
    run.add_argument("--model", default=None)
    run.add_argument("--sub-provider", default=None,
                     help="provider for llm_query sub-calls (default: same as --provider)")
    run.add_argument("--sub-model", default=None)
    run.add_argument("--depth", type=int, default=1)
    run.add_argument("--max-iters", type=int, default=12)
    run.add_argument("--query", default=None, help="the question to answer over the context")
    run.add_argument("--file", default=None, help="read the context/prompt from a file")
    run.add_argument("--script", default=None,
                     help="(mock only) JSON list of scripted completions")
    run.add_argument("--verbose", action="store_true")
    run.add_argument("prompt", nargs="?", default=None,
                     help="the prompt/context (else --file or stdin)")

    args = p.parse_args(argv)

    if args.provider == "mock":
        script = (
            json.loads(pathlib.Path(args.script).read_text(encoding="utf-8"))
            if args.script else ["FINAL(no script)"]
        )
        provider = get_provider("mock", script=script)
    else:
        provider = _build_provider(args.provider, args.model)

    sub_provider = (
        _build_provider(args.sub_provider, args.sub_model) if args.sub_provider else None
    )

    if args.prompt is not None:
        context = args.prompt
    elif args.file:
        context = pathlib.Path(args.file).read_text(encoding="utf-8")
    else:
        context = sys.stdin.read()

    model = RLM(
        provider,
        sub_provider=sub_provider,
        depth=args.depth,
        max_iters=args.max_iters,
        verbose=args.verbose,
    )
    print(model.run(context, query=args.query))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
