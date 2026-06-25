# RLM — Design & Paper Notes

Reference: **Recursive Language Models**, Alex L. Zhang, Tim Kraska, Omar Khattab
(MIT CSAIL), arXiv:2512.24601v3 (May 2026). Authors' code: github.com/alexzhang13/rlm.

## 1. The idea

Frontier models degrade on long contexts ("context rot") well before their
window limit, and the degradation is *task-dependent* (a line-by-line task like
OOLONG degrades far sooner than needle-in-a-haystack). RLMs sidestep this by not
putting the long prompt in the model's context at all:

> Treat the prompt `P` as part of an **external environment** (a Python REPL).
> The root model sees only constant-size **metadata** about `P` and writes code
> that peeks into / decomposes / **recursively calls a model over slices of** `P`,
> building the answer in REPL variables, until it emits a `FINAL` answer.

Reported result: RLM(GPT-5, depth=1) beats GPT-5 + compaction / CodeAct / Claude
Code on four long-context benchmarks (CodeQA, BrowseComp-Plus, OOLONG,
OOLONG-Pairs) at comparable cost, scaling to 10M+ token inputs.

## 2. Algorithm 1 (what we implement)

```
state  <- InitREPL(prompt = P)          # P is a REPL variable, not in context
state  <- AddFunction(state, sub_RLM)   # expose recursive sub-call to the model
hist   <- [Metadata(state)]             # root sees only metadata (len, type, chunks)
while True:
    code         <- LLM(hist)           # root emits a ```repl code block
    state, stdout<- REPL(state, code)   # execute it; capture stdout
    hist         <- hist || code || Metadata(stdout)   # only *truncated* stdout returns
    if state[Final] is set:             # root emitted FINAL(...) / FINAL_VAR(...)
        return state[Final]
```

Three design choices that distinguish it from a naive scaffold (Algorithm 2 in
the paper):
1. **Symbolic handle** — the prompt is a variable; never copied into the root's
   window. (`rlm/repl.py`: `context`.)
2. **Unbounded output** — the answer comes from a REPL variable (`FINAL_VAR`) or a
   marker (`FINAL`), so it isn't bounded by the model's output limit.
3. **Symbolic recursion** — code can call the model *programmatically in loops*
   (`llm_query`, `rlm_query`) over `Ω(|P|)` slices, not just a few verbalized
   sub-calls.

## 3. Our architecture (`/opt/rlm`)

| Component | File | Maps to |
|-----------|------|---------|
| Root loop | `rlm/core.py` `RLM._run` | the `while True` loop above |
| REPL env | `rlm/repl.py` `Repl` | `InitREPL` + `REPL(state, code)` |
| Metadata | `rlm/metadata.py` | `Metadata(state)`, truncated `Metadata(stdout)` |
| Sub-calls | `core._make_llm_query`, `_make_rlm_query` | `llm_query`, `sub_RLM` |
| Termination | `repl.extract_final` + `core._resolve_final` | `state[Final]` |
| Root prompt | `rlm/prompts/root_system.md` | Appendix C (1a)+(1c) |
| Models | `rlm/providers/*` | the LLM `M` (any backend) |

### Interface (from Appendix C, verbatim semantics)
- `context` — the prompt variable (a `str` or `list[str]` of chunks).
- `llm_query(prompt) -> str` — a single sub-LLM call (handles ~500K chars).
- `rlm_query(context, query) -> str` — a full recursive RLM sub-loop for complex
  sub-tasks; **falls back to `llm_query` at max depth**.
- code goes in ` ```repl … ``` ` fences; only truncated stdout returns to the root.
- terminate with `FINAL(answer)` or `FINAL_VAR(variable_name)`.

### Depth
`RLM(model, depth=N)` (paper's `RLM(model, depth=N)`):
- `depth=0` — `rlm_query` ≡ one `llm_query` (no nested loop).
- `depth=1` — sub-calls can be RLMs (paper's default).
- `depth>1` — each level spawns child RLMs at `depth-1`; iterations are capped per
  level so total work stays bounded.

### Model-agnostic by construction
The only model-specific code is `providers/` (a `complete(messages) -> str`
adapter). The same `core`/`repl` runs on:
- `anthropic` (Claude Messages API)
- `openai` (ChatGPT) and `deepseek` (OpenAI-compatible `base_url`)
- `claude_code` — `claude -p` as the model M (the "RLM for Claude Code" path)
- `mock` — deterministic, for tests.

The paper uses a strong root model + a cheaper sub-model (GPT-5 / GPT-5-mini); we
expose that via `RLM(provider=root, sub_provider=cheaper)`.

## 4. Deliberate simplifications vs. the paper
- **No real sandbox.** We exec model code in an isolated namespace, not a
  container. The paper notes sandboxed REPLs as future work; production use must
  add isolation. (README "Security".)
- **No fine-tuning.** RLM-Qwen3-8B (Appendix A) is a training contribution; this
  repo is the inference scaffold only.
- **stdout truncation** keeps a head+tail with an elision marker (paper: "short
  prefix and length"); the exact truncation policy is a tunable (`stdout_max_chars`).
- **In-context decomposition examples** in the root prompt are kept (the paper
  finds they improve first-step decomposition); the Qwen batching caveat (1b) is
  noted but not auto-applied.

## 5. Verification
`pytest` (deterministic, no API): metadata sizing, REPL exec/persist/error-feedback,
FINAL/FINAL_VAR parsing (incl. ignoring markers inside code), the full loop
(direct final, sub-call + `FINAL_VAR`, `rlm_query` depth-0 fallback, depth-1
recursion spawning a child, iteration cap). Live provider adapters have opt-in
smoke tests gated on the relevant `*_API_KEY`.
