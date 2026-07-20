# RLM-Qwen3-8B — Native-Recursion Fine-Tuning Recipe (reference notes)

> **Documentation only — NOT implemented in this repo.** `/opt/rlm` is the
> inference scaffold (Algorithm 1); it does no training. This file distills
> **Appendix A** of *Recursive Language Models* (Zhang, Kraska, Khattab, MIT CSAIL,
> [arXiv:2512.24601v3](https://arxiv.org/abs/2512.24601); authors' code
> [github.com/alexzhang13/rlm](https://github.com/alexzhang13/rlm)) so the recipe is
> captured for reference. Every number below is **the paper's**; fields the paper
> does not state are marked *not stated* and are NOT guessed here. Verify against the
> paper / authors' code before any real training run.

## Why
The default RLM uses a **frontier** root model (paper: GPT-5). Appendix A shows a
small model can be **post-trained to be natively recursive** — to drive the RLM loop
itself — recovering much of the benefit without a frontier root. Reported result:
**RLM-Qwen3-8B beats its base Qwen3-8B by a median +28.3%** across the four evaluation
tasks, and approaches vanilla GPT-5 on three long-context tasks.

## Models
| Role | Model |
|------|-------|
| Base (fine-tuned) | **Qwen3-8B** |
| Teacher (trajectory generator) | **Qwen3-Coder-480B-A35B-Instruct** |

## 1. Trajectory generation (the training data)
1. Run `RLM(Qwen3-Coder-480B-A35B)` — the teacher as the **root** model in the
   Algorithm-1 loop — over **750 English LongBenchPro tasks**.
2. Collect **2,250 candidate trajectories** (≈3 per task).
3. Filter → **1,072 trajectories**: drop zero-scoring and single-turn trajectories.
4. Drop turns whose history exceeds Qwen3-8B's context (**≈100k characters**).
5. **Programmatic correction** of template mistakes in RLM usage: ~16% had an
   incorrect `FINAL` answer, ~13% had incorrect variable references — fixed
   programmatically rather than discarded.

## 2. Training-data format
- **One SFT sample per root RLM turn (iteration):**
  - **input** = the full history at that turn (metadata + prior code + truncated stdout),
  - **target** = the root LLM's output at that step (the ` ```repl ` code block / `FINAL`).
- A *k*-turn trajectory therefore yields *k* samples. This is exactly the
  `(history → next root action)` shape the loop already produces at inference.

## 3. Training
| Field | Value (paper) |
|-------|---------------|
| Objective | Supervised fine-tuning (SFT) |
| Loss masking (target-only?) | *not stated* |
| Batch size | **64** |
| Training steps | **300** |
| Compute | **48 H100-hours** |
| Learning rate | *not stated* |
| Optimizer | *not stated* |
| Sequence length | *not stated* (samples pre-filtered to ≤ ≈100k chars) |

## 4. Reproducing the data step from this repo
`/opt/rlm` is already the **data-generation harness** for §1 — it does not train, but
it produces exactly the trajectories the recipe needs:
- `RLM(get_provider(<backend>, model=<teacher>), depth=1).run(task_context, query)`
  drives the Algorithm-1 loop (`rlm/core.py` `_run`).
- Each root turn's `(history, output)` in that loop **is** the SFT sample of §2.
- To build a training set you would add, around the existing loop: a per-turn
  trajectory logger, the score / turn-count / context-length filters (§1.3–1.4), the
  programmatic corrector (§1.5), then hand the samples to any SFT trainer with the
  paper's batch/steps (§3). **Training itself stays out of scope for this repo.**

## Not reproduced / open (consult the paper's Appendix A)
Learning rate, optimizer, sequence length, and the exact loss-masking policy are not
stated in the captured paper text — do not assume values; consult the paper or the
authors' code (github.com/alexzhang13/rlm) before any real run.
