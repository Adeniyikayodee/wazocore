# Contributing to WazoCore

Thank you for helping make Nigerian languages as fast to serve as English.
Contributions of every size are welcome: a new language, a better training
objective, an evaluation, a runtime integration, or a clearer sentence in the
docs.

## Getting set up

Requirements: Python 3.11+, [uv](https://docs.astral.sh/uv/) and, for
benchmarks, [llama.cpp](https://github.com/ggml-org/llama.cpp).

```sh
git clone https://github.com/Adeniyikayodee/wazocore.git && cd wazocore
uv run --project wazocore --group ref pytest wazocore/tests
```

The tests run on a CPU in seconds, using a tiny model in
`wazocore/tests/fixtures/`. Training and evaluation of the full model need a
GPU; see [`docs/cloud-vx.md`](docs/cloud-vx.md).

## Where things live

| Path | Role |
|---|---|
| `wazocore/wazocore/vx_core.py` | Pure functions: initialisation, data mixing, chunk alignment, filters |
| `wazocore/wazocore/vocab_expand.py` | Tokenizer expansion |
| `wazocore/wazocore/vx_torch.py` | Everything that touches a model or a GPU |
| `wazocore/wazocore/paired_vx.py` | Paired speed benchmark |
| `scripts/` | GPU pipelines |
| `docs/METHOD.md` | The method and every reported result |

## Adding a language

1. Add its codes to `CODES` and `NAMES` in `vx_torch.py` and to the build step
   in `vocab_expand.py`.
2. Provide Wikipedia text in `data/wiki/<code>.txt` and its FLORES-200 files.
3. If the language uses special letters, add a spelling filter rule in
   `vx_core.careful_line` so badly spelled training text is excluded.
4. Build the tokenizer, run `scripts/cloud_vx_pod.sh`, and report the results
   the same way as `docs/METHOD.md`: speed with the paired benchmark, quality
   with bootstrap intervals, on a test set not used during development.

## Before you open a pull request

- The tests pass, and new logic in `vx_core.py` comes with a test.
- Claims about speed or quality include the command and the raw results that
  produced them. A result that did not work is worth documenting too.
- Commits follow the standards below.

## Engineering principles

The goal is predictable, testable, modular code: functional programming as a
discipline for managing state and side effects, not as a syntax.

1. **Write pure functions by default.** A function computes its output only
   from its inputs: no global state, no hidden operations. The same inputs
   always give the same output.
2. **Enforce immutability.** Data does not change in place. When state must
   change, compute and return a new copy.
3. **Push side effects to the edges.** The core is pure logic. File access,
   network calls, GPU work and logging live in an imperative shell around it
   (here, `vx_core.py` is the core and `vx_torch.py` the shell).
4. **Treat errors as values.** Do not use exceptions for control flow. Make
   failure explicit in the return type so the caller must handle it; in
   Python, return `None` or a result value for expected failures (for example,
   `chunk_align` returns `None` when two tokenisations do not match) and reserve
   exceptions for programming errors.
5. **Separate data from execution.** Plain, immutable data structures for
   state; separate modules of pure functions that transform them.
6. **Design for composition.** Small, single-purpose functions whose outputs
   feed cleanly into the next.

## Commit standards

Commit messages are the permanent record of engineering decisions.

1. **Explain the why, not the what.** The diff shows what changed; the message
   explains the technical or product reason.
2. **Keep commits atomic.** One logical change per commit.
3. **No external metadata.** Never include links to AI chat sessions, prompt
   logs or personal tracking links.
4. **Format.** `scope(area): action`, under 50 characters on the first line.
   Examples from this repository:
   - `fix(trace): reject merges that split a character`
   - `feat(trace): add on-policy distillation stage`
   - `refactor(scripts): share one GGUF export script`
   - `docs(method): record v5 and why it was rejected`
