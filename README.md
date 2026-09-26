# WazoCore: N-ATLaS at the speed of English

Tools and evidence for **in-place vocabulary expansion** of
[N-ATLaS](https://huggingface.co/NCAIR1/N-ATLaS), Nigeria's open multilingual
LLM. The expanded model, **WazoCore Powered by Awarri.**, writes Yoruba, Hausa and Igbo in about 1.5× fewer
tokens, so replies come about 1.5× faster in llama.cpp, Ollama, LM Studio or any
other engine, with no engine changes. The whole method runs on one laptop plus
about 2 rented GPU hours.

| | tokens per word, original → expanded | words/s speed-up (llama.cpp, CPU) | chrF++ vs original (100 sentences) |
|---|---|---|---|
| Yoruba | 2.92 → 2.55 | 1.19× (1.31× by characters) | +0.8 (not significant) |
| Hausa | 2.10 → 1.41 | 1.50× | −1.1 (not significant) |
| Igbo | 2.54 → 1.62 | 1.63× | −1.9 (significant, about 5%) |
| English | 1.28 → 1.30 | 0.98× | not affected by design |

Model card: `release/MODEL_CARD.md`. Method, what failed on the way, and
results: `docs/METHOD.md`.

## Why this matters

Llama-3's vocabulary was built mostly for English. A Yoruba word costs 2–3×
more tokens than an English word, and every token costs time and money. This
"token tax" makes N-ATLaS slower and more expensive in exactly the languages it
exists for. Nobody had expanded the vocabulary of N-ATLaS, or of any public
Yoruba, Hausa or Igbo Llama-3 model, before this work.

## Reproduce

Requirements: Python 3.11 with [uv](https://docs.astral.sh/uv/), llama.cpp on
`PATH`, a Hugging Face read token with N-ATLaS access, and one rented 80 GB GPU
(A100/H100).

**1. Data and tokenizer (laptop, minutes).** Place the Wikipedia text in
`data/wiki/{en,yo,ha,ig}.txt`, FLORES-200 in `data/flores200_dataset`, and the
N-ATLaS bf16 files in `data/models/n-atlas-bf16`. Then:

```sh
uv run --project wazocore --group ref python -m wazocore.vocab_expand build --out data/models/n-atlas-vx2
```

**2. Train, evaluate and export (GPU, about 2 hours).** See `docs/cloud-vx.md`.
In short:

```sh
scripts/cloud_vx_upload.sh HOST PORT            # from the laptop
bash /workspace/cloud_vx_pod.sh                  # on the pod, after hf auth login
```

The pod script runs the v3 recipe: it builds the expanded model, trains it
(60% the original model's own replies, 40% Wikipedia), measures BPC and the
100-sentence chrF++ gate with bootstrap intervals, and exports a Q4_K_M GGUF.
If `data/runs/cloud/chat-*.jsonl` exist, the upload reuses them; otherwise the
pod regenerates the distilled replies (about 45 extra minutes).

**3. Check and benchmark (laptop, about 1 hour).**

```sh
scripts/paired_vx.sh m4-cpu 4 20 0 path/to/expanded.gguf
```

This gives paired words/s against the original, with one model in memory at a
time and the first-mover alternating.

**Another language?** Change the language codes in `vocab_expand` and
`vx_torch` (`CODES`, `NAMES`), supply that language's Wikipedia and FLORES
text, and rerun steps 1–3.

## Layout

| Path | What it is |
|---|---|
| `wazocore/wazocore/vocab_expand.py` | BPE continuation on top of the original tokenizer (pure core, tested) |
| `wazocore/wazocore/vx_core.py` | Pure helpers: new-row initialisation, data windows, schedule |
| `wazocore/wazocore/vx_torch.py` | Expansion, training, distillation, BPC, sampling and scoring on a CUDA GPU |
| `wazocore/wazocore/paired_vx.py` | Paired llama.cpp words/s benchmark |
| `wazocore/wazocore/tokens_per_word.py` | Token-tax audit of a tokenizer on FLORES |
| `scripts/` | Cloud upload, the pod pipeline, benchmark launchers |
| `results/vocab/`, `results/tokenizer/` | Every measurement behind the tables above |

Tests: `uv run --project wazocore --group ref pytest wazocore/tests`.

## Licence and attribution

The code is in this repository. Model weights derived from N-ATLaS are covered
by the N-ATLaS Open-Source Research and Innovation License; see the model card.

> N-ATLaS is an initiative of the Federal Ministry of Communications,
> Innovation and Digital Economy, and powered by Awarri Technologies.

Contributions follow `CONTRIBUTING.md`: a functional core, errors as values, and
atomic commits that explain why.
