# WazoCore

**Nigerian languages at the speed of English.**

WazoCore is N-ATLaS, Nigeria's open language model, rebuilt to read and write
Yoruba, Hausa and Igbo in far fewer tokens. It generates these languages
**1.55 to 1.64 times faster** on an ordinary laptop, in the engines developers
already use, with no code changes. It ships with the open toolkit that built
it, so the next language takes a laptop and about three hours on a rented GPU.

[Method](docs/METHOD.md) · [Model card](release/MODEL_CARD.md) · [Results](results/vocab/) · [Citation](#citation)

## The problem: a token tax on Nigerian languages

Language models read and write in tokens, and most vocabularies were built for
English. N-ATLaS inherits Llama-3's vocabulary, so the same sentence costs far
more tokens in Nigerian languages (FLORES-200):

| Language | Tokens per word | Cost vs English |
|---|---|---|
| English | 1.24 | 1.0× |
| Hausa | 2.13 | 2.0× |
| Igbo | 2.62 | 2.4× |
| Yoruba | 2.91 | 2.7× |

Every product built on N-ATLaS pays this tax: replies arrive slower, serving
costs more, and less text fits in the context window. The users who pay most
are the ones N-ATLaS was built for.

## What WazoCore delivers

Measured against the original N-ATLaS on test sets never used in development:

| Language | Words per second | Tokens per character | Translation quality (chrF++) |
|---|---|---|---|
| Yoruba | **1.55× faster** | 1.5× fewer | equal to slightly better |
| Hausa | **1.64× faster** | 1.4× fewer | equal (within noise) |
| Igbo | **1.60× faster** | 1.6× fewer | about 1 point lower (2 to 3%) |
| English | unchanged | unchanged | unchanged |

In practical terms, a 5,000-word health leaflet in Yoruba drops from about
14,500 tokens to 9,200. That takes a laptop's generation time from 27 minutes
to 17, and lets the model read 8.8 pages at once instead of 5.6.

## How it works

1. **A bigger vocabulary, without losing the old one.** We continue N-ATLaS's
   own BPE tokenizer with 15,000 merges learned from Yoruba, Hausa and Igbo
   text. Every original token keeps its id, so English and everything the
   model already knows stay intact. Merges that would split a character are
   rejected.
2. **New tokens start where the model's knowledge already is.** Each new
   embedding is built from the embeddings of the pieces it replaces.
3. **The model teaches its own replacement.** Most of the training data is
   N-ATLaS's own answers to about 30,000 translation requests, rewritten in
   the new vocabulary. A KL anchor holds the model's predictions close to the
   original, and a gradient mask keeps every original vocabulary row
   untouched. Only the new rows and 4 of the 32 layers train, in about 40
   minutes on one A100.
4. **Everything is measured, including what failed.** Quality is scored with
   chrF++ and bootstrap confidence intervals on two test sets and two training
   runs, and speed with a paired benchmark in llama.cpp. A control run
   separates the effect of the new vocabulary from the effect of training.

The full method, every experiment and the ones we rejected are in
[`docs/METHOD.md`](docs/METHOD.md).

## Quick start

WazoCore is a standard Llama-3 GGUF. Serve it with llama.cpp or LM Studio and
call it through the OpenAI-compatible API:

```sh
llama-server -m WazoCore-8B.Q4_K_M.gguf -c 4096 --port 8080
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="local")
reply = client.chat.completions.create(
    model="wazocore",
    messages=[{"role": "user", "content": "Translate into Hausa: The clinic opens at eight in the morning."}],
)
print(reply.choices[0].message.content)
```

It runs offline on a laptop with 8 GB of memory or more.

## Build your own

The toolkit reproduces WazoCore end to end, and adapts to other languages by
changing the language tables (`CODES`, `NAMES`) and the input text.

Requirements: Python 3.11+, [uv](https://docs.astral.sh/uv/),
[llama.cpp](https://github.com/ggml-org/llama.cpp), access to N-ATLaS on
Hugging Face, and one 80 GB GPU (A100 or H100).

```sh
git clone https://github.com/Adeniyikayodee/wazocore.git && cd wazocore
uv run --project wazocore --group ref pytest wazocore/tests          # 36 tests

# 1. Expanded tokenizer (laptop, minutes)
uv run --project wazocore --group ref python -m wazocore.vocab_expand build --out data/models/n-atlas-vx2

# 2. Train, evaluate and export (GPU, about 3 hours; see docs/cloud-vx.md)
scripts/cloud_vx_upload.sh HOST PORT
bash /workspace/cloud_vx_pod.sh

# 3. Paired speed benchmark against the original (laptop)
scripts/paired_vx.sh LABEL 4 20 0 path/to/WazoCore.gguf
```

## Roadmap

- **Close the Igbo gap.** The control run located the cause in the new
  vocabulary. The next training objective scores only the new-token
  segmentation, so quality can rise without giving back speed.
- **More languages.** Nigerian Pidgin and Fulfulde next, then other African
  languages that pay the same tax. Each model is about three GPU hours.
- **Distribution.** Listings in Ollama and the LM Studio catalogue, and a
  public technical report.
- **Upstream.** Working with the N-ATLaS team so future releases ship with an
  expanded vocabulary and every developer gets the speed-up by default.

## Repository

| Path | Contents |
|---|---|
| `wazocore/wazocore/vocab_expand.py` | Tokenizer expansion (BPE continuation) |
| `wazocore/wazocore/vx_core.py` | Initialisation, data mixing, chunk alignment |
| `wazocore/wazocore/vx_torch.py` | Expansion, distillation, training, evaluation |
| `wazocore/wazocore/paired_vx.py` | Paired speed benchmark on llama.cpp |
| `wazocore/wazocore/tokens_per_word.py` | Tokenizer audit on FLORES-200 |
| `scripts/` | GPU pipelines and benchmark launchers |
| `results/` | Raw measurements behind every number above |
| `docs/` | Method, validation and GPU runbook |
| `tester-kit/` | Blind A/B test for native speakers |

## Citation

```bibtex
@misc{adeniyi2026wazocore,
  title  = {WazoCore: In-Place Vocabulary Expansion for Yoruba, Hausa and Igbo on N-ATLaS},
  author = {Adeniyi, Kayode},
  year   = {2026},
  howpublished = {\url{https://github.com/Adeniyikayodee/wazocore}}
}
```

## License

Model weights derived from N-ATLaS are released under the N-ATLaS Open-Source
Research and Innovation License, which requires attribution, the same licence
for derivatives, and a name ending in "Powered by Awarri."; see the
[model card](release/MODEL_CARD.md).

> N-ATLaS is an initiative of the Federal Ministry of Communications,
> Innovation and Digital Economy, and powered by Awarri Technologies.

## Acknowledgements

Built on N-ATLaS by Awarri Technologies and the Federal Ministry of
Communications, Innovation and Digital Economy; the FLORES-200 and NTREX-128
benchmarks; Wikipedia in Yoruba, Hausa and Igbo; llama.cpp; and published work
on vocabulary expansion (arXiv 2607.15232), low-budget adaptation (arXiv
2406.11477) and embedding initialisation (arXiv 2608.03494).
