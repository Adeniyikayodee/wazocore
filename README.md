![WazoCore: faster in your own language](assets/banner.svg)

![The same passage streamed by N-ATLaS and by WazoCore in Yoruba, Hausa and Igbo](assets/demo.gif)

<sub>The same FLORES-200 passage in Yoruba, Hausa and Igbo. Replay of real tokenisations, streamed at the decode speed measured in llama.cpp on an Apple M4 CPU (8 tokens per second), shown at 2× speed. Text colour alternates at each token boundary.</sub>

A version of N-ATLaS, Nigeria's open language model, that reads and writes Yoruba, Hausa and Igbo in 1.4 to 1.6 times fewer tokens. It drops into llama.cpp, LM Studio and any OpenAI-compatible stack as a standard GGUF, and generates these languages 1.6 to 1.7 times faster on an ordinary laptop. Two defects remain and are documented: about one reply in a thousand contains a broken character, and Yoruba replies fall into a repetition loop slightly more often.

- **Faster replies**: the same answer is written in 58 to 62% of the time, because each token now carries more of a word. Decode speed in tokens per second is about the same for both models, so the gain comes from the token count and not from the machine.
- **Lower serving cost**: a reply costs a third fewer tokens. A 5,000-word health leaflet written in Yoruba falls from about 14,350 tokens to 9,550.
- **More in context**: N-ATLaS's 8,092-token window holds 8.8 pages of Yoruba instead of 5.6.
- **Drop-in**: same file format, same chat template, same API. All 128,256 original tokens keep their ids and their exact embeddings, and training is anchored to the original model's predictions.

![WazoCore against N-ATLaS: speed-up, token reduction and quality](assets/results.gif)

WazoCore continues N-ATLaS's byte-level BPE with 15,000 merges learned jointly on Yoruba, Hausa and Igbo, rejecting any merge that would split a UTF-8 character. New input embeddings start as the mean of their original pieces, rescaled to the pieces' norm, and output embeddings as the length-weighted mean. Training updates only the new rows and the outer four of 32 transformer layers, for 12M tokens, on N-ATLaS's own translations and on professional human translations, with a KL anchor to the frozen original on the same text and a gradient mask that leaves every original row exact. It takes about 85 minutes on one A100.

## Benchmarks

| WazoCore against N-ATLaS | Yoruba | Hausa | Igbo |
|---|---|---|---|
| Tokens per word | 2.87 → 1.91 | 2.12 → 1.34 | 2.53 → 1.56 |
| Speed-up in words per second, laptop CPU | 1.63× | 1.61× | 1.72× |
| Speed-up for the whole request, laptop CPU | 1.28× | 1.16× | 1.38× |
| Speed-up in words per second, laptop GPU | 1.66× | 1.49× | 1.54× |
| Speed-up for the whole request, laptop GPU | 1.42× | 1.36× | 1.56× |
| chrF++ change, full precision | +2.2 (+1.5, +2.9) | −0.4 (−0.8, +0.1) | +0.2 (−0.4, +0.8) |
| chrF++ change, shipped Q4_K_M file | +1.7 (+0.3, +3.1) | −0.3 (−1.1, +0.6) | −0.1 (−1.4, +1.2) |
| chrF++ change, same training data and no new tokens | +2.3 (+1.6, +3.1) | +0.6 (+0.1, +1.1) | +1.2 (+0.6, +1.8) |

Speed is a paired benchmark in llama.cpp (Q4_K_M, Apple M4, 20 translations per language, both models answering each sentence in turn); the whole request includes reading the prompt, with no prompt cache. Quality is chrF++ against human references with 95% paired-bootstrap intervals: pooled over NTREX-128 and FLORES-200 and two training runs at full precision, and on 250 NTREX-128 sentences for the shipped file. No test sentence is trained on.

The protocol, every table and the command behind each number are in [docs/METHOD.md](docs/METHOD.md); earlier versions and the experiments we rejected are in [docs/HISTORY.md](docs/HISTORY.md).

## Get started

Download the GGUF from [Hugging Face](https://huggingface.co/mkbadeniyi/WazoCore-8B-v6-long) and serve it with llama.cpp (or load it in LM Studio and start its server):

```sh
hf download mkbadeniyi/WazoCore-8B-v6-long WazoCore-8B.Q4_K_M.gguf --local-dir .
llama-server -m WazoCore-8B.Q4_K_M.gguf -c 4096 --port 8080
```

Call it with any OpenAI client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="local")
reply = client.chat.completions.create(
    model="wazocore",
    messages=[{"role": "user", "content": "Translate into Hausa: The clinic opens at eight in the morning."}],
)
print(reply.choices[0].message.content)
# Asibitin yana bude daga karfe takwas na safe.
```

It runs offline on a laptop with 8 GB of memory or more.

## Guides

- [Method and results](docs/METHOD.md): the release recipe, the evaluation protocol, every result, the known defects and the limits of the evidence.
- [History](docs/HISTORY.md): the previous release and what was wrong with it, the version history, and the approaches we rejected.
- [Training on a cloud GPU](docs/cloud-vx.md): the four pipelines, the upload, run times, outputs and known pitfalls.
- [Model card](release/MODEL_CARD.md): intended use, results, limitations and licence terms.
- [Blind A/B tester kit](tester-kit/): the guide, tasks and feedback form for native-speaker evaluation.
- [Contributing](CONTRIBUTING.md): setup, tests, adding a language and engineering standards.

## Build for your language

The toolkit that produced WazoCore reproduces it end to end, and adapts to a new language by changing two tables and the input text.

| | Use WazoCore | Build your own |
|---|---|---|
| What you get | Yoruba, Hausa and Igbo, ready to serve | An expanded N-ATLaS for your languages or domain |
| What you need | llama.cpp or LM Studio | Python 3.11, uv, N-ATLaS access, one 80 GB GPU |
| Time | Minutes | A laptop for the tokenizer, about 2 GPU hours for training and evaluation |
| Evidence | This repository's results | The same benchmarks, run on your model |

```sh
git clone https://github.com/Adeniyikayodee/wazocore.git && cd wazocore
uv run --project wazocore --group ref pytest wazocore/tests
uv run --project wazocore --group ref python -m wazocore.vocab_expand build --out data/models/n-atlas-vx2
scripts/cloud_vx_upload.sh HOST PORT        # then run ARMS=long cloud_vx_v6.sh on the GPU host
scripts/paired_vx.sh LABEL 4 20 0 path/to/model.gguf
```

## Roadmap

- **Remove the two known defects**: Yoruba repetition loops and the rare broken character, then lift Hausa above the original ([METHOD.md](docs/METHOD.md), sections 5 and 6).
- **More languages**: Nigerian Pidgin and Fulfulde next, then other African languages that pay the same token tax.
- **Distribution**: Ollama and the LM Studio catalogue, and a public technical report.
- **Upstream**: future N-ATLaS releases that ship with an expanded vocabulary, so every developer gets the speed-up by default.

## Citation

```bibtex
@misc{adeniyi2026wazocore,
  title        = {WazoCore: In-Place Vocabulary Expansion for Yoruba, Hausa and Igbo on N-ATLaS},
  author       = {Adeniyi, Kayode},
  year         = {2026},
  howpublished = {\url{https://github.com/Adeniyikayodee/wazocore}}
}
```

## Licence and acknowledgements

The code in this repository is released under the [Apache License 2.0](LICENSE). The files under `results/` reproduce human reference translations from FLORES-200 and NTREX-128, both released under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); that text keeps its own licence and is not covered by the Apache licence. Model weights derived from N-ATLaS are released under the N-ATLaS Open-Source Research and Innovation License, which requires attribution, the same licence for derivatives, and a name ending in "Powered by Awarri."; see the [model card](release/MODEL_CARD.md).

> N-ATLaS is an initiative of the Federal Ministry of Communications, Innovation and Digital Economy, and powered by Awarri Technologies.

Built on N-ATLaS by Awarri Technologies and the Federal Ministry of Communications, Innovation and Digital Economy; FLORES-200 and NTREX-128; Google's SMOL translations; Wikipedia in Yoruba, Hausa and Igbo; llama.cpp; and published work on vocabulary expansion (arXiv 2607.15232), low-budget adaptation (arXiv 2406.11477) and embedding initialisation (arXiv 2608.03494).
