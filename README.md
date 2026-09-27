# WazoCore

**In-place vocabulary expansion for Yoruba, Hausa and Igbo on N-ATLaS**

[Method](docs/METHOD.md) · [Model card](release/MODEL_CARD.md) · [Results](results/vocab/) · [Citation](#citation)

Large language models process text as tokens, and most vocabularies are built
around English. [N-ATLaS](https://huggingface.co/NCAIR1/N-ATLaS), Nigeria's
open multilingual model, inherits Llama-3's vocabulary. It needs 2.0–2.7× as
many tokens to express the same sentence in Hausa, Igbo or Yoruba as in English,
so generation in these languages is slower and more expensive.

WazoCore adds 15,000 tokens learned from Yoruba, Hausa and Igbo text to N-ATLaS
while keeping every original token unchanged, and then trains the model to use
them. The resulting model generates 1.55–1.64× more words per second in these
languages on unmodified inference engines. Translation quality is statistically
unchanged for Yoruba and Hausa, and 1.3 chrF++ lower for Igbo. To our knowledge,
this is the first vocabulary expansion of N-ATLaS or of any public Llama-3 model
for these languages.

## Results

The comparison is WazoCore against the original N-ATLaS on FLORES-200 devtest,
which was held out from all training.

| Language | Tokens per word | Words/s speed-up | chrF++ Δ (95% CI) |
|---|---|---|---|
| Yoruba | 2.92 → 1.99 | 1.55× | +1.0 (−0.1, +2.1) |
| Hausa | 2.10 → 1.43 | 1.64× | −1.1 (−2.5, +0.2) |
| Igbo | 2.54 → 1.58 | 1.60× | −1.3 (−2.2, −0.3) |
| English | 1.28 → 1.31 | 0.96× | — |

How the numbers were measured:

- **Speed:** llama.cpp (build 9430), Q4_K_M, Apple M4 CPU with 4 threads, and
  20 translations per language. The two models answered each sentence in turn,
  with the first-mover alternating. Raw decoding speed (tokens/s) is unchanged,
  so the gain comes entirely from fewer tokens per word.
- **Quality:** greedy English → X translation of 200 sentences per language.
  The confidence intervals come from a paired bootstrap with 1,000 resamples.
- **Tokenisation:** llama.cpp's tokenisation matches the reference tokenizer on
  all 160 test sentences.

The Yoruba speed figure excludes one sample in which the host stalled for 223 s
(0.59× with it included). All raw measurements are in
[`results/vocab/`](results/vocab/).

## Approach

1. **Vocabulary.** We continue byte-pair encoding from the original tokenizer
   and learn 15,000 merges jointly on Yoruba, Hausa and Igbo text. Merges that
   would split a UTF-8 character are rejected, and all original token ids are
   preserved.
2. **Initialisation.** A new token's input embedding is the mean of its
   constituent pieces' embeddings. Its output embedding is the byte-length-
   weighted mean of theirs.
3. **Training.** We train the new embedding rows and the first and last two
   transformer layers on 5M tokens:
   - 85% is the original model's own translations of about 30,000 requests
     (self-distillation), re-tokenised with the new vocabulary;
   - 15% is orthographically filtered Wikipedia text;
   - a KL term anchors the model to the frozen original on the same text under
     the original tokenisation.

   Gradients to the original rows are masked, so those rows never change.
4. **Evaluation.** Quality is measured by chrF++ with bootstrap intervals, and
   speed by paired A/B runs. A control run trains the original model with the
   identical recipe but no new tokens. It separates the effect of the
   vocabulary change from the effect of training.

[`docs/METHOD.md`](docs/METHOD.md) describes the method in full, including the
approaches that did not work and why.

## Getting started

Requirements: Python 3.11+, [uv](https://docs.astral.sh/uv/), and
[llama.cpp](https://github.com/ggml-org/llama.cpp) for inference and
benchmarks.

```sh
git clone https://github.com/Adeniyikayodee/wazocore.git
cd wazocore
uv run --project wazocore --group ref pytest wazocore/tests
```

To run a WazoCore GGUF with llama.cpp:

```sh
llama-server -m WazoCore-8B.Q4_K_M.gguf -c 2048 --port 8080
curl -s localhost:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Translate into Hausa: The clinic opens at eight in the morning."}]}'
```

The weights have not been released yet. The model card in
[`release/`](release/MODEL_CARD.md) describes the planned release.

## Reproducing the results

Reproduction needs access to N-ATLaS on Hugging Face and one 80 GB GPU
(A100 or H100). The full pipeline takes about three GPU hours.

1. **Build the tokenizer** (CPU, minutes):

   ```sh
   uv run --project wazocore --group ref python -m wazocore.vocab_expand build --out data/models/n-atlas-vx2
   ```

2. **Train, evaluate and export** (GPU). `scripts/cloud_vx_pod.sh` runs these
   steps in order:
   - expansion;
   - self-distillation, when cached replies are absent;
   - training;
   - the 200-sentence quality gate;
   - GGUF export;
   - the control run.

   [`docs/cloud-vx.md`](docs/cloud-vx.md) covers pod setup.

   ```sh
   scripts/cloud_vx_upload.sh HOST PORT     # from your machine
   bash /workspace/cloud_vx_pod.sh           # on the GPU host
   ```

3. **Benchmark speed** against the original model (CPU):

   ```sh
   scripts/paired_vx.sh LABEL 4 20 0 path/to/WazoCore.gguf
   ```

To extend the method to another language, change the language tables (`CODES`,
`NAMES`) in `vocab_expand.py` and `vx_torch.py`, provide Wikipedia and FLORES-200
text for it, and rerun the three steps.

## Repository structure

| Path | Contents |
|---|---|
| `wazocore/wazocore/vocab_expand.py` | BPE continuation on an existing tokenizer |
| `wazocore/wazocore/vx_core.py` | Initialisation, data mixing and schedule (pure functions) |
| `wazocore/wazocore/vx_torch.py` | Expansion, distillation, training, evaluation and scoring |
| `wazocore/wazocore/paired_vx.py` | Paired words/s benchmark on llama.cpp |
| `wazocore/wazocore/tokens_per_word.py` | Tokenizer audit on FLORES-200 |
| `scripts/` | GPU pipeline and benchmark launchers |
| `results/` | Raw measurements behind every reported number |
| `docs/` | Method, findings and GPU runbook |

## Limitations

- **Igbo quality.** Igbo translation quality is 1.3 chrF++ below the original,
  a statistically significant gap. The control run attributes about a third of
  it to the training recipe and the rest to the vocabulary change.
- **Hausa spelling.** The training recipe reduces the use of Hausa hooked
  letters (ƙ ɗ ɓ): 0.29 per 100 characters against 0.39 for the original.
- **Scope of the evaluation.** Quality was measured on English → X translation
  in one domain (FLORES-200). Dialogue, reasoning and X → English were not
  scored.
- **Hardware coverage.** Speed was measured on one CPU. GPU inference has only
  been smoke-tested (about 21 tokens/s with Metal on an M4).
- **Inherited behaviour.** WazoCore keeps N-ATLaS's biases, its 8,092-token
  context and the limits of its instruction tuning.

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
Research and Innovation License. It requires attribution, the same licence for
derivatives, and a name ending in "Powered by Awarri."; see the
[model card](release/MODEL_CARD.md).

> N-ATLaS is an initiative of the Federal Ministry of Communications,
> Innovation and Digital Economy, and powered by Awarri Technologies.

## Acknowledgements

This work builds on:
- **N-ATLaS** by Awarri Technologies and the Federal Ministry of Communications,
  Innovation and Digital Economy;
- the **FLORES-200** benchmark (NLLB Team, Meta AI);
- **Wikipedia** in Yoruba, Hausa and Igbo;
- **llama.cpp**;
- published work on vocabulary expansion (arXiv 2607.15232), low-budget
  adaptation (arXiv 2406.11477) and subword-composition initialisation
  (arXiv 2608.03494).
