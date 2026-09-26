---
license: other
license_name: n-atlas-open-source-research-and-innovation-license
base_model: NCAIR1/N-ATLaS
language: [yo, ha, ig, en]
tags: [llama-3, vocabulary-expansion, yoruba, hausa, igbo, gguf]
---

<!-- The N-ATLaS licence requires renamed derivatives to end with "Powered by Awarri." -->
# WazoCore Powered by Awarri.

WazoCore is N-ATLaS with 15,000 new Yoruba, Hausa and Igbo tokens. It writes the same
languages in about **1.5× fewer tokens**, so replies arrive about 1.5× faster
on any engine, with no change to the runtime.

> N-ATLaS is an initiative of the Federal Ministry of Communications,
> Innovation and Digital Economy, and powered by Awarri Technologies.

## Why

N-ATLaS reads and writes in tokens. Its vocabulary comes from Llama-3, which was
built mostly for English, so a Yoruba word costs about 2.9 tokens, while an
English word costs about 1.3. Every Nigerian-language reply therefore takes 2–3×
longer to generate and costs more to serve. This model adds whole-word tokens
for the three languages and keeps every original token, so English is
unaffected.

## Results

All numbers compare this model with the original N-ATLaS on the same FLORES-200
devtest sentences, which were never used in training.

**Speed.** Measured in llama.cpp (build 9430) on an M4 MacBook Air, CPU,
4 threads, Q4_K_M. There were 20 translations per language, and the two models
answered each sentence back to back in alternating order.

| | words/s original → this | speed-up (words) | speed-up (characters) | tokens per word |
|---|---|---|---|---|
| Yoruba* | 2.97 → 4.62 | 1.55× | 1.51× | 2.92 → 1.99 |
| Hausa | 3.99 → 6.54 | 1.64× | 1.65× | 2.10 → 1.43 |
| Igbo | 3.64 → 5.83 | 1.60× | 1.61× | 2.54 → 1.58 |
| English | 7.43 → 7.14 | 0.96× | 0.98× | 1.28 → 1.31 |

\*19 sentences. One Yoruba request stalled for 223 s because of the laptop
(0.3 tokens/s against 7–13 for every other request); with it included, the
Yoruba figure is 0.59×. Raw data: `results/vocab/m4-cpu-v4-paired.json`.

Raw decoding speed (tokens/s) is the same for both models, so the gain comes
entirely from needing fewer tokens.

**Quality.** Greedy English → X translations, 200 sentences per language. The
score is chrF++ against the human FLORES references. The range is a 95%
paired-bootstrap interval for the difference.

| | chrF++ original | chrF++ this model | difference (95% range) | replies that ended properly |
|---|---|---|---|---|
| Yoruba | 25.81 | 26.84 | +1.0 (−0.1 to +2.1) | 196 (original 193) |
| Hausa | 48.97 | 47.86 | −1.1 (−2.5 to +0.2) | 199 (original 199) |
| Igbo | 38.60 | 37.33 | **−1.3 (−2.2 to −0.3)** | 198 (original 199) |

Yoruba is slightly better than the original, and Hausa is statistically
indistinguishable from it. **Igbo is about 3% lower** on this measure. A
control run shows that about a third of the Igbo gap, and a small drop in
Hausa hooked letters (ƙ ɗ ɓ), come from the training recipe rather than the
new vocabulary.

## Use

The GGUF works with any llama.cpp-based runtime. With llama.cpp:

```sh
llama-server -m WazoCore-8B.Q4_K_M.gguf -c 2048 --port 8080
curl -s localhost:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Translate into Hausa: The clinic opens at eight in the morning."}]}'
```

The full-precision weights load with `transformers` (5.x), in the same way as
N-ATLaS. Ollama and LM Studio should load the GGUF like any Llama-3 model; this
has **not been tested yet**.

## How it was made

1. **Vocabulary.** The original tokenizer is kept, and 15,000 BPE merges are
   learned on top of it from Yoruba, Hausa and Igbo Wikipedia and the FLORES
   dev split, with an equal share per language. Merges that would split a
   character are rejected. Original token ids are unchanged, and the tokenizer
   round-trips all test text exactly.
2. **Initialisation.** Each new token's input row is the mean of its original
   pieces; its output row is the byte-length-weighted mean.
3. **Training.** On one A100 for about 40 minutes, 5M tokens train the new rows
   plus the bottom and top two transformer layers. The data is 15% carefully
   spelled Wikipedia text and 85% the original model's own replies to about
   30,000 translation requests (FLORES dev and Wikipedia, both directions),
   re-tokenised with the new vocabulary. A KL term keeps the model's
   predictions close to the original's on the same replies tokenised the old
   way. The original vocabulary rows receive no updates.
4. **Export.** GGUF from the bf16 weights, then quantised to Q4_K_M.
   llama.cpp's tokenisation matches the reference on all 160 test sentences.

Code, data recipe and every measurement: the WazoCore repository
(`docs/METHOD.md`).

## Limitations

- Igbo translation quality is lower than the original (see above).
- Quality was measured on 200 English → X sentences per language, one domain
  (FLORES). Chat, reasoning and X → English were not scored.
- Speed was measured on one laptop CPU. The Metal/GPU path was only
  smoke-tested (about 21 tokens/s on an M4).
- Everything else about N-ATLaS applies unchanged, including its biases, its
  8K context and its instruction-tuning data.

## Licence

This model is a derivative of N-ATLaS and is released under the same
**N-ATLaS Open-Source Research and Innovation License (Terms of Use v1.0)**.
Key terms:
- the attribution above must be kept;
- renamed derivatives must end in "Powered by Awarri.";
- derivatives must use the same licence;
- at most 1,000 active end-users per rolling 30 days without a commercial
  licence;
- surveillance, discriminatory profiling, disinformation, military use and
  unauthorised personal-data scraping are prohibited.

Read the full terms on the N-ATLaS model page before use.
