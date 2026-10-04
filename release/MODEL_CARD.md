---
license: other
license_name: n-atlas-open-source-research-and-innovation-license
base_model: NCAIR1/N-ATLaS
base_model_relation: finetune
language: [yo, ha, ig, en]
pipeline_tag: text-generation
library_name: transformers
datasets: [google/smol]
metrics: [chrf]
tags:
  - yoruba
  - hausa
  - igbo
  - nigerian-languages
  - african-languages
  - low-resource-languages
  - vocabulary-expansion
  - tokenizer
  - bpe
  - translation
  - n-atlas
  - llama-3
  - llama-cpp
  - gguf
---

<!-- The N-ATLaS licence requires renamed derivatives to end with "Powered by Awarri." -->
# WazoCore Powered by Awarri.

**Faster in your own language.** WazoCore is N-ATLaS with
15,000 new Yoruba, Hausa and Igbo tokens. It writes these languages in 1.4 to
1.6 times fewer tokens, so replies are written 1.6 to 1.7 times faster in
llama.cpp and LM Studio, with no change to your code.

> N-ATLaS is an initiative of the Federal Ministry of Communications,
> Innovation and Digital Economy, and powered by Awarri Technologies.

## Model details

| | |
|---|---|
| Base model | NCAIR1/N-ATLaS (Llama-3 architecture, 8B parameters) |
| What changed | 15,000 added tokens for Yoruba, Hausa and Igbo: the vocabulary grows from 128,256 to 143,256. Every original token keeps its id and its exact embedding |
| Files | A Q4_K_M GGUF (5.0 GB) and the full-precision weights |
| Languages | Yoruba, Hausa, Igbo, English |
| Licence | N-ATLaS Open-Source Research and Innovation License (see Licence) |
| Code and evidence | [github.com/Adeniyikayodee/wazocore](https://github.com/Adeniyikayodee/wazocore) |

N-ATLaS inherits Llama-3's vocabulary, which was built mostly for English. A
Yoruba word costs about 2.9 tokens where an English word costs 1.2, so every
Nigerian-language reply is slower to generate, more expensive to serve, and
fits less context. WazoCore adds word and word-piece tokens for the three
languages, so English text tokenises as before.

## Intended use

Generating and translating Yoruba, Hausa and Igbo text on local hardware with
llama.cpp or LM Studio, in place of N-ATLaS, where the token count in these
languages limits speed, cost or context. Use is governed by the N-ATLaS
licence.

## Out of scope

- Any use the licence prohibits (see Licence).
- Tasks that were not evaluated: dialogue, reasoning, translation into
  English, and English quality (see Limitations).
- Applications that cannot handle a failed request or a looping reply. Both
  occur at the rates under Limitations and need handling in the client.

## How to use

WazoCore is a standard Llama-3 GGUF. It is tested in llama.cpp and LM Studio:

```sh
llama-server -m WazoCore-8B.Q4_K_M.gguf -c 4096 --port 8080
curl -s localhost:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Translate into Hausa: The clinic opens at eight in the morning."}]}'
```

Always pass `-c`. Without it llama-server takes the context size from the
file, which reports 131,072 tokens, and reserves memory for all of it.

The full-precision weights load with `transformers` 5.x, like N-ATLaS. It runs
offline on a laptop with 8 GB of memory or more.

## Training data

| Source | Used for | Amount |
|---|---|---|
| N-ATLaS's own greedy replies to translation requests, English to and from each language, with sources from Wikipedia | Training | About 24,000 requests |
| google/smol, professional translations (CC BY 4.0) | Training | 23,662 sentence pairs: Yoruba 8,647, Hausa 8,653, Igbo 6,362 |
| Yoruba, Hausa and Igbo Wikipedia, filtered for careful spelling | Training and vocabulary | 15% of training tokens |
| FLORES-200 dev split | Vocabulary only | With Wikipedia, 4M characters per language |

No FLORES or NTREX text is used in training.

## Training procedure

1. **Vocabulary.** N-ATLaS's tokenizer is kept, and 15,000 BPE merges are
   learned on top of it from Yoruba, Hausa and Igbo Wikipedia and the FLORES
   dev split, with an equal share per language. Merges that would split a
   character are rejected. Original token ids are unchanged, and the tokenizer
   round-trips all test text exactly.
2. **Initialisation.** Each new token's input row is the mean of its original
   pieces, rescaled to the pieces' norm; its output row is the
   length-weighted mean.
3. **Training.** About 85 minutes on one A100: 12M tokens train the new rows
   and the bottom and top two of the 32 transformer layers. The data is the
   original model's own replies to about 24,000 translation requests,
   23,662 professionally translated sentence pairs (google/smol, CC BY 4.0)
   and carefully spelled Wikipedia text, all tokenised with the new
   vocabulary. A KL term keeps predictions close to the original's on the
   same text, a penalty discourages spelling a word in pieces when it has a
   token of its own, and the original vocabulary rows receive no updates.
   No FLORES or NTREX text is used in training.
4. **Export.** GGUF from the bf16 weights, quantised to Q4_K_M. llama.cpp's
   tokenisation matches the reference tokenizer on 2,000 test texts.


## Evaluation

All numbers compare WazoCore with the original N-ATLaS.

| WazoCore against N-ATLaS | Yoruba | Hausa | Igbo |
|---|---|---|---|
| Tokens per word | 2.87 → 1.91 | 2.12 → 1.34 | 2.53 → 1.56 |
| Speed-up in words per second, laptop CPU | 1.63×* | 1.61× | 1.72× |
| Speed-up for the whole request, laptop CPU | 1.28× | 1.16× | 1.38× |
| Speed-up in words per second, laptop GPU | 1.66×* | 1.49× | 1.54× |
| Speed-up for the whole request, laptop GPU | 1.42× | 1.36× | 1.56× |
| chrF++ change, full precision | +2.2 (+1.5, +2.9) | −0.4 (−0.8, +0.1) | +0.2 (−0.4, +0.8) |
| chrF++ change, shipped Q4_K_M file | +1.7 (+0.3, +3.1) | −0.3 (−1.1, +0.6) | −0.1 (−1.4, +1.2) |
| chrF++ change, same training data and no new tokens | +2.3 (+1.6, +3.1) | +0.6 (+0.1, +1.1) | +1.2 (+0.6, +1.8) |

\*19 of 20 sentences: one WazoCore reply contained a broken character and
the server returned an error (see Limitations).

- **Speed.** llama.cpp (build 9430), Q4_K_M, one Apple M4 laptop, 20
  translations per language. Both models answered each sentence back to
  back, taking turns going first. Decoding speed in tokens per second is
  about the same for both models; the gain comes from needing fewer tokens.
  The whole request includes reading the prompt, with no prompt cache.
- **Quality at full precision.** Greedy English → X translation, scored with
  chrF++ against human references in Unicode NFC, with 95% paired-bootstrap
  intervals for the difference, on NTREX-128 (500 news sentences per
  language) and FLORES-200 (200), pooled over two independent training runs.
- **Quality of the shipped file.** The two Q4_K_M files compared with each
  other in llama.cpp on 250 NTREX-128 sentences per language.
- **Control.** The last row is the original model trained on the same data
  with no new tokens. It separates the effect of the vocabulary from the
  effect of the training data.

The protocol, the full tables and the command behind each number are in
[`docs/METHOD.md`](https://github.com/Adeniyikayodee/wazocore/blob/main/docs/METHOD.md).

## Limitations

- **Broken characters.** About one reply in a thousand (4 of 4,200 in our
  tests) contains half of a multi-byte letter such as ṣ or ƙ. The text is
  then not valid Unicode and llama-server returns an error for that request.
  The original model does not do this. Retrying the request with sampling is
  the practical workaround: on the one prompt that fails on the shipped file,
  24 of 30 retries at temperature 0.3 and 27 of 30 at 0.7 returned valid text.
- **Yoruba loops.** Under greedy decoding 5 and 11 of 700 Yoruba replies in
  the two training runs fall into a repetition loop, usually a run of one
  accent mark, against 2 for the original. Hausa and Igbo are unaffected. On
  the shipped file a repetition penalty of 1.1 cut Yoruba loops from 4 to 2
  in 250 replies and changed chrF++ by +0.9 (−0.1, +1.9); it does not stop a
  run of accent marks. Use it for Yoruba only: the same penalty changed Hausa
  by −0.6 (−1.3, +0.0) and Igbo by +0.7 (−0.2, +1.6).
- **Hausa** is 0.3 to 0.4 chrF++ below the original in every measurement. The
  intervals include zero, so a loss is not established, but Hausa is the one
  language where WazoCore has not reached the original.
- Input should be in composed Unicode (NFC). On decomposed text (a letter
  followed by a combining accent) the token saving falls to about 1.15 times
  for Yoruba and 1.27 times for Igbo.
- Quality was measured on English → X translation in two domains (FLORES-200
  and NTREX-128 news). Dialogue, reasoning and X → English were not scored,
  and English quality has not been scored. English speed is 1.06× on the
  CPU and 0.95× on the GPU.
- Every training variant was scored on the same 700 sentences per language
  and the release was chosen among them, so its scores on those sentences are
  not an independent confirmation. No test sentence is trained on.
- Speed was measured on one laptop, on its CPU and on its GPU, with 20
  prompts per language. The GPU run was throttled to about half the
  machine's cool speed, and no other GPU was measured.
- Everything else about N-ATLaS carries over, including its biases, its
  8,092-token context and the limits of its instruction tuning.

## Licence

WazoCore is a derivative of N-ATLaS, released under the same **N-ATLaS
Open-Source Research and Innovation License (Terms of Use v1.0)**. Key terms:

- the attribution above must be kept;
- renamed derivatives must end in "Powered by Awarri.";
- derivatives must use the same licence;
- at most 1,000 active end-users per rolling 30 days without a commercial
  licence;
- surveillance, discriminatory profiling, disinformation, military use and
  unauthorised personal-data scraping are prohibited.

Read the full terms on the N-ATLaS model page before use.
