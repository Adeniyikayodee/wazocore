---
license: other
license_name: n-atlas-open-source-research-and-innovation-license
base_model: NCAIR1/N-ATLaS
language: [yo, ha, ig, en]
tags: [llama-3, vocabulary-expansion, yoruba, hausa, igbo, gguf]
---

<!-- The N-ATLaS licence requires renamed derivatives to end with "Powered by Awarri." -->
# WazoCore Powered by Awarri.

**Faster in your own language.** WazoCore is N-ATLaS with
15,000 new Yoruba, Hausa and Igbo tokens. It writes these languages in 1.4 to
1.6 times fewer tokens, so replies arrive 1.6 to 1.7 times faster in
llama.cpp and LM Studio, with no change to your code.

> N-ATLaS is an initiative of the Federal Ministry of Communications,
> Innovation and Digital Economy, and powered by Awarri Technologies.

## Why it exists

N-ATLaS inherits Llama-3's vocabulary, which was built mostly for English. A
Yoruba word costs about 2.9 tokens where an English word costs 1.2, so every
Nigerian-language reply is slower to generate, more expensive to serve, and
fits less context. WazoCore adds word and word-piece tokens for the three
languages and keeps every original token with its exact embedding, so English
text tokenises as before.

## Results

All numbers compare WazoCore with the original N-ATLaS. Speed and the second
quality table are measured on the shipped Q4_K_M file. NTREX-128 was never
used in development.

**Speed.** llama.cpp (build 9430), Apple M4 laptop CPU, 4 threads, Q4_K_M, 20
translations per language. Both models answered each sentence back to back,
taking turns going first.

| Language | Words/s, original → WazoCore | Speed-up (words) | Speed-up (characters) | Tokens per word |
|---|---|---|---|---|
| Yoruba* | 2.73 → 4.45 | 1.63× | 1.58× | 2.87 → 1.91 |
| Hausa | 3.77 → 6.09 | 1.61× | 1.49× | 2.12 → 1.34 |
| Igbo | 3.06 → 5.24 | 1.72× | 1.72× | 2.53 → 1.56 |
| English** | 6.23 → 6.60 | 1.06× | 1.07× | 1.28 → 1.31 |

\*19 sentences: one WazoCore reply contained a broken character and the
server returned an error (see Limitations). \*\*19 sentences: one request to
the original stalled for 82 s on the host; with it the English figure is
1.99×. Raw data: `results/vocab/m4-cpu-v6-paired.json`.

Decoding speed in tokens per second is about 8 for both models. The gain
comes from needing fewer tokens.

**Quality at full precision.** Greedy English → X translation, scored with
chrF++ against human references in Unicode NFC, with 95% paired-bootstrap
intervals for the difference, on NTREX-128 (500 news sentences per language)
and FLORES-200 (200), for two independent training runs.

| Language | chrF++ change vs original, run 1 | Run 2 | Both runs pooled |
|---|---|---|---|
| Yoruba | +2.3 (+1.6, +3.1) | +2.0 (+1.2, +2.8) | +2.2 (+1.5, +2.9) |
| Hausa | −0.4 (−0.9, +0.2) | −0.4 (−0.9, +0.2) | −0.4 (−0.8, +0.1) |
| Igbo | +0.3 (−0.3, +1.0) | +0.1 (−0.5, +0.7) | +0.2 (−0.4, +0.8) |

**Quality of the Q4_K_M file.** The two Q4_K_M files compared with each other
in llama.cpp on 250 NTREX-128 sentences per language.

| chrF++ | Yoruba | Hausa | Igbo |
|---|---|---|---|
| Original | 28.57 | 46.64 | 42.79 |
| WazoCore | 30.31 | 46.38 | 42.69 |
| Difference | +1.7 (+0.3, +3.1) | −0.3 (−1.1, +0.6) | −0.1 (−1.4, +1.2) |

A control run with the same training data and no new tokens is +2.3, +0.6
and +1.2, so most of the Yoruba gain comes from the added human translations
and the new vocabulary costs up to a point in Hausa and Igbo against that
control.

## Use

WazoCore is a standard Llama-3 GGUF. It is tested in llama.cpp and LM Studio:

```sh
llama-server -m WazoCore-8B.Q4_K_M.gguf -c 4096 --port 8080
curl -s localhost:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Translate into Hausa: The clinic opens at eight in the morning."}]}'
```

The full-precision weights load with `transformers` 5.x, like N-ATLaS. It runs
offline on a laptop with 8 GB of memory or more.

## How it was made

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

The code, the data recipe and every measurement are in the WazoCore repository
(`docs/METHOD.md`).

## Limitations

- **Broken characters.** About one reply in a thousand (4 of 4,200 in our
  tests) contains half of a multi-byte letter such as ṣ or ƙ. The text is
  then not valid Unicode and llama-server returns an error for that request.
  The original model does not do this. Retrying the request with sampling is
  the practical workaround; we have not measured how often it succeeds.
- **Yoruba loops.** Under greedy decoding 5 and 11 of 700 Yoruba replies in
  the two training runs fall into a repetition loop, usually a run of one
  accent mark, against 2 for the original. A repetition penalty in the
  sampler is the usual remedy; we have not measured it. Hausa and Igbo are
  unaffected.
- **Hausa** is 0.3 to 0.4 chrF++ below the original in every measurement. The
  intervals include zero, so a loss is not established, but Hausa is the one
  language where WazoCore has not reached the original.
- Input should be in composed Unicode (NFC). On decomposed text (a letter
  followed by a combining accent) the token saving falls to about 1.15 times
  for Yoruba and 1.27 times for Igbo.
- Quality was measured on English → X translation in two domains (FLORES-200
  and NTREX-128 news). Dialogue, reasoning and X → English were not scored,
  and English quality has not been scored; English speed is unchanged.
- Speed was measured on one laptop CPU with 20 prompts per language; GPU
  inference was only spot-checked (about 21 tokens/s with Metal on an M4).
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
