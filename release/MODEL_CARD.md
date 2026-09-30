---
license: other
license_name: n-atlas-open-source-research-and-innovation-license
base_model: NCAIR1/N-ATLaS
language: [yo, ha, ig, en]
tags: [llama-3, vocabulary-expansion, yoruba, hausa, igbo, gguf]
---

<!-- The N-ATLaS licence requires renamed derivatives to end with "Powered by Awarri." -->
# WazoCore Powered by Awarri.

**Nigerian languages at the speed of English.** WazoCore is N-ATLaS with
15,000 new Yoruba, Hausa and Igbo tokens. It writes these languages in 1.4 to
1.6 times fewer tokens, so replies arrive 1.55 to 1.64 times faster in any
standard engine, with no change to your code.

> N-ATLaS is an initiative of the Federal Ministry of Communications,
> Innovation and Digital Economy, and powered by Awarri Technologies.

## Why it exists

N-ATLaS inherits Llama-3's vocabulary, which was built mostly for English. A
Yoruba word costs about 2.9 tokens where an English word costs 1.2, so every
Nigerian-language reply is slower to generate, more expensive to serve, and
fits less context. WazoCore adds whole-word tokens for the three languages and
keeps every original token, so English and everything N-ATLaS already knows
stay intact.

## Results

All numbers compare WazoCore with the original N-ATLaS on sentences never used
in development.

**Speed.** llama.cpp (build 9430), Apple M4 laptop CPU, 4 threads, Q4_K_M, 20
translations per language. Both models answered each sentence back to back,
taking turns going first.

| Language | Words/s, original → WazoCore | Speed-up (words) | Speed-up (characters) | Tokens per word |
|---|---|---|---|---|
| Yoruba* | 2.97 → 4.62 | 1.55× | 1.51× | 2.92 → 1.99 |
| Hausa | 3.99 → 6.54 | 1.64× | 1.65× | 2.10 → 1.43 |
| Igbo | 3.64 → 5.83 | 1.60× | 1.61× | 2.54 → 1.58 |
| English | 7.43 → 7.14 | 0.96× | 0.98× | 1.28 → 1.31 |

\*19 sentences. One Yoruba request stalled for 223 s because of the host (0.3
tokens/s against 7 to 13 for every other request); including it, the Yoruba
figure is 0.59×. Raw data: `results/vocab/m4-cpu-v4-paired.json`.

Raw decoding speed in tokens per second is the same for both models. The whole
gain comes from needing fewer tokens.

**Quality.** Greedy English → X translation, scored with chrF++ against human
references, with 95% paired-bootstrap intervals for the difference. The table
shows NTREX-128 (500 news sentences per language, never used in development)
for two independent training runs.

| Language | chrF++ change vs original, run 1 | Run 2 |
|---|---|---|
| Yoruba | +0.1 (−0.2, +0.4) | +0.4 (+0.1, +0.7) |
| Hausa | −0.6 (−1.3, +0.1) | −0.4 (−1.0, +0.2) |
| Igbo | −1.2 (−2.1, −0.3) | −0.9 (−1.7, −0.0) |

Yoruba matches or beats the original, Hausa is statistically indistinguishable
from it, and Igbo is about 1 point lower (2 to 3%). FLORES-200 shows the same
pattern. A control run attributes about a third of the Igbo gap to the training
recipe and the rest to the new vocabulary.

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
   pieces; its output row is the byte-length-weighted mean.
3. **Training.** About 40 minutes on one A100: 5M tokens train the new rows and
   the bottom and top two of the 32 transformer layers. The data is 85% the
   original model's own replies to about 30,000 translation requests,
   re-tokenised with the new vocabulary, and 15% carefully spelled Wikipedia
   text. A KL term keeps predictions close to the original's, and the original
   vocabulary rows receive no updates.
4. **Export.** GGUF from the bf16 weights, quantised to Q4_K_M. llama.cpp's
   tokenisation matches the reference on all 160 test sentences.

The code, the data recipe and every measurement are in the WazoCore repository
(`docs/METHOD.md`).

## Limitations

- Igbo translation quality is about 1 chrF++ point below the original.
- Quality was measured on English → X translation in two domains (FLORES-200
  and NTREX-128 news). Dialogue, reasoning and X → English were not scored.
- Speed was measured on one laptop CPU; GPU inference was only spot-checked
  (about 21 tokens/s with Metal on an M4).
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
