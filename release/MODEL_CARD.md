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
1.6 times fewer tokens, so replies arrive 1.55 to 1.64 times faster in
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

All numbers compare WazoCore with the original N-ATLaS. Speed is measured on
the shipped Q4_K_M file; quality is given for the full-precision weights and
for the Q4_K_M file. NTREX-128 was never used in development.

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
references in Unicode NFC, with 95% paired-bootstrap intervals for the
difference. The table shows NTREX-128 (500 news sentences per language, never
used in development) for two independent training runs, and both runs pooled
with 200 FLORES-200 sentences.

| Language | chrF++ change vs original, run 1 | Run 2 | Pooled (700 sentences, both runs) |
|---|---|---|---|
| Yoruba | +0.6 (−0.3, +1.4) | +0.4 (−0.4, +1.2) | +0.6 (−0.1, +1.2) |
| Hausa | −0.6 (−1.3, +0.1) | −0.4 (−1.0, +0.2) | −0.6 (−1.1, −0.0) |
| Igbo | −1.2 (−2.1, −0.3) | −0.9 (−1.7, −0.0) | −1.1 (−1.7, −0.5) |

Yoruba is equal to or slightly better than the original, Hausa is about half a
point lower and Igbo about 1 point lower (2 to 3%). On 200 FLORES-200
sentences a control run with the same recipe and no new tokens is 0.4 below
the original in Igbo and WazoCore is 0.9 below the control (95% interval −1.9
to 0.0), which suggests most of the Igbo gap comes from the new vocabulary.

**Quality of the Q4_K_M file.** The two Q4_K_M files compared with each other
in llama.cpp on 250 NTREX-128 sentences per language: Yoruba +0.6 (−0.7,
+1.8), Hausa −1.2 (−2.0, −0.3), Igbo −2.9 (−4.3, −1.6). WazoCore loses more
than the original when quantised to 4 bits. If Igbo quality matters more to
you than file size, use the full-precision weights until a higher-precision
GGUF is published.

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

- Igbo translation quality is about 1 chrF++ point below the original and
  Hausa about half a point at full precision; in the Q4_K_M file the gaps are
  about 3 points and 1 point.
- Input should be in composed Unicode (NFC). On decomposed text (a letter
  followed by a combining accent) the token saving falls to about 1.15 times
  for Yoruba and 1.27 times for Igbo.
- Under greedy decoding WazoCore falls into a repetition loop slightly more
  often than the original (7 to 8 replies in 1,500 against 1).
- Quality was measured on English → X translation in two domains (FLORES-200
  and NTREX-128 news). Dialogue, reasoning and X → English were not scored.
- Speed was measured on one laptop CPU with 20 prompts per language; GPU
  inference was only spot-checked (about 21 tokens/s with Metal on an M4).
  English quality has not been scored; English speed is 0.96 times the
  original's.
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
