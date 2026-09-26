# WazoCore: method and findings

## Problem: the token tax

N-ATLaS uses Llama-3's 128,256-token vocabulary, which was built mostly for
English. On the FLORES-200 devtest set (`results/tokenizer/`), the same
sentences need:

| | tokens per word | tokens vs English |
|---|---|---|
| English | 1.24 | 1.00× |
| Hausa | 2.13 | 1.97× |
| Igbo | 2.62 | 2.40× |
| Yoruba | 2.91 | 2.70× |

An engine produces a fixed number of tokens per second. Yoruba text therefore
arrives at about a third of English's speed and costs about 2.7× as much to
serve. Earlier attempts to speed up the engine itself (a Rust CPU runtime, TEAL
activation sparsity, speculative decoding) were closed after measurement,
because llama.cpp stayed about 3× faster than our kernels. Their code is in git
history under the `pre-cleanup` tag. Cutting the number of tokens helps every
engine at once.

## Method

1. **Vocabulary (`vocab_expand`).** Each text is first segmented with the
   original tokenizer. 15,000 BPE merges are then learned on top, jointly for
   Yoruba, Hausa and Igbo, from 4M characters per language (Wikipedia plus the
   FLORES dev split). Merges whose result is not whole UTF-8 are rejected.
   Every original token keeps its id, so English is unaffected.
2. **Initialisation (`vx_core`).** A new token's input row is the mean of its
   original pieces' rows; its output row is the byte-length-weighted mean
   (arXiv 2608.03494).
3. **Training (`vx_torch train`).** The new rows are trained together with the
   bottom and top two transformer layers (arXiv 2406.11477) on 5M tokens:
   - 40% Wikipedia text (yo/ha/ig 0.3 each, en 0.1);
   - 60% self-distilled replies (below);
   - sequence length 512, batch 16, learning rate 5e-5, warm-up then cosine,
     gradient clipping at 1.0;
   - fp32 master weights for the trained tensors and bf16 for the frozen
     layers.

   A gradient mask keeps every original row exact.
4. **Self-distillation (`vx_torch distill`).** The original model's own greedy
   replies to about 30,000 translation requests (English ↔ yo/ha/ig), drawn
   from FLORES dev and from Wikipedia sentences with English passages filtered
   out. Training on these, re-tokenised, teaches the new model to give the same
   answers, including where they end, written with the new tokens.
5. **Export.** Hugging Face bf16 → GGUF → Q4_K_M with llama.cpp. The converter
   identifies pre-tokenizers by a fingerprint that the new merges change, so
   that fingerprint is mapped to `llama-bpe` after checking that the
   pre-tokenizer is byte-identical.

## What we learned getting there

Every step was measured; the evidence is in `results/vocab/`.

| Attempt | Outcome | Lesson |
|---|---|---|
| Untrained expansion | Yoruba BPC 2.74 → 4.19 | Initialisation alone is far from enough |
| Training on a 16 GB Mac (MLX, 4-bit) | ~15 tok/s from swapping; lr 5e-4 diverged | Rent one 80 GB GPU (about $2/hour) |
| New rows only | Chat kept, but new tokens barely used, so no speed-up | The layers must adapt too |
| Rows + edge layers on Wikipedia | BPC *below* the original, English included; replies looped | **BPC is a misleading gate**: it rewards drifting from chat towards encyclopedic text |
| + 6k distilled replies (v1) | Looked equal on 30 sentences | **30 sentences is too few**: 12 fragment tokens also emitted invalid UTF-8 about once in 80 replies |
| Fragment-free vocabulary (v2) | 100 sentences: ha −2.0, ig −2.3 chrF++ (significant) | The gate needs paired-bootstrap intervals |
| + 24k Wikipedia replies at a 60% share (**v3**) | yo, ha equal; ig −1.9 (significant) | More behavioural data helps; Igbo is the remaining gap |

## Results (v3, the release)

**Quality.** 100 English → X FLORES devtest translations per language; chrF++
against the human references; 95% paired-bootstrap interval of the
difference (`results/vocab/cloud-v3/`).

| | original | WazoCore | difference | replies ended |
|---|---|---|---|---|
| Yoruba | 24.45 | 25.25 | +0.8 (−0.9, +2.5) | 94 → 95 |
| Hausa | 49.03 | 47.89 | −1.1 (−2.9, +0.8) | 99 → 100 |
| Igbo | 39.19 | 37.30 | −1.9 (−3.6, −0.4) | 100 → 98 |

**Speed.** llama.cpp 9430, M4 MacBook Air CPU, 4 threads, Q4_K_M, 20 sentences
per language. One model was in memory at a time, and the first-mover alternated
each sentence (`results/vocab/m4-cpu-v3-paired.json`).

| | words/s | words/s × | chars/s × | tokens per word |
|---|---|---|---|---|
| Yoruba | 2.35 → 2.81 | 1.19 | 1.31 | 2.92 → 2.55 |
| Hausa | 3.25 → 4.89 | 1.50 | 1.50 | 2.10 → 1.41 |
| Igbo | 2.68 → 4.36 | 1.63 | 1.61 | 2.54 → 1.62 |
| English | 5.49 → 5.40 | 0.98 | 0.97 | 1.28 → 1.30 |

Decode tokens/s is unchanged (about 7 on this run), so the larger vocabulary
costs nothing. llama.cpp's tokenisation matches the reference tokenizer on
160/160 test sentences. There were 0 invalid replies in 80.

## Igbo diagnosis (v3, 100 sentences)

- **Loss is concentrated.** 17 Igbo sentences lose more than 10 chrF++ and 7
  gain that much. The 10 worst account for 45% of the total loss.
- **Two of them are repetition loops inside a word.** Examples are
  `ahịa`+`ere`+`ere`… and `ọ`+`bar`+`abar`…, and they are the two replies that
  never ended. The original model also loops on 2 of 300 replies (Yoruba), so
  greedy decoding is partly to blame, but these two loops are new.
- **The raw Wikipedia share teaches sloppy spelling.** Hausa Wikipedia spells
  `dan`/`kasa` more often than `ɗan`/`ƙasa`, and v3's Hausa replies use hooked
  letters at 0.28 per 100 characters (original 0.50, reference 0.42). About
  12% of common Igbo words in Igbo Wikipedia are undotted (`ndi` for `ndị`),
  whereas the original model's own replies are cleaner than Wikipedia (dotted
  letters 8.67 per 100 characters, reference 8.43).
- **The tokenizer is not the cause.** The expansion made hooked and plain
  spellings equally cheap (1 token each for common words).

## Open work

- **Close the Igbo gap (v4).** Candidates:
  - replace most of the raw Wikipedia share with distilled replies, or filter it
    for orthography;
  - add a KL anchor to the frozen original on old-tokenised text, against drift
    and loops;
  - run a control (same recipe, no new tokens) to separate drift from vocabulary;
  - score on 200 sentences.
- **Measure more of the model.** So far only one direction and one domain were
  scored; X → English, chat, and speed on GPU (Metal/CUDA) remain.
- **Test other runtimes.** Ollama and LM Studio have not been tested.
