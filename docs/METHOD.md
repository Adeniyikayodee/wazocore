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
   - 15% carefully spelled Wikipedia text (yo/ha/ig 0.3 each, en 0.1 of that share);
   - 85% self-distilled replies (below);
   - a KL anchor (weight 1.0) to the frozen original on the replies tokenised the old way;
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

## Results (v4, the release)

**Quality.** 200 English → X FLORES devtest translations per language; see
the v4 table below (`results/vocab/cloud-v4/`). Compared with the original,
Yoruba is +1.0, Hausa −1.1 (not significant) and Igbo −1.3 (significant).

**Speed.** llama.cpp 9430, M4 MacBook Air CPU, 4 threads, Q4_K_M, 20 sentences
per language. One model was in memory at a time, and the first-mover alternated
each sentence (`results/vocab/m4-cpu-v4-paired.json`).

| | words/s | words/s × | chars/s × | tokens per word |
|---|---|---|---|---|
| Yoruba* | 2.97 → 4.62 | 1.55 | 1.51 | 2.92 → 1.99 |
| Hausa | 3.99 → 6.54 | 1.64 | 1.65 | 2.10 → 1.43 |
| Igbo | 3.64 → 5.83 | 1.60 | 1.61 | 2.54 → 1.58 |
| English | 7.43 → 7.14 | 0.96 | 0.98 | 1.28 → 1.31 |

\*19 sentences. One Yoruba request stalled for 223 s because of the laptop
(0.3 tokens/s against 7–13 for every other request); with it included, the
Yoruba figure is 0.59×. Raw data: `results/vocab/m4-cpu-v4-paired.json`. Decode tokens/s is unchanged (about 9), so the larger
vocabulary costs nothing. llama.cpp's tokenisation matches the reference
tokenizer on 160/160 test sentences. There were 0 failed replies in 80.

v3's numbers, the previous release candidate, are in `results/vocab/cloud-v3/`
and `results/vocab/m4-cpu-v3-paired.json`.

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

## v4 and the control run (200 sentences)

v4 trains on 85% distilled replies and 15% carefully spelled Wikipedia, with
a KL anchor (weight 1.0) to the frozen original on replies tokenised the old
way. The control run uses the identical recipe on the original model, with
no new tokens. Evidence: `results/vocab/cloud-v4/`.

| vs original | Yoruba | Hausa | Igbo | Hausa ƙɗɓ /100 chars | Igbo dots /100 chars | loops /600 |
|---|---|---|---|---|---|---|
| original | 25.81 | 48.97 | 38.60 | 0.39 | 9.25 | 2 |
| control | +0.3 (−0.8, +1.4) | −0.7 (−1.7, +0.5) | −0.4 (−1.3, +0.5) | 0.29 | 9.41 | 1 |
| **v4** | +1.0 (−0.1, +2.1) | −1.1 (−2.5, +0.2) | **−1.3 (−2.2, −0.3)** | 0.29 | 8.59 | 4 |

- **The Hausa spelling loss comes from the training recipe.** The control
  drops the hooked letters just as much. The Wikipedia filter did not fix
  it, so the Wikipedia hypothesis above was wrong.
- **Of Igbo's −1.3, about −0.4 is recipe and about −0.9 is the vocabulary
  change.** The control's Igbo is statistically equal to the original.
- **The extra repetition loops come with the new tokens** (control 1, v4 4).
- **v4 is the release candidate.** Yoruba is +1.0, Hausa is within noise,
  Igbo is −1.3 (v3: −1.9 on 100 sentences), and it uses 1.53× (yo),
  1.47× (ha) and 1.65× (ig) fewer tokens per character.

## Open work

- **Close the rest of the Igbo gap.** It is mostly a vocabulary effect now.
  Candidates:
  - more training signal for the rarely seen Igbo tokens;
  - pruning the Igbo tokens behind the loops;
  - a lighter recipe (a stronger KL anchor, fewer trained layers) for the
    Hausa spelling drift.
- **Measure more of the model.** So far only one direction and one domain were
  scored; X → English, chat, and speed on GPU (Metal/CUDA) remain.
- **Test other runtimes.** Ollama and LM Studio have not been tested.
