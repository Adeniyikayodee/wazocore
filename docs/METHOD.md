# WazoCore: method and results

WazoCore expands N-ATLaS's vocabulary with 15,000 Yoruba, Hausa and Igbo tokens
and trains the model to use them. This document describes the recipe, the
evidence behind every reported number, and the approaches we tested and
rejected. Raw results are in `results/`.

## 1. The problem

N-ATLaS uses Llama-3's 128,256-token vocabulary, built mostly for English. On
FLORES-200 devtest (`results/tokenizer/`) the same sentences need:

| Language | Tokens per word | Tokens vs English |
|---|---|---|
| English | 1.24 | 1.00× |
| Hausa | 2.13 | 1.97× |
| Igbo | 2.62 | 2.40× |
| Yoruba | 2.91 | 2.70× |

An inference engine produces a roughly fixed number of tokens per second, so
Yoruba text arrives at about a third of English's speed and costs about 2.7
times as much to serve. We first tried to make the engine faster (a Rust CPU
runtime, TEAL activation sparsity, speculative decoding). llama.cpp remained
about three times faster than our kernels, so we closed that line; its code is
preserved under the `pre-cleanup` git tag. Reducing the number of tokens helps
every engine at once, which is the approach taken here.

## 2. The recipe

1. **Vocabulary** (`vocab_expand`). Text is first segmented by the original
   tokenizer; 15,000 BPE merges are then learned on top, jointly for Yoruba,
   Hausa and Igbo, from 4M characters per language (Wikipedia plus the FLORES
   dev split). Merges whose result is not whole UTF-8 are rejected. Every
   original token keeps its id, so English is unaffected.
2. **Initialisation** (`vx_core`). A new token's input embedding is the mean of
   its original pieces' embeddings; its output embedding is the
   byte-length-weighted mean (arXiv 2608.03494).
3. **Self-distillation data** (`vx_torch distill`). The original model's own
   greedy replies to about 30,000 translation requests (English to and from
   each language), drawn from FLORES dev and from Wikipedia with English
   passages filtered out, re-tokenised with the new vocabulary.
4. **Training** (`vx_torch train`). The new embedding rows and the bottom and
   top two transformer layers (arXiv 2406.11477) train on 5M tokens:
   - 85% self-distilled replies, 15% carefully spelled Wikipedia text;
   - a KL anchor (weight 1.0) to the frozen original on the same replies under
     the original tokenisation;
   - sequence length 512, effective batch 16, learning rate 5e-5 with warm-up
     and cosine decay, gradient clipping at 1.0;
   - fp32 master weights for trained tensors, bf16 for frozen layers, and a
     gradient mask that keeps every original vocabulary row exact.
5. **Export.** bf16 weights to GGUF to Q4_K_M with llama.cpp. The converter
   identifies pre-tokenizers by a fingerprint that the new merges change; after
   confirming the pre-tokenizer is byte-identical, the fingerprint is mapped to
   `llama-bpe`.

Training takes about 40 minutes on one A100 80GB.

## 3. Results

### Speed

llama.cpp build 9430, Apple M4 laptop CPU, 4 threads, Q4_K_M, 20 translations
per language. Both models answered each sentence back to back with the
first-mover alternating, and only one model was in memory at a time
(`results/vocab/m4-cpu-v4-paired.json`).

| Language | Words/s, original → WazoCore | Speed-up, words | Speed-up, characters | Tokens per word |
|---|---|---|---|---|
| Yoruba* | 2.97 → 4.62 | 1.55× | 1.51× | 2.92 → 1.99 |
| Hausa | 3.99 → 6.54 | 1.64× | 1.65× | 2.10 → 1.43 |
| Igbo | 3.64 → 5.83 | 1.60× | 1.61× | 2.54 → 1.58 |
| English | 7.43 → 7.14 | 0.96× | 0.98× | 1.28 → 1.31 |

\*19 sentences. One Yoruba request stalled for 223 s on the host (0.3 tokens/s
against 7 to 13 for every other request); including it, the Yoruba figure is
0.59×.

Decode speed in tokens per second is unchanged (about 9), so the larger
vocabulary costs nothing at inference. llama.cpp's tokenisation matches the
reference tokenizer on 160 of 160 test sentences, and none of the 80 benchmark
replies failed.

### Quality

Greedy English → X translation, scored with chrF++ against human references.
Intervals are 95% paired-bootstrap intervals (1,000 resamples) on the
difference from the original. NTREX-128 was never used during development;
the recipe was trained twice with different seeds (`results/vocab/validate/`).

| chrF++ change vs original | Yoruba | Hausa | Igbo |
|---|---|---|---|
| NTREX-128 (500), run 1 | +0.1 (−0.2, +0.4) | −0.6 (−1.3, +0.1) | −1.2 (−2.1, −0.3) |
| NTREX-128 (500), run 2 | +0.4 (+0.1, +0.7) | −0.4 (−1.0, +0.2) | −0.9 (−1.7, −0.0) |
| FLORES-200 (200), run 1 | +0.6 (−0.5, +1.6) | −0.9 (−2.1, +0.4) | −1.1 (−2.1, −0.1) |
| FLORES-200 (200), run 2 | +0.5 (−0.6, +1.5) | −0.7 (−1.9, +0.5) | −1.5 (−2.4, −0.5) |

- Yoruba is equal to or slightly better than the original, Hausa is
  statistically indistinguishable from it, and Igbo is about 1 chrF++ lower
  (2 to 3%), consistently across both test sets and both runs.
- Run-to-run spread is about ±0.3 to 0.5 chrF++.
- Tokens per character on NTREX fall 1.54× (yo), 1.41× (ha) and 1.62× (ig),
  matching FLORES.
- NTREX Yoruba scores are low for every model (about 13 chrF++): its references
  follow other conventions and contain some wrong characters (for example ϙ
  for ọ). This affects all models equally.
- An earlier single-run FLORES estimate of Yoruba +1.0 was optimistic; the
  figures above supersede it.

## 4. Where the Igbo gap comes from

A control run applies the identical recipe to the original model, with no new
tokens (`results/vocab/cloud-v4/`, FLORES-200, 200 sentences):

| vs original | Yoruba | Hausa | Igbo | Hausa ƙɗɓ /100 chars | Igbo dots /100 chars | Loops /600 |
|---|---|---|---|---|---|---|
| Original | 25.81 | 48.97 | 38.60 | 0.39 | 9.25 | 2 |
| Control | +0.3 (−0.8, +1.4) | −0.7 (−1.7, +0.5) | −0.4 (−1.3, +0.5) | 0.29 | 9.41 | 1 |
| WazoCore | +1.0 (−0.1, +2.1) | −1.1 (−2.5, +0.2) | −1.3 (−2.2, −0.3) | 0.29 | 8.59 | 4 |

- About a third of the Igbo gap comes from the training recipe and two thirds
  from the new vocabulary; the control's Igbo is statistically equal to the
  original.
- The drop in Hausa hooked letters (ƙ ɗ ɓ) comes from the recipe: the control
  shows it too.
- The extra repetition loops come with the new tokens (control 1, WazoCore 4).
  The original also loops occasionally under greedy decoding.

## 5. How we got here

Each version was measured before the next was built.

| Version | Change | Result | Lesson |
|---|---|---|---|
| Untrained | Expansion and initialisation only | Yoruba BPC 2.74 → 4.19 | Initialisation alone is far from enough |
| Local | Training on a 16 GB laptop (MLX, 4-bit) | ~15 tokens/s from memory swapping; lr 5e-4 diverged | Training needs one 80 GB GPU |
| Rows only | Train new embeddings only | Behaviour kept, new tokens barely used | The layers must adapt too |
| Edge layers | Rows and edge layers on Wikipedia | BPC below the original, English included; replies looped | BPC is a misleading gate: it rewards drift toward encyclopedic text |
| v1 | + 6k self-distilled replies | Looked equal on 30 sentences; 12 half-character tokens produced invalid UTF-8 | 30 sentences is too few; never split a character |
| v2 | Character-safe vocabulary | Hausa −2.0, Igbo −2.3 chrF++ on 100 sentences | The gate needs bootstrap intervals |
| v3 | + 24k Wikipedia replies, 60% share | Yoruba and Hausa equal; Igbo −1.9 | More behavioural data helps |
| **v4** | 85% replies, filtered Wikipedia, KL anchor | Section 3 | Release |
| v5 | On-policy distillation from v4 | Igbo gap closed, speed-up lost | Section 6 |

Two hypotheses from the v3 analysis were tested and ruled out. Hausa Wikipedia
often drops hooked letters (`dan` for `ɗan`), which suggested the model learned
careless spelling from it; filtering Wikipedia for spelling did not restore the
letters, and the control run showed the cause is the training recipe. The
tokenizer was also ruled out: it makes correct and careless spellings equally
cheap.

## 6. v5: on-policy distillation, and why it was rejected

v5 refined v4 with on-policy distillation across tokenizers (`vx_torch
onpolicy`, `scripts/cloud_vx_v5.sh`, `results/vocab/v5-rejected/`). The student
wrote replies with its new tokens; the frozen original scored the same text
under its own tokenisation; the two were aligned chunk by chunk (every chunk is
one WazoCore token) and trained with reverse KL in REINFORCE form, keeping the
KL anchor. Two seeds, 300 steps of 32 rollouts each.

| NTREX-128 (500) | Original | v4 | v5 (seed 0 / 1) |
|---|---|---|---|
| Igbo chrF++ vs original | n/a | −1.2 / −0.9 | −0.6 / 0.0 |
| Hausa chrF++ vs original | n/a | −0.6 / −0.4 | 0.0 / +0.1 |
| Hausa ƙɗɓ, Igbo dots (/100 chars) | 0.206, 8.87 | 0.174, 8.47 | 0.211 to 0.237, 9.08 to 9.25 |
| Loops (/1,500) | 1 | 7 | 7 / 6 |
| Igbo characters per token | 2.12 | 3.44 | 2.26 |

v5 closed the Igbo gap and restored the special letters, but only by returning
to the original tokens: the share of new tokens in its own replies fell from
23% to 6 to 8% during training, leaving its token count within 7% of the
original's. The on-policy objective rewards whatever the original finds likely,
and the original's likelihood lives on its own short tokens; reverse KL is
mode-seeking, so it pulls the student onto that segmentation.

The fix is to score only the canonical path: re-tokenise each rollout with the
new tokenizer and train on that segmentation, so the student can lower its loss
only by writing better text. That is the basis of the next version.

## 7. Limitations and open work

- **Igbo quality** is about 1 chrF++ below the original; closing it without
  losing speed is the main open problem (section 6).
- **Evaluation scope.** Quality is measured on English → X translation in two
  domains (FLORES-200 and NTREX-128 news). Dialogue, reasoning and X → English
  are not yet scored; native-speaker judgement is being collected with the
  blind A/B kit in `tester-kit/`.
- **Hardware.** Speed was measured on one laptop CPU. GPU inference was
  spot-checked (about 21 tokens/s with Metal on an M4); llama.cpp and LM Studio
  are tested, Ollama is not yet.
- **Inherited behaviour.** WazoCore keeps N-ATLaS's biases, its 8,092-token
  context and the limits of its instruction tuning.
