# WazoCore: history and rejected approaches

The release and its evidence are in [METHOD.md](METHOD.md). This document is
the record behind it: every earlier version, the previous release (v4) with
its full results, the analysis of its weaknesses, and the approaches that
were measured and rejected. Raw results are in `results/`.

## 1. Versions

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
| v4 | 85% replies, filtered Wikipedia, KL anchor | Sections 2 to 4 | Previous release |
| v5 | On-policy distillation from v4 | Igbo gap closed, speed-up lost | Section 5 |
| **v6** | Human translations, canonical anchor, calibrated initialisation, 12M tokens | [METHOD.md](METHOD.md) | Release. The new rows had been under-trained |

Two hypotheses from the v3 analysis were tested and ruled out. Hausa Wikipedia
often drops hooked letters (`dan` for `ɗan`), which suggested the model learned
careless spelling from it; filtering Wikipedia for spelling did not restore the
letters, and the control run showed the cause is the training recipe. The
tokenizer was also ruled out: it makes correct and careless spellings equally
cheap.

## 2. The v4 recipe

1. **Vocabulary** (`vocab_expand`). Text is first segmented by the original
   tokenizer; 15,000 BPE merges are then learned on top, jointly for Yoruba,
   Hausa and Igbo, from 4M characters per language (Wikipedia plus the FLORES
   dev split). Merges whose result is not whole UTF-8 are rejected. Every
   original token keeps its id and its embedding; English text changes
   slightly (0.12% fewer tokens, [METHOD.md](METHOD.md), section 2).
2. **Initialisation** (`vx_core`). A new token's input embedding is the mean of
   its original pieces' embeddings; its output embedding is the
   byte-length-weighted mean (after arXiv 2608.03494, which weights by
   characters and also calibrates the norm of input rows; v4 does neither; v6 does both).
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

## 3. v4 results

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

Decode speed in tokens per second did not change measurably (about 9 for
both models, with about 10% noise between requests), so the gain comes from
the token count. English runs at 0.96× in words per second; its quality has
not been scored. llama.cpp's tokenisation matches the
reference tokenizer on 160 of 160 test sentences, and none of the 80 benchmark
replies failed.

### Quality

Greedy English → X translation, scored with chrF++ against human references,
with both sides normalised to Unicode NFC. Intervals are 95% paired-bootstrap
intervals (1,000 resamples; 5,000 for the pooled row) on the difference from
the original. NTREX-128 was never used during development;
the recipe was trained twice with different seeds (`results/vocab/validate/`).

| chrF++ change vs original | Yoruba | Hausa | Igbo |
|---|---|---|---|
| NTREX-128 (500), run 1 | +0.6 (−0.3, +1.4) | −0.6 (−1.3, +0.1) | −1.2 (−2.1, −0.3) |
| NTREX-128 (500), run 2 | +0.4 (−0.4, +1.2) | −0.4 (−1.0, +0.2) | −0.9 (−1.7, −0.0) |
| FLORES-200 (200), run 1 | +0.7 (−0.6, +1.9) | −0.9 (−2.1, +0.4) | −1.1 (−2.1, −0.1) |
| FLORES-200 (200), run 2 | +0.7 (−0.6, +1.8) | −0.7 (−1.9, +0.5) | −1.5 (−2.4, −0.5) |
| **Pooled (700), both runs** | **+0.6 (−0.1, +1.2)** | **−0.6 (−1.1, −0.0)** | **−1.1 (−1.7, −0.5)** |

- Pooled over both test sets and both runs (`vx_torch pooled`,
  `results/vocab/validate/pooled-summary.json`), Yoruba is equal to or
  slightly better than the original, Hausa is about half a chrF++ point lower
  and Igbo about one point lower (2 to 3%). No single test set could show the
  Hausa loss; the pooled sentences do. An interval that includes zero shows a
  test lacked power, not that two models are equal.
- Against a non-inferiority margin of 1.0 chrF++ (one-sided 95% bound), Yoruba
  passes (+0.0), Hausa sits on the line (−1.0) and Igbo does not (−1.6). The
  margin was chosen after seeing these data, so this is exploratory.
- Run-to-run spread is about ±0.3 to 0.5 chrF++.
- On the models' own NTREX replies, tokens per character fall 1.54× (yo),
  1.41× (ha) and 1.62× (ig), matching FLORES.
- Earlier versions of this table reported NTREX Yoruba at about 13 chrF++ with
  much tighter intervals. That was a scoring artefact: 482 of the 500
  references are stored in decomposed Unicode (a letter followed by a
  combining accent) while the models write composed characters. With both
  sides in NFC the original scores 29.3 and WazoCore 29.9 and 29.8. The
  scorer now normalises (`vx_core.nfc_rows`). Hausa and Igbo are unaffected.
- An earlier single-run FLORES estimate of Yoruba +1.0 was optimistic; the
  figures above supersede it.

### Quality of the shipped file

Everything above was scored on the full-precision (bf16) weights. The file
users run is the Q4_K_M GGUF, so we scored that too: both Q4_K_M files in
llama.cpp (build 9430, greedy, the same prompts), on the first 250 NTREX-128
sentences per language (`scripts/gguf_quality.py`,
`results/vocab/gguf-quality/`).

| chrF++ | Yoruba | Hausa | Igbo |
|---|---|---|---|
| Original, Q4_K_M | 28.57 | 46.64 | 42.79 |
| WazoCore, Q4_K_M | 29.12 | 45.46 | 39.89 |
| Difference | +0.6 (−0.7, +1.8) | −1.2 (−2.0, −0.3) | −2.9 (−4.3, −1.6) |
| Same sentences, bf16 | +0.3 (−0.9, +1.5) | −0.7 (−1.7, +0.2) | −1.0 (−2.3, +0.2) |

- In the file we ship, the cost is about three chrF++ points in Igbo and one
  in Hausa, not one and a half. Yoruba is unchanged.
- WazoCore loses quality when quantised to 4 bits and the original does not:
  from bf16 to Q4_K_M WazoCore moves −1.2 in Igbo (−2.4, −0.3) and −0.5 in
  Hausa (−1.1, +0.1), the original +0.6 and 0.0. These two rows also change
  engine (PyTorch to llama.cpp), so they are indicative; the file-against-file
  row above has no such mix.
- The extra loss in the 4-bit files is real for Igbo: the difference between
  the two rows above is −1.9 (−3.3, −0.5). For Hausa it is −0.5 (−1.2, +0.3)
  and for Yoruba +0.2 (−1.1, +1.5).
- The cause took several eliminations. Ruled out first: the chat template and tensor
  types (identical in both files, no importance matrix); llama.cpp's
  tokenisation (identical to the reference tokenizer on 2,000 prompts, replies
  and references); the date in the system line (replies are identical with
  the training-time date); and the precision of the token embeddings (a file
  with `token_embd.weight` at Q8_0 scores +0.5 (−0.9, +1.9) against the
  shipped one on 200 Igbo sentences and stays 3.3 behind the original). What
  remained was 4-bit rounding of the transformer layers, or a difference
  between PyTorch and llama.cpp.
- A Q8_0 file in llama.cpp separates the two (Igbo, 240 sentences): it scores
  the same as the bf16 weights in PyTorch (0.0, interval −0.5 to +0.4, two
  thirds of replies identical) and 1.2 above the Q4_K_M file (+0.1, +2.3).
  The engine and the conversion are sound; the loss is 4-bit rounding.
- Part of that rounding loss sits in the four retrained layers. A Q4_K_M file
  that keeps blocks 0, 1, 30 and 31 at Q8_0 (`EDGE_TYPE=q8_0` in
  `scripts/cloud_gguf.sh`; 5.38 GB against 5.01 GB), built from the bf16
  weights, changes the scores by:

  | chrF++, 250 sentences | Yoruba | Hausa | Igbo |
  |---|---|---|---|
  | vs the shipped Q4_K_M file | +0.1 (−0.6, +0.7) | −0.2 (−0.6, +0.3) | +1.1 (−0.0, +2.3) |
  | vs the original's Q4_K_M file | +0.6 (−0.5, +1.7) | −1.4 (−2.3, −0.5) | −1.8 (−2.8, −0.7) |

  It recovers about one Igbo point of the 1.2 lost to rounding, writes the
  same replies in 5% fewer tokens, and leaves Yoruba and Hausa unchanged. An
  earlier build of the same layout, requantised from Q8_0, measured +1.7
  (+0.7, +2.9) for Igbo; the two builds do not differ significantly (−0.2).
  In both builds the same Hausa request (1 of 750) failed: the reply
  contained half of a two-byte letter and llama-server refuses invalid UTF-8.
  The shipped and original files had no such failure.
- Token savings in the same replies are unchanged: tokens per word fall from
  3.02 to 1.97 (Yoruba), 2.05 to 1.44 (Hausa) and 2.61 to 1.65 (Igbo).

## 4. Where v4's Igbo gap comes from

A control run applies the identical recipe to the original model, with no new
tokens (`results/vocab/cloud-v4/`, FLORES-200, 200 sentences):

| vs original | Yoruba | Hausa | Igbo | Hausa ƙɗɓ /100 chars | Igbo dots /100 chars | Loops /600 |
|---|---|---|---|---|---|---|
| Original | 26.90 | 48.97 | 38.61 | 0.39 | 9.25 | 2 |
| Control | +0.5 (−0.6, +1.8) | −0.7 (−1.7, +0.5) | −0.4 (−1.3, +0.5) | 0.29 | 9.41 | 1 |
| WazoCore | +1.2 (−0.1, +2.4) | −1.1 (−2.5, +0.2) | −1.3 (−2.2, −0.3) | 0.29 | 8.59 | 4 |

- By point estimate, about a third of the Igbo gap comes from the training
  recipe and two thirds from the new vocabulary; the control's Igbo is
  statistically equal to the original. The split is suggestive only: on these
  200 sentences WazoCore against the control is −0.9 with a 95% interval of
  (−1.9, +0.0). The control has not been run on NTREX-128.
- Hausa hooked letters (ƙ ɗ ɓ) are rarer per 100 characters for both the
  control and WazoCore, so the recipe is the likelier source. A paired
  word-by-word comparison of the two models' NTREX replies finds losses and
  gains balanced (33 words lost a hooked letter, 30 gained one; Igbo dots 13
  lost, 17 gained), so we do not claim a systematic loss of special letters.
- WazoCore loops more often than the control under greedy decoding (control
  1, WazoCore 4). The original also loops occasionally. On NTREX-128 the
  count is 7 and 8 loops in 1,500 replies for the two runs against 1 for the
  original (paired exact test, p = 0.07 and 0.04), mostly in Yoruba and Igbo.
  The loops themselves are short stutters written in original tokens, usually
  starting inside a word after a foreign name or a number. Removing looped and
  unstopped replies moves the Igbo difference by about 0.15 chrF++, so loops
  are a small part of the gap.
- The gap is a broad shift, not a few failures: dropping the 25 best and 25
  worst sentences leaves Igbo at about −1.0, and sentences with more new
  tokens are not worse than sentences with fewer.
- WazoCore does not always take the longest token. Its replies use 4% (Yoruba),
  7% (Hausa) and 5% (Igbo) more tokens than its own tokenizer gives the same
  text, because a word can be written with the new token or with its original
  pieces. That is speed left unclaimed, and it is the mild form of what
  undid v5 (section 5).

## 5. v5: on-policy distillation, and why it was rejected

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
to the original tokens: the share of new tokens in its sampled training
replies fell from 22% to 6 to 8% within the first 30 steps (v4's greedy
replies use 31 to 38%), leaving its token count within 7% of the original's.

The cause is the target, not the choice of divergence. A word can be written
with its new token or with the original pieces, and the per-chunk target
counted that word's probability twice: once under the new token, and again
under its first piece, whose probability under the original model includes
every continuation. A target whose mass exceeds one is best matched by
spreading probability across both spellings, which is what the student did.
The correction for scoring a larger vocabulary with a model of the smaller one
is published (arXiv 2512.14954): an original token's target is the probability
that it occurs and the merge does not follow.

The fix therefore has two parts: train only on the canonical segmentation
(re-tokenise each rollout with the new tokenizer), and correct the target for
original tokens as above. The KL anchor on the old segmentation, which also
keeps the piece spelling alive in v4, is replaced by one on canonical text.
That is the basis of the next version.

## 6. From v4 to v6

v4 trains the student on the original's own greedy replies, so the best it can
do is match the original, and the new vocabulary then costs 0.6 chrF++ in Hausa
and 1.1 in Igbo. v6 changes the two things the measurements point at: what the
student is anchored to, and what it learns from. Those changes and their
results are in [METHOD.md](METHOD.md), sections 2 and 4. One planned change
was measured first and left out.

**MBR replies** (`distill --candidates`). The original writes several replies
to each request and the one closest to the others is kept (arXiv
2309.10966). We measured the headroom on 31 FLORES dev sentences per
language with the original's Q4_K_M file, nine candidates each
(`results/vocab/v6-prep/mbr-headroom.json`):

| chrF++ vs greedy | Yoruba | Hausa | Igbo |
|---|---|---|---|
| MBR choice of 9 | +1.7 (−2.0, +5.4) | +0.7 (−0.3, +1.8) | +0.9 (−0.1, +1.9) |
| Best candidate by reference (upper bound) | +9.1 | +4.5 | +4.2 |

The gain is positive in every language and not yet significant in any
(all three together: +0.8 per sentence, interval −0.3 to +2.0). Better
replies exist among the candidates, but agreement between candidates finds
few of them. MBR is therefore an optional arm, not the main lever, and the `mbr` arm of
`cloud_vx_v6.sh` was not run.
