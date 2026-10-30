# WazoCore: method and results

WazoCore expands N-ATLaS's vocabulary with 15,000 Yoruba, Hausa and Igbo tokens
and trains the model to use them. This document gives the release recipe, the
evaluation protocol, every reported number with the file it comes from, the
known defects, and the limits of the evidence. Raw results are in `results/`.

The release is v6. Earlier versions, their results and the approaches we
rejected are in [HISTORY.md](HISTORY.md).

## 1. The problem

N-ATLaS uses Llama-3's 128,256-token vocabulary, built mostly for English. On
FLORES-200 devtest (`results/tokenizer/`) the same sentences need:

| Language | Tokens per word | Tokens vs English |
|---|---|---|
| English | 1.24 | 1.00× |
| Hausa | 2.13 | 1.97× |
| Igbo | 2.62 | 2.40× |
| Yoruba | 2.91 | 2.70× |

WazoCore's tokenizer lowers the tax on the same sentences to 1.25× (Hausa),
1.42× (Igbo) and 1.66× (Yoruba), and English moves by 0.2%
(`results/tokenizer/wazocore-v6-flores200-devtest.json`).

An inference engine produces a roughly fixed number of tokens per second, so
Yoruba text arrives at about a third of English's speed and costs about 2.7
times as much to serve. We first tried to make the engine faster (a Rust CPU
runtime, TEAL activation sparsity, speculative decoding). llama.cpp remained
about three times faster than our kernels, so we closed that line; its code is
preserved under the `pre-cleanup` git tag. Reducing the number of tokens helps
every engine at once, which is the approach taken here.

## 2. Method

The release is built by `ARMS=long bash scripts/cloud_vx_v6.sh` on one A100
80GB ([cloud-vx.md](cloud-vx.md)).

1. **Vocabulary** (`vocab_expand`). Text is first segmented by the original
   tokenizer; 15,000 BPE merges are then learned on top, jointly for Yoruba,
   Hausa and Igbo, from 4M characters per language (Wikipedia plus the FLORES
   dev split). Merges whose result is not whole UTF-8 are rejected. Every
   original token keeps its id and its embedding. Text is re-segmented only
   where adjacent original pieces form a new merge, and 9,499 of the new
   tokens are plain ASCII, so English changes slightly: on 355,000 words of
   English prose, 404 of 31,875 lines tokenise differently and the token
   count falls by 0.12% (`scripts/english_tokenisation.py`,
   `results/tokenizer/english-v6.json`).
2. **Initialisation** (`vx_torch expand --calibrate-input-norm --out-weights
   chars`). A new token's input embedding is the mean of its original pieces'
   embeddings, rescaled to the pieces' norm; without the rescaling the new
   rows start and stay at about 55% of it. Its output embedding is the
   length-weighted mean of the pieces' rows, following arXiv 2608.03494.
3. **Training data.**
   - *The original's own replies* (`vx_torch distill`). Greedy replies of the
     original model to about 24,000 translation requests, English to and from
     each language, with sources drawn from Wikipedia.
   - *Human translations* (`vx_torch smol`). google/smol (CC BY 4.0,
     professional translations made for training) gives 23,662 sentence
     pairs: Yoruba 8,647, Hausa 8,653, Igbo 6,362. None of its English
     sources occurs in FLORES-200 dev or devtest or in NTREX-128.
   - *Wikipedia* in the three languages, filtered for careful spelling.

   All of it is tokenised with the new vocabulary. No FLORES or NTREX text is
   trained on.
4. **Training** (`vx_torch train`). The new embedding rows and the bottom and
   top two of the 32 transformer layers (arXiv 2406.11477) train on 12M
   tokens, about 85 minutes:
   - 85% replies and human translations, 15% Wikipedia text;
   - *canonical anchor* (`--anchor canonical`, weight 1.0). A KL term to the
     frozen original on the training text itself. The original reads that
     text in its own tokens (the new tokenisation decomposes into the
     original one exactly: 0 differences on 56,554 Wikipedia lines), and the
     student's distribution is projected onto the original vocabulary by
     counting each new token for its first original piece. The original's
     probability for a piece that would complete a merge with the previous
     token is removed first, following arXiv 2512.14954. This anchor cannot
     reward spelling a word in pieces when it has a token of its own;
   - *canonical penalty* (`--canon-weight 1.0`). An unlikelihood term (arXiv
     1908.04319) on any token that would form a pair the tokenizer merges.
     Text from the tokenizer contains such a pair 13 times in 1.2M tokens;
   - sequence length 512, effective batch 16, learning rate 5e-5 with warm-up
     and cosine decay, gradient clipping at 1.0;
   - fp32 master weights for trained tensors, bf16 for frozen layers, and a
     gradient mask that keeps every original vocabulary row exact.
5. **Export** (`scripts/cloud_gguf.sh`). bf16 weights to GGUF to Q4_K_M with
   llama.cpp, with no importance matrix. The converter identifies
   pre-tokenizers by a fingerprint that the new merges change; after
   confirming the pre-tokenizer is byte-identical, the fingerprint is mapped
   to `llama-bpe`.

## 3. Evaluation protocol

Every table in sections 4 and 5 follows one of these four protocols.

**Quality at full precision.** Greedy English → X translation with the bf16
weights in PyTorch, scored with chrF++ against human references, both sides
normalised to Unicode NFC. Test sets are NTREX-128 (the first 500 news
sentences per language) and FLORES-200 devtest (200), 700 sentences per
language. Differences from the original carry 95% paired-bootstrap intervals
(1,000 resamples; 5,000 for a pooled row), computed by `vx_torch pooled`. The
release recipe was trained twice with different seeds.

**Quality of the shipped file.** Both Q4_K_M files answer the same prompts in
llama.cpp (build 9430, greedy, at most 160 tokens) on the first 250 NTREX-128
sentences per language, one model in memory at a time
(`scripts/gguf_quality.py`, scored by `vx_torch score`).

**Speed.** llama.cpp build 9430 on one Apple M4 laptop, Q4_K_M, 4 threads, 20
FLORES-200 devtest translations per language, at most 256 tokens, no prompt
cache (`scripts/paired_vx.sh`). Both models answer each sentence back to
back with the first-mover alternating, and only one model is in memory at a
time. The quantity compared is words and characters of reply per second of
decoding; whole-request time adds the time to read the prompt.

**Defects.** A *loop* is a unit of 2 to 6 characters repeated six times back
to back (`vx_core.looping`). A *broken character* is invalid UTF-8 inside a
reply: half of a multi-byte letter followed by something else. llama-server
answers such a request with an error.

**Arms.** Each arm of `cloud_vx_v6.sh` changes one thing and is scored on the
same 700 sentences per language:

| Arm | What it is |
|---|---|
| `fix` | The previous release's data (the original's replies only), with steps 2 and 4's canonical changes, 6M tokens |
| `gold` | `fix` plus the human translations |
| `control` | The original model trained on `gold`'s data, with no new tokens |
| `small` | `gold` with the first 5,000 merges, which carry 83% of new-token use |
| `long` | `gold` with twice the training, 12M tokens. The release |

**Gate.** A version becomes the release only if, in the pooled estimate, no
language is below the original (interval containing zero or above it), new
tokens are used at least as much as in the previous release, generated
tokens are within 1% of the tokenizer's own count, and loops do not exceed
the original's. A gain over the original counts as a gain of the vocabulary
work only by the margin over `control`; if `control` also rises, the honest
claim is "faster, and better than N-ATLaS thanks to the added data".

## 4. Results

### 4.1 Quality at full precision

chrF++ change against the original with 95% intervals (`results/vocab/v6/`).
Token ratio is characters per token on the NTREX replies, relative to the
original. The v4 row is the previous release ([HISTORY.md](HISTORY.md),
section 3).

| Arm | Yoruba | Hausa | Igbo | Token ratio (yo, ha, ig) |
|---|---|---|---|---|
| v4 (both runs) | +0.6 (−0.1, +1.2) | −0.6 (−1.1, −0.0) | −1.1 (−1.7, −0.5) | 1.54, 1.41, 1.62 |
| `fix` | −0.2 (−0.9, +0.6) | −0.6 (−1.3, −0.0) | −0.8 (−1.5, −0.1) | n/a |
| `gold` | +0.4 (−0.5, +1.3) | −0.4 (−1.0, +0.2) | −0.7 (−1.4, +0.1) | 1.38, 1.38, 1.55 |
| `control` (no new tokens) | +2.3 (+1.6, +3.1) | +0.6 (+0.1, +1.1) | +1.2 (+0.6, +1.8) | 0.96, 0.99, 1.00 |
| `small` (5,000 tokens) | +0.7 (−0.1, +1.5) | −0.2 (−0.8, +0.3) | −0.2 (−0.8, +0.3) | 1.39, 1.31, 1.47 |
| **`long`** (gold, 12M tokens), run 1 | +2.3 (+1.6, +3.1) | −0.4 (−0.9, +0.2) | +0.3 (−0.3, +1.0) | 1.51, 1.43, 1.60 |
| **`long`**, run 2 | +2.0 (+1.2, +2.8) | −0.4 (−0.9, +0.2) | +0.1 (−0.5, +0.7) | n/a |
| **`long`**, both runs pooled | **+2.2 (+1.5, +2.9)** | **−0.4 (−0.8, +0.1)** | **+0.2 (−0.4, +0.8)** | |

- The human translations lift the original in every language (`control`).
  With the new vocabulary and 6M tokens the gain does not arrive: `gold` is
  1.9, 1.0 and 1.8 below `control` (intervals exclude zero).
- The new rows were under-trained. A third of the vocabulary (`small`) or
  twice the training (`long`) both remove the loss; `long` keeps v4's token
  saving. With 12M tokens the model passes non-inferiority at a 1.0 margin in
  all three languages.
- Against `control`, paired and pooled over both `long` runs, the vocabulary
  costs −0.2 (−0.9, +0.5) in Yoruba, −1.0 (−1.5, −0.4) in Hausa and
  −1.0 (−1.6, −0.4) in Igbo (`results/vocab/v6/pooled-long-vs-control.json`).
  The cost is real in Hausa and Igbo; against the original it is hidden
  because the added data lifts both models. On matched budgets and three
  seeds, see 4.4.
- The canonical changes did not remove the piece spelling by themselves
  (`fix` and `gold` write 4 to 7% more tokens than the tokenizer would, like
  v4); `small` and `long` write 2 to 4% more. Forcing v4 onto the canonical
  path at decoding time costs quality (−0.4, −1.4, −1.2), so for v4 the piece
  spelling is a fallback, not waste.

### 4.2 The shipped file

The plain Q4_K_M file of `long` (run 1) against the original's Q4_K_M file
(`results/vocab/gguf-quality/summary-v6-long.json`):

| chrF++ | Yoruba | Hausa | Igbo |
|---|---|---|---|
| Original, Q4_K_M | 28.57 | 46.64 | 42.79 |
| `long`, Q4_K_M | 30.31 | 46.38 | 42.69 |
| Difference | +1.7 (+0.3, +3.1) | −0.3 (−1.1, +0.6) | −0.1 (−1.4, +1.2) |
| vs the v4 Q4_K_M file | +1.2 (−0.0, +2.4) | +0.9 (+0.2, +1.7) | +2.8 (+1.5, +4.2) |
| Tokens per word | 3.02 → 2.09 | 2.05 → 1.42 | 2.61 → 1.59 |

No request failed in 750, and no special tensor layout is needed: unlike v4,
`long` holds its quality at 4 bits (bf16 to Q4_K_M: Hausa −0.1, Igbo −0.8
with an interval of −1.8 to +0.2).

### 4.3 Speed

**CPU** (`results/vocab/m4-cpu-v6-paired.json`):

| Language | Words/s, original → WazoCore | Speed-up, words | Speed-up, characters | Tokens per word |
|---|---|---|---|---|
| Yoruba* | 2.73 → 4.45 | 1.63× | 1.58× | 2.87 → 1.91 |
| Hausa | 3.77 → 6.09 | 1.61× | 1.49× | 2.12 → 1.34 |
| Igbo | 3.06 → 5.24 | 1.72× | 1.72× | 2.53 → 1.56 |
| English** | 6.23 → 6.60 | 1.06× | 1.07× | 1.28 → 1.31 |

\*19 sentences: one WazoCore reply contained a broken character and the
server returned an error. \*\*19 sentences: one request to the original
stalled for 82 s on the host; with it the English figure is 1.99×. Per-prompt
geometric means with 95% intervals are 1.49 (1.32, 1.67) for Yoruba, 1.53
(1.44, 1.61) for Hausa and 1.74 (1.59, 1.94) for Igbo. Decoding speed in
tokens per second is about 8 for both models, so the gain comes from the
token count.

**Whole request, CPU.** The same run, counting the time to read the prompt as
well as the time to write the reply. Mean seconds per request, original →
WazoCore, with the same exclusions:

| Language | Reading the prompt | Writing the reply | Whole request | Speed-up | Words per reply |
|---|---|---|---|---|---|
| Yoruba | 6.0 → 5.6 | 11.8 → 8.4 | 17.9 → 14.0 | 1.28× | 32 → 37 |
| Hausa | 5.8 → 5.6 | 8.3 → 6.6 | 14.1 → 12.2 | 1.16× | 31 → 40 |
| Igbo | 5.8 → 5.7 | 9.7 → 5.6 | 15.5 → 11.3 | 1.38× | 30 → 29 |
| English | 10.0 → 7.2 | 4.6 → 4.2 | 14.6 → 11.3 | 1.29× | 29 → 27 |

The prompts into Yoruba, Hausa and Igbo are in English (about 83 tokens with
the chat template), so the new vocabulary does not shorten them and the time
to read them is unchanged. The English row translates from Yoruba: its prompt
falls from 142 tokens to 108. The two models' replies differ in length, so
this table compares requests and the table above compares speed per word.

**GPU.** The same benchmark with every layer on the laptop's GPU through
Metal (`results/vocab/m4-metal-v6-paired.json`):

| Language | Words/s, original → WazoCore | Speed-up, words | Speed-up, characters | Whole request, s | Speed-up, whole request | Tokens per word |
|---|---|---|---|---|---|---|
| Yoruba* | 3.26 → 5.43 | 1.66× | 1.63× | 12.0 → 8.5 | 1.42× | 2.93 → 1.93 |
| Hausa | 4.71 → 7.00 | 1.49× | 1.46× | 8.5 → 6.2 | 1.36× | 2.10 → 1.39 |
| Igbo | 4.00 → 6.18 | 1.54× | 1.51× | 9.9 → 6.3 | 1.56× | 2.64 → 1.58 |
| English | 8.22 → 7.77 | 0.95× | 0.94× | 5.8 → 5.4 | 1.08× | 1.30 → 1.31 |

\*19 sentences: the sentence that failed in the CPU run failed again with the
same broken character.

The laptop slowed under sustained load. Both models wrote 17 to 20 tokens per
second for the first two sentences and 8 to 12 for the other eighteen, and
read prompts at 95 to 138 tokens per second and then 40 to 60. They slowed
together and took turns going first, so the ratios stand. The absolute
figures describe back-to-back requests; a single request on a cool machine
runs at about 20 tokens per second. The pattern is that of thermal
throttling; no temperature was recorded.

### 4.4 Held-out test, matched control and ablations (v7)

Preregistered in [PLAN-v7.md](PLAN-v7.md) before any output existed. Every
model trains on `gold` for 12M tokens. Test: NTREX-128 lines 500 to 1996,
1,497 sentences per language that no decision has used, in both directions.
chrF++ differences with 95% intervals (`results/vocab/v7/analysis.json`).
AfriCOMET never contradicts a chrF++ difference whose interval excludes
zero, but does not confirm all of them; the exceptions are noted.

**The release on unseen text**, against the original:

| | Yoruba | Hausa | Igbo |
|---|---|---|---|
| English → X | +1.6 (+0.9, +2.2) | −0.3 (−0.8, +0.1) | +0.7 (+0.2, +1.2) |
| X → English | −0.7 (−1.2, −0.2) | −0.1 (−0.6, +0.4) | +1.3 (+0.8, +1.8) |

Non-inferior at 1.0 in five of six; Yoruba → English is not shown
non-inferior (one-sided bound −1.1). AfriCOMET does not confirm that
shortfall (−0.3, interval including zero) and puts Hausa → English above
the original (+0.5, +0.0 to +0.9).

**The cost of the vocabulary**, three release seeds against three control
seeds at the same budget, averaged per side:

| | Yoruba | Hausa | Igbo |
|---|---|---|---|
| English → X | +0.2 (−0.3, +0.6) | −1.2 (−1.6, −0.9) | −0.8 (−1.1, −0.5) |
| X → English | −1.0 (−1.3, −0.7) | −0.5 (−0.8, −0.2) | −0.4 (−0.7, −0.1) |

v6 measured −0.2, −1.0 and −1.0 at unequal budgets; the matched figure
confirms it. Control alone gains +0.9 to +1.8 over the original English → X:
the release's gains come from the data, and the vocabulary buys speed at
about one point.

**Ablations**, seed 0 against the release. The three release seeds differ by
at most 0.7 (Yoruba English → X) and 0.4 elsewhere.

| Removed | English → X (yo, ha, ig) | X → English (yo, ha, ig) |
|---|---|---|
| Projected KL anchor (`no-kl`) | +0.8, −0.6, +1.0 | −2.2, −3.2, −2.3 |
| Projection: anchor in the original's tokenisation (`old-anchor`)* | +1.0, −0.4, +0.7 | −1.8, −2.7, −2.1 |
| Canonical penalty (`no-canon`) | +0.4, +0.2, +0.1 | +0.1, +0.1, −0.1 |
| Calibrated input rows (`no-calib`) | +0.3, −0.1, −0.1 | −0.3, 0.0, −0.3 |
| All of it: plain-mean rows, no anchor (`fvt`) | +1.1, −0.4, +0.8 | −1.6, −2.7, −2.0 |

\*Half the micro-batch, same windows per step (PLAN-v7 deviations).

- The projected anchor is the part that matters: without it, or with the
  anchor in the original's tokenisation, X → English falls 1.6 to 3.2,
  5 to 15 times the seed spread. The standard recipe (`fvt`, after
  FVT) loses the same.
- In return those arms gain 0.7 to 1.0 in Igbo English → X, above the seed
  spread, and AfriCOMET confirms two of three. The Yoruba gains (0.8 to
  1.1) pass narrowly on chrF++ and AfriCOMET confirms none.
- The canonical penalty and calibration change nothing measurable.

**Comprehension and reasoning.** IrokoBench (AfriXNLI, AfriMMLU, AfriMGSM),
zero-shot, accuracy in points, paired over items, one seed. Training effect:
control − original. Vocabulary effect: release − control.

| | Training | Vocabulary |
|---|---|---|
| AfriXNLI yo, ha, ig, en | −4.5, −3.2, −4.3, −6.2 | −5.3, −1.5, −4.8, −1.7 |
| AfriMGSM yo, ha, ig, en | −12.0, −1.6, −9.2, +0.8 | −0.8, −5.2, −2.0, −6.0 |
| AfriMMLU | within ±1.6 | within ±3.0, intervals include 0 |

Intervals are 6 to 13 points wide on AfriMGSM (250 items), so its single
cells are weak. The pattern is not: inference and maths fall after training,
by about as much from the recipe as from the vocabulary; knowledge does not.
The training mixture is translation only.

**English.** Bits per character on FLORES devtest: original 1.228, release
1.015, control 1.005. English text is modelled better, not worse; English
reasoning (above) is the cost.

## 5. Known defects and workarounds

**Yoruba loops.** Mostly the vocabulary. On the 1,497 held-out sentences
(4.4): original 7, three control seeds 6, 9 and 13, three release seeds 19,
18 and 23. The data adds a few; the new tokens double them. v6 blamed the
data from two runs on 700 sentences (original 2, `control` 10, `long` 5 and
11); three seeds each overturn that. Most are runs of one combining mark.
Hausa and Igbo are unaffected.

**Broken characters.** Some original tokens are half of a multi-byte letter
(ṣ, ƙ and others). Occasionally a model writes the first half and then
something else. Counted in replies that ended normally, out of 2,100 per
model: the original 0, v4 3 and 1 (two runs), `gold` 20, `small` 0, `long` 2
and 2. The shipped-file check of `long` had none in 750; the speed benchmark
had one in 80. About one reply in a thousand for the release, the same rate
as v4, and worse when the new rows are under-trained.

**Workarounds on the shipped file.** Both defects have a remedy on the
client side, and both were measured on the Q4_K_M file in llama.cpp.

A repetition penalty of 1.1 in the sampler, on the same 250 NTREX-128
sentences per language (`SAMPLER` in `scripts/gguf_quality.py`;
`results/vocab/gguf-quality/summary-v6-long-penalty-vs-plain.json`):

| No penalty → penalty 1.1 | Yoruba | Hausa | Igbo |
|---|---|---|---|
| Loops | 4 → 2 | 0 → 0 | 0 → 0 |
| Replies cut off at the token limit | 9 → 2 | 0 → 0 | 3 → 0 |
| chrF++ | 30.31 → 31.23 | 46.38 → 45.78 | 42.69 → 43.42 |
| chrF++ change | +0.9 (−0.1, +1.9) | −0.6 (−1.3, +0.0) | +0.7 (−0.2, +1.6) |

The original's file loops 2, 0 and 1 times on these sentences. The two Yoruba
loops that remain are runs of combining marks, which this penalty does not
stop. Hausa has no loops to remove and 97% of resamples are worse, so the
penalty belongs on Yoruba requests only. One strength was tried, and the
loop count rests on four sentences.

A retry with sampling after a broken character (`scripts/retry_probe.py`;
`results/vocab/gguf-quality/retry-v6-long.json`). Of the five prompts that
gave a broken character in any earlier run, one fails on the shipped file
under greedy decoding: the first FLORES sentence into Yoruba, the failure in
both speed runs. The four found in the full-precision samples do not.
Retried 30 times, that prompt returns valid text 24 times at temperature 0.3
and 27 times at 0.7; four of the valid replies at 0.3 loop and none at 0.7.
One prompt shows that a retry usually works, not how often.

## 6. Limitations

- **Test-set reuse.** Section 4.1 chose the release on its 700 sentences.
  Section 4.4 scores 1,497 others per language that no decision used.
- **Seeds.** Three seeds for the release and control; the ablations and
  IrokoBench have one, judged against the release's seed spread.
- **Evaluation scope.** Translation is scored in news text (NTREX-128) and
  FLORES-200, both directions. Dialogue is not scored; native-speaker
  judgement is being collected with the blind A/B kit in `tester-kit/`.
- **Reasoning.** The release scores lower than the original on inference
  in every language, English included, and on maths in all four
  (English's interval reaches zero) (4.4). The training mixture has no general
  instruction data.
- **Hausa** is 0.3 to 0.4 chrF++ below the original English → Hausa in
  every test, with intervals that include zero; against matched control the vocabulary costs
  it 1.2.
- **Hardware.** Speed was measured on one laptop with 20 prompts per
  language, which shows a speed-up of about 1.5 to 1.7× but not its second
  decimal. The GPU run was throttled to about half the machine's cool speed
  (section 4.3); no other GPU was measured. llama.cpp and LM Studio are
  tested, Ollama is not yet.
- **Unicode form.** The tokenizer has no normaliser and the new tokens were
  learned on composed (NFC) text. On decomposed input (a letter followed by a
  combining accent, as in the NTREX-128 Yoruba references) the token saving
  falls from 1.63× to 1.15× for Yoruba and from 1.69× to 1.27× for Igbo.
  Normalise input to NFC before tokenising.
- **Vocabulary size.** The low-budget recipe we follow (arXiv 2406.11477) was
  validated with 100 to 1,000 new tokens; 15,000 is beyond what that paper
  tested, and more than half of our new tokens are rare in the training text.
  Section 4.1 compares both remedies: 5,000 tokens at the same training
  length, and 15,000 tokens at twice the length. Both remove the loss; the
  second keeps the full token saving.
- **Workarounds.** The repetition penalty was tried at one strength on four
  looping sentences, and the retry on one failing prompt (section 5).
- **Inherited behaviour.** WazoCore keeps N-ATLaS's biases, its 8,092-token
  context and the limits of its instruction tuning.

## 7. Reproducing the numbers

Paths are relative to the repository root. `V` is `uv run --project wazocore
--group ref python -m wazocore.vx_torch`; scoring also needs `sacrebleu`.

| Result | Command | Output |
|---|---|---|
| Tokens per word (section 1) | `uv run --with tokenizers python -m wazocore.tokens_per_word --flores data/flores200_dataset --split devtest --tokenizer TOKENIZER.json --out OUT.json` | `results/tokenizer/` |
| English tokenisation (section 2) | `uv run --project wazocore python scripts/english_tokenisation.py TOKENIZER.json NEW_TOKENS.json OUT.json TEXT...` on the released tokenizer, two Project Gutenberg novels (1342, 2701) and this repository's docs | `results/tokenizer/english-v6.json` |
| Training, samples and gate (4.1) | `ARMS="fix gold control small long" bash scripts/cloud_vx_v6.sh` on the GPU host; `SEED` repeats an arm for a second run | `results/vocab/v6/pooled-*.json` |
| Release against `control` (4.1) | `V pooled --base ntrex-control.json flores-control.json --run ntrex-long.json flores-long.json --run ntrex-long2.json flores-long2.json --out OUT.json` (files in `results/vocab/v6/`) | `results/vocab/v6/pooled-long-vs-control.json` |
| Token tax after expansion | `uv run --project wazocore python -m wazocore.tokens_per_word --flores FLORES_DIR --split devtest --tokenizer WAZOCORE_TOKENIZER.json --out OUT.json` | `results/tokenizer/wazocore-v6-flores200-devtest.json` |
| Shipped file (4.2) | `python scripts/gguf_quality.py 250 OUT NAME FILE.gguf`, then `V score OUT/ntrex-gguf-n-atlas.json OUT/ntrex-gguf-NAME.json --out OUT/summary.json` | `results/vocab/gguf-quality/` |
| Speed, CPU (4.3) | `scripts/paired_vx.sh LABEL 4 20 0 FILE.gguf` | `results/vocab/LABEL-paired.json` |
| Speed, GPU (4.3) | `scripts/paired_vx.sh LABEL 4 20 99 FILE.gguf` | `results/vocab/LABEL-paired.json` |
| Whole request (4.3) | Mean of `prompt_ms + gen_ms` over the paired samples in the two files above | n/a |
| v7 (4.4) | `bash scripts/cloud_vx_v7.sh` on the GPU host, `scripts/cloud_pull.sh` on the laptop, then `uv run --project wazocore --with sacrebleu python scripts/v7_analysis.py results/vocab/v7` | `results/vocab/v7/analysis.json` |
| Repetition penalty (5) | `SAMPLER='{"repeat_penalty": 1.1}' python scripts/gguf_quality.py 250 OUT NAME FILE.gguf`, then `V score` against the plain file | `results/vocab/gguf-quality/` |
| Retry (5) | `uv run --project wazocore python scripts/retry_probe.py FILE.gguf OUT.json` | `results/vocab/gguf-quality/retry-v6-long.json` |
