# v8 plan

Written and committed before any v8 output exists; the commit date is the
registration. A change made after the first result is logged under
Deviations with its date and reason, never edited in place.

v7 found that the release loses inference and maths accuracy on IrokoBench,
about half from the training mixture and half from the vocabulary
(`METHOD.md`, v7 results). The mixture holds translation and Wikipedia only.
v8 tests the remedy the paper names, general instruction data, with
`scripts/cloud_vx_v8.sh`. It trains no release unless the decision rules
below say so.

## Questions

1. **Does general instruction data recover the reasoning the release
   loses?** Primary.
2. **Does translation hold** when instruction data takes part of the
   translation budget?
3. **Does the vocabulary still cost reasoning** once the mixture includes
   instruction data?
4. **How much do IrokoBench scores vary between seeds?** v7 scored one seed
   per model, so its IrokoBench differences have no seed spread.

## Instruction data

- **Source.** Aya Dataset (`CohereLabs/aya_dataset`, Apache-2.0), human-
  written prompts and answers. Train split, pinned by revision in
  `results-v8/inputs.sha256`. Rows checked on 31 October 2026: Yoruba
  11,758, Hausa 3,512, Igbo 1,534, English 3,944.
- **Why Aya.** Openly licensed, written by native speakers rather than
  machine-translated, and covers all four languages.
- **No maths or benchmark-shaped data.** No GSM8K, XNLI or MMLU training
  text, so AfriMGSM and AfriXNLI measure transfer, not practice on the
  test format.
- **Contamination check, before training.** After NFC and lower-casing, a
  row is dropped if it shares any 13-word sequence with an IrokoBench item
  (AfriXNLI, AfriMMLU, AfriMGSM, all four languages), FLORES-200 dev or
  devtest, or NTREX-128. Exact duplicates are dropped. Dropped counts are
  reported.
- **Format.** Each row becomes one user/assistant conversation in the
  chat template, like the translation data.

## Mixture

The release recipe, with one change: instruction conversations make up
**20% of training tokens**, taken from the translation conversations.

| | Release (v6 `long`) | v8 `inst` |
|---|---|---|
| Translation conversations (original's replies + SMOL) | 85% | 65% |
| Instruction conversations (Aya) | 0% | 20% |
| Wikipedia (careful spelling) | 15% | 15% |

20% is the midpoint of the 15 to 25% range we considered. It is fixed
here and not swept. Shares are by
tokens: the trainer cuts the shuffled conversation stream into 512-token
windows, so instruction tokens are 20/85 of that stream. The data step
logs the share actually consumed. If Aya yields fewer tokens than the share
needs after filtering, it is repeated, and the number of repeats is
reported.

Everything else is the release's: v6 `gold` translation data (same hashes
as v7), 12M tokens, seed 0, same rows and layers trained, same
hyperparameters, same pinned base weights.

## Models

| Model | Differs from the release by | Answers |
|---|---|---|
| `inst` | 20% instruction data | 1, 2 |
| `control-inst` | 20% instruction data, no new tokens | 3 |
| `long-s1` | Seed 1 (retrained; IrokoBench only) | 4 |

The release, `control-s0` and the original are reused from v7: their
held-out samples, AfriCOMET scores and IrokoBench logs are already in
`results/vocab/v7`. They are not rerun.

## Test data

The same as v7, unchanged:

- **IrokoBench** through lm-evaluation-harness 0.4.13: AfriXNLI, AfriMMLU
  and AfriMGSM, prompt 1, zero-shot, chat template, Yoruba, Hausa, Igbo
  and English. AfriMGSM with the `flexible-extract` filter (v7 deviation,
  29 October).
- **Held-out translation.** NTREX-128 lines 500 to 1996, both directions.
  v7 has already reported on these lines, but v8 chooses nothing among
  variants on them: there is one preregistered arm per question.
- **Bits per character** on FLORES-200 devtest, all four languages.

## Analysis

Fixed now, run on the laptop from the pulled files, with the v7 analysis
code where it applies.

- **Primary endpoint (question 1).** Reasoning score *R*: accuracy over the
  AfriXNLI and AfriMGSM items in Yoruba, Hausa and Igbo pooled (2,550
  items), paired over items, 5,000 bootstrap rounds, 95% intervals.
  AfriMMLU is left out of *R* because the original scores near chance in
  these languages (v7).
- **Recovery.** `inst` recovers reasoning if `inst` − release on *R* has an
  interval above zero **and** exceeds the difference between release seeds
  0 and 1 on *R*. The share of the release's loss recovered,
  (`inst` − release) / (original − release), is reported with its interval.
- **Translation holds (question 2).** `inst` against the release, chrF++ on
  the held-out lines: non-inferior if the one-sided 95% bound is above
  −1.0 in each language and direction (six tests), the v6 and v7 margin.
  `inst` against the original by the same rule, reported.
- **Vocabulary cost (question 3).** `inst` − `control-inst` on *R* and on
  chrF++, single seeds, judged by the v7 ablation rule: interval excludes
  zero and the difference exceeds the seed spread (*R*: question 4;
  chrF++: v7's three release seeds).
- **Seed spread (question 4).** Release seed 0 against `long-s1` on every
  IrokoBench task and language, reported in full.
- **Secondary.** Each IrokoBench task and language separately, English
  included; AfriMMLU; AfriCOMET both directions; bits per character; loops,
  invalid UTF-8 and piece spelling, as in v7.
- **Multiplicity.** Every comparison is reported; none is dropped.
  Intervals are not adjusted; only the primary results are headline claims.

## Exploratory check: reading with the original tokens

Not part of the decisions; reported as exploratory.

v7 split the release's AfriXNLI loss into training and vocabulary (Yoruba
−5.3, Igbo −4.8 from the vocabulary). The paper's account of the anchor is
that new input rows are where reading breaks. This check tests that
directly, with no training.

- **What.** The release, scored on AfriXNLI in Yoruba, Hausa, Igbo and
  English with the prompts and answer options encoded by the original
  N-ATLaS tokenizer. Every original id keeps its embedding in the release,
  so this is valid input; the new input rows are never used. The options
  are also scored in original tokens, which the release writes less often;
  the shift is shared by all options of an item, so it largely cancels in
  the choice.
- **Why AfriXNLI only.** It is scored by log-likelihood over fixed options,
  so no new token has to be generated. AfriMGSM generates text and AfriMMLU
  is near chance.
- **Comparison.** Release with original tokens against the release as v7
  scored it, and against `control-s0`, paired over items, 95% intervals.
- **Reading.** If accuracy moves most of the way from the release to
  `control-s0`, the vocabulary's cost is in reading the new tokens. If it
  does not move, the cost lies elsewhere and the paper's account is
  weakened. Either result is reported.
- **Cost.** About one GPU hour, run in the `inst` unit's pod before
  training starts.

## Decisions

Fixed now, applied as written.

| Recovery | Translation holds | Then |
|---|---|---|
| Yes | Yes | Adopt as the recipe. Stage 2: `inst` seeds 1 and 2 and `control-inst` seeds 1 and 2, so the claim rests on three seeds, as the release's does. A new public release is a separate decision. |
| Yes | No | Report the trade-off. No adoption, no stage 2. |
| No | Either | Report that 20% Aya does not recover reasoning. No stage 2. The paper's remedy stays untested beyond this. |

Stage 2 uses this plan's analysis unchanged, with seeds averaged within
each bootstrap round as in v7.

## Run

On one A100 80GB, in priority order, so a stopped budget still answers the
primary question. Times are from the v7 logs.

| Unit | Work | GPU time |
|---|---|---|
| `data` | Aya download, contamination filter, mixture (CPU) | minutes |
| `inst` | Exploratory check on the release, then train `inst`, held-out both ways, BPC, IrokoBench | about 3.5 h |
| `seed` | Train `long-s1`, IrokoBench only | about 2 h |
| `control` | Train `control-inst`, full evaluation | about 2.5 h |
| `comet` | AfriCOMET on the new held-out samples | under 1 h |

About 8 GPU hours, roughly $10 to $14 on Community Cloud. Stage 2, if
triggered, is about 9 more. See `docs/cloud-vx.md`.

## Deviations

- **31 October 2026, before any result: Aya order and tokenizer.** If Aya
  holds more tokens than the share needs, rows are taken in a fixed
  shuffled order (seed 0) until it is met. Shares are counted with the
  release's tokenizer; `control-inst` trains on the same file, as v7's
  control did, so its share under the original tokenizer differs
  slightly and is reported (`aya-report.json`). Written while building
  `scripts/v8_aya.py`; no v8 output existed.
