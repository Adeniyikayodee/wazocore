# v7 plan

Written and committed before any v7 output exists; the commit date is the
registration. A change made after the first result is logged under
Deviations with its date and reason, never edited in place.

v7 trains no new release. It answers the five questions that
`METHOD.md` section 6 leaves open, with `scripts/cloud_vx_v7.sh`.

## Questions

1. **Does the release hold on text no decision has seen?** Every v6 arm
   was scored on the same 700 sentences per language, and the release was
   chosen on them.
2. **What does the vocabulary cost at a matched budget?** v6's `control`
   trained on 6M tokens and the release on 12M, so the −1.0 chrF++ gap in
   Hausa and Igbo mixes the vocabulary with the budget.
3. **Does the model still understand the languages?** New input rows are
   where comprehension could break, and only English → X has been scored.
4. **Which parts of the method matter**, and does it beat the standard
   recipe?
5. **Does English get worse?**

## Test data

- **Held-out translation.** NTREX-128 lines 500–1996 (0-based; 1,497
  sentences per language), English → X and X → English. Lines 0–499 made
  up the v6 gate; no decision has used these. Each model is scored on them
  once, and nothing is chosen by them.
- **Bits per character** on all 1,012 FLORES-200 devtest sentences of
  Yoruba, Hausa, Igbo and English. Tokenizer-independent, so it compares
  models with different vocabularies; English is question 5.
- **IrokoBench** through lm-evaluation-harness 0.4.13: AfriXNLI, AfriMMLU
  and AfriMGSM, prompt 1, zero-shot, chat template applied, in Yoruba,
  Hausa, Igbo and English. Original, release and `control-s0` only.

## Models

All train on v6's `gold` data (hashes in `results-v7/inputs.sha256`) for
12M tokens, with the same trainable rows and layers. Base models are pinned:
N-ATLaS at `e294476`, the release at `be50a5d` with its weight hash checked.

| Model | Differs from the release by | Answers |
|---|---|---|
| `n-atlas` | The original | 1, 3, 5 |
| `release` | Nothing (v6 `long`, seed 0) | 1, 3, 5 |
| `long-s1`, `long-s2` | Seed 1, seed 2 | Seed noise |
| `control-s0..s2` | No new tokens; seeds 0–2 | 2 |
| `no-canon` | No penalty on pieces the tokenizer would merge | 4 |
| `no-kl` | No anchor to the original | 4 |
| `old-anchor` | Anchor on the original's tokenisation, not projected | 4 |
| `no-calib` | New input rows not rescaled to their pieces' norm | 4 |
| `fvt` | Plain-mean rows, plain continued training: the standard recipe | 4 |

## Analysis

Fixed now, run on the laptop from the pulled files.

- **Metric.** chrF++ (sacrebleu, word order 2, NFC), per language; 5,000
  bootstrap rounds over sentences; 95% intervals.
- **Primary (questions 1 and 2), English → X on the held-out lines.**
  Release against the original: non-inferior if the one-sided 95% bound
  is above −1.0 chrF++, the v6 margin. Release against control: the three
  release seeds and the three control seeds, each side averaged over its
  seeds within every bootstrap round; the difference and its interval are
  the cost of the vocabulary.
- **Secondary.** X → English chrF++; AfriCOMET (`masakhane/africomet-stl-1.1`)
  in both directions; IrokoBench accuracy, paired over items; BPC; loops,
  invalid UTF-8 and tokens against the tokenizer's count, as in v6.
- **Ablations.** Each arm against the release, seed 0 against seed 0. A part
  matters if the interval excludes zero and the difference is larger than
  the spread of the three release seeds.
- **English.** BPC and X → English for original, release and control.
- **Multiplicity.** Every comparison is reported; none is dropped. Intervals
  are not adjusted, and only the primary results are headline claims.

## Run

On one A100 80GB, in priority order, so a stopped budget still answers the
first questions. Training rates are from the v6 logs.

| Unit | GPU time |
|---|---|
| `base`: evaluate original and release | about 2 h |
| `control`: `control-s0` | about 2.5 h |
| `fvt` | about 1.5 h |
| `ablate`: four arms | about 6 h |
| `seeds`: two release, two control | about 6.5 h |
| `comet` | under 1 h |

About 20 GPU hours in total, plus restarts. See `docs/cloud-vx.md`.

## Deviations

None yet.
