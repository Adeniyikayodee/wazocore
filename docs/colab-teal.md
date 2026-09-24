# Running TEAL calibration on Colab (P1, Gate G1)

`atlas-trace/atlas_trace/teal.py` needs no other repository files. It needs a
GPU with **at least 24 GB** for bf16 N-ATLaS: an **L4 or A100**. A T4 (16 GB) is
too small.

## Cells

```python
# 1. Setup: runtime > change runtime type > L4 or A100
!pip -q install torch transformers sacrebleu numpy
from huggingface_hub import login; login()          # read token with N-ATLaS access
# Upload atlas-trace/atlas_trace/teal.py to the session (Files pane), then:
!nvidia-smi --query-gpu=name,memory.total --format=csv
```

```python
# 2. Calibrate: thresholds for en, yo, ha, ig, pooled at 25/40/50% (a few minutes)
!python teal.py calibrate --out work/profiles
```

```python
# 3. Evaluate the calibration x evaluation matrix (C1/C2)
#    40 chunks x 511 tokens per language; about 5 min on A100, about 12 on L4
!python teal.py evaluate --profiles work/profiles --out work/eval.json
```

```python
# 4. Traces for atlas-sim: calibration (dev) and evaluation (devtest), about 1.5 GB each
for lang in ["en", "yo", "ha", "ig"]:
    !python teal.py traces --profile work/profiles/{lang}-50.json --lang {lang} --split dev     --out work/traces/{lang}-{lang}50-dev.atr
    !python teal.py traces --profile work/profiles/{lang}-50.json --lang {lang} --split devtest --out work/traces/{lang}-{lang}50-devtest.atr
# English thresholds applied to Yoruba: the C1 condition for the cache study
!python teal.py traces --profile work/profiles/en-50.json --lang yo --split devtest --out work/traces/yo-en50-devtest.atr
```

```python
# 5. Optional: chrF++ on a subset of profiles (generation is slow; limit the set)
!mkdir -p work/chrf_profiles && cp work/profiles/en-50.json work/profiles/yo-50.json work/profiles/ha-50.json work/profiles/ig-50.json work/chrf_profiles/
!python teal.py translate --profiles work/chrf_profiles --n 50 --out work/chrf.json
```

```python
# 6. Save everything (Drive, or download work.zip)
!zip -qr work.zip work
```

## Bringing results back

- `work/profiles/*.json` go to `profiles/` in the repository.
- `work/eval.json` and `work/chrf.json` go to `results/teal/`.
- Traces stay out of git (1.5 GB each). Keep them in `data/traces/`, then run:
  `target/release/atlas-sim run --trace data/traces/yo-yo50-devtest.atr --calib data/traces/{en,yo,ha,ig}-*-dev.atr --cache-frac 0.5 --probe <pi5 manifest>`

## What the outputs answer

| Output | Question |
|---|---|
| `eval.json`, rows with `calib_lang = en` | **C1:** do English thresholds give yo/ha/ig a different effective sparsity and larger KL? |
| `eval.json`, diagonal vs the English row | **C2:** does per-language calibration close the gap at equal target sparsity? |
| `eval.json`, KL against target sparsity | Per-language sparsity choice at the 0.05-nat gate (§4.3) |
| traces + atlas-sim | **G1:** storage streaming in or out |

**Not yet covered:** AfriMMLU (IrokoBench) knowledge retention, §4.1. That's the next addition to `teal.py`.
