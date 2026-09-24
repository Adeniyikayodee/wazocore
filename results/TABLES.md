# Results

Generated from `results/` by `python -m atlas_trace.tables`. Do not edit by hand.

## Hardware (atlas-probe)

| Device | CPU | Read GB/s | Dense ceiling tok/s | Seq read MB/s |
|---|---|---|---|---|
| m4-macbook-dev | Apple M4 | 106.2 (10 thr) | 24.4 | 2262 |

## Tokenizer tax (FLORES-200 devtest)

| Lang | Tokens/word | Tokens vs en | NFD inflation |
|---|---|---|---|
| en | 1.24 | 1.00x | 1.000 |
| yo | 2.91 | 2.70x | 1.187 |
| ha | 2.13 | 1.97x | 1.000 |
| ig | 2.62 | 2.40x | 1.083 |

## llama.cpp baseline

| Run | Lang | Prefill tok/s | Decode tok/s | Words/s | Peak RSS GiB |
|---|---|---|---|---|---|
| m4-cpu-t6 | en | 26.4 | 8.14 | 6.03 | 6.63 |
| m4-cpu-t6 | yo | 24.9 | 7.35 | 2.74 | 6.63 |
| m4-cpu-t6 | ha | 27.2 | 7.50 | 3.40 | 6.63 |
| m4-cpu-t6 | ig | 26.3 | 7.69 | 3.11 | 6.63 |

## Int8 activations vs f32 (same q4 weights)

| Lang | Tokens | Mean KL | p95 KL | Top-1 | NLL f32 → int8 |
|---|---|---|---|---|---|
| eng | 160 | 7.81e-03 | 2.71e-02 | 91.9% | 2.9774 → 2.9607 |
| hau | 160 | 5.66e-03 | 2.45e-02 | 95.6% | 2.3914 → 2.3947 |
| ibo | 160 | 5.47e-03 | 2.31e-02 | 96.9% | 2.3585 → 2.3617 |
| yor | 160 | 5.08e-03 | 2.25e-02 | 96.2% | 2.6729 → 2.6658 |

## Dense format quality vs Q8_0

llama.cpp perplexity: n_ctx=256, 12 chunks, BOS per chunk, second half scored; FLORES-200 devtest; identical token ids verified

| Lang | PPL Q8_0 | PPL Q4_K_M | PPL .satl | ΔNLL Q4_K_M vs Q8_0 | ΔNLL .satl vs Q8_0 | .satl − Q4_K_M |
|---|---|---|---|---|---|---|
| en | 15.215 | 15.771 | 16.580 | +0.0359 [+0.0181, +0.0531] | +0.0859 [+0.0524, +0.1223] | +0.0500 [+0.0154, +0.0822] |
| yo | 16.151 | 16.287 | 16.282 | +0.0084 [-0.0181, +0.0363] | +0.0081 [-0.0252, +0.0447] | -0.0003 [-0.0342, +0.0344] |
| ha | 9.481 | 9.584 | 9.923 | +0.0109 [-0.0051, +0.0282] | +0.0456 [+0.0118, +0.0834] | +0.0347 [-0.0002, +0.0698] |
| ig | 11.778 | 12.019 | 11.948 | +0.0203 [-0.0010, +0.0403] | +0.0143 [-0.0202, +0.0521] | -0.0059 [-0.0462, +0.0342] |

## .satl conversion error vs bf16

| Tensor | Count | Mean rel RMSE | Max |
|---|---|---|---|
| embed | 1 | 0.0054 | 0.0054 |
| attn_norm | 32 | 0.0000 | 0.0000 |
| attn.qkv | 32 | 0.0973 | 0.1015 |
| attn.o | 32 | 0.0925 | 0.1003 |
| mlp_norm | 32 | 0.0000 | 0.0000 |
| mlp.gate_up | 32 | 0.0934 | 0.0967 |
| mlp.down | 32 | 0.0911 | 0.0929 |
| final_norm | 1 | 0.0000 | 0.0000 |
| lm_head | 1 | 0.0057 | 0.0057 |
| worst | 1 | 0.1015 | 0.1015 |
