# Activation trace format (`.atr`, version 1)

Written by `atlas-trace` (Python, GPU), read by `atlas-sim` (Rust). One file
holds one language's trace under one sparsity profile.

## Layout

| Offset | Size | Content |
|---|---|---|
| 0 | 4 | Magic `ATRC` |
| 4 | 4 | Header length `H`, u32 little-endian |
| 8 | H | Header, UTF-8 JSON, right-padded with spaces so `8 + H` is a multiple of 8 |
| 8 + H | rest | Payload: u64 little-endian bitmap words |

## Header

```json
{
  "format": "atlas-trace",
  "version": 1,
  "model": "N-ATLaS",
  "lang": "yo",
  "source": "FLORES-200 dev",
  "profile": "teal-en-50",
  "layers": 32,
  "tokens": 20000,
  "matrices": [
    {"name": "gate_up", "channels": 4096, "bytes_per_channel": 16128},
    {"name": "down", "channels": 14336, "bytes_per_channel": 2304}
  ]
}
```

- `matrices` lists the sparsified inputs, in payload order. A channel is one
  **input** channel. For `gate_up`, gate and up share their input, so one channel is
  one `.satl` bundle: 2 × 14,336 × 4.5 bits = 16,128 bytes. For `down`, a channel is one
  column: 4,096 × 4.5 bits = 2,304 bytes.
- Attention inputs (`qkv`: 4,096 channels, `o`: 4,096) may be included for
  sparsity analysis. The simulator treats any matrix not marked as cacheable as
  resident in RAM.

## Payload

The payload is token-major: `for token, for layer, for matrix: ceil(channels / 64)` words.
Bit `c % 64` of word `c / 64` is set if and only if input channel `c` passed its
threshold (`|x_c| > t`) for that token. Bits past `channels` are zero. The payload
length must equal `tokens × layers × Σ ceil(channels / 64) × 8` bytes.

With the two MLP matrices, one token takes 73.7 KB, so 20,000 tokens take 1.47 GB.
