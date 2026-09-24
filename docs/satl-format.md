# Sparse-friendly weight format (`.satl`, version 1)

Written by `atlas-convert` from the bf16 safetensors. Read by `atlas-rt`.
Designed for TEAL-style **input sparsity**: a skipped input channel skips one
contiguous column (PROJECT-PLAN-B §3.2).

## File layout

| Offset | Size | Content |
|---|---|---|
| 0 | 4 | Magic `SATL` |
| 4 | 4 | Header length `H`, u32 little-endian |
| 8 | H | Header, UTF-8 JSON, right-padded with spaces |
| `D` | … | Data section. `D` = `8 + H` rounded up to a multiple of 4,096. Holds the tensor segments; gaps are zero bytes |

The `offset` of each tensor is relative to `D`, and is a multiple of 4,096. This
keeps every segment page-aligned, and lets the header be written without knowing
its own length in advance.

## Header

```json
{
  "format": "satl",
  "version": 1,
  "source": {"model": "NCAIR1/N-ATLaS", "files": [{"name": "model-00001-of-00004.safetensors", "sha256": "…"}]},
  "config": {
    "hidden": 4096, "ffn": 14336, "layers": 32, "n_heads": 32, "n_kv_heads": 8,
    "head_dim": 128, "vocab": 128256, "rms_norm_eps": 1e-5, "rope_theta": 500000.0,
    "rope_scaling": {"rope_type": "llama3", "factor": 8.0, "…": "…"},
    "rope_layout": "hf_rotate_half"
  },
  "tensors": [
    {"name": "layers.0.mlp.gate_up", "offset": 0, "length": 67108864,
     "kind": {"columns": {"count": 4096, "parts": [
        {"source": "model.layers.0.mlp.gate_proj.weight", "rows": 14336},
        {"source": "model.layers.0.mlp.up_proj.weight", "rows": 14336}],
      "codec": {"q4": {"group": 64}}, "column_bytes": 16128, "stride": 16384}}}
  ]
}
```

`rope_layout: hf_rotate_half` records that the q/k projections use Hugging Face's
permuted layout, so RoPE must rotate the halves `(x[i], x[i + d/2])`.

## Tensor kinds

### `columns`: column-major, input-sparse projections

Covers q/k/v (bundled), o, gate/up (bundled) and down. The weight `W[out, in]`
computes `y = W·x`. The segment holds `count = in` columns, `stride` bytes apart.
Column *j* concatenates the *j*-th column of each part, in order. For each part:

1. **Nibbles**, `rows / 2` bytes. Values are laid out in blocks of 32. In block *b*,
   byte *k* (0 ≤ *k* < 16) holds `q[32b + k]` in its low nibble and `q[32b + k + 16]`
   in its high nibble. So `AND 0x0F` gives elements 0–15, and `>> 4` gives 16–31.
2. **Params**, `rows / group × 4` bytes: little-endian f16 pairs `(scale, min)`, one
   pair per group of `group` consecutive output rows.

Dequantised weight: `w = q · scale + min`, with `q` in 0..15.

`group` must divide `rows` and be a multiple of 32. At `group` 64, a column costs
4.5 bits per weight. At `group` 32, it costs 5.0.

| Bundle (N-ATLAS, group 64) | Parts (rows) | `column_bytes` | `stride` |
|---|---|---|---|
| `attn.qkv` | q 4096, k 1024, v 1024 | 3,456 | 3,456 |
| `attn.o` | o 4096 | 2,304 | 2,304 |
| `mlp.gate_up` | gate 14336, up 14336 | 16,128 | **16,384** (4 pages, for streaming) |
| `mlp.down` | down 4096 | 2,304 | 2,304 |

The gate/up bundles are padded to whole pages, so a streamed bundle is exactly
four 4 KiB reads. This costs 1.6% more space. The down columns are not padded, so
streaming them reads at page granularity, 1.78 times the bytes needed. Pairing
columns is a v2 option if streaming passes Gate G1.

### `rows`: row-major, dense

Covers `embed` (read one row per token) and `lm_head` (fully read every token).
The segment holds `count` rows of `cols` values, `stride` bytes apart. Codec
`q8 {group: 32}`: each block is a little-endian f16 `scale` followed by 32 × i8,
so `w = q · scale`, at 8.5 bits per weight.

### Codec `f32` (reference files only)

`"codec": "f32"` stores `columns` and `rows` tensors as plain little-endian f32,
with the same column-major or row-major order and no groups. It is lossless. It
exists so the runtime's forward pass can be checked against PyTorch exactly
(`crates/atlas-rt/tests`). It is never used for deployment.

### `vector`: f32

Covers RMSNorm weights: `len` little-endian f32 values.

## Tensor names

`embed`, `final_norm`, `lm_head`, and per layer *l*: `layers.{l}.attn_norm`,
`layers.{l}.attn.qkv`, `layers.{l}.attn.o`, `layers.{l}.mlp_norm`,
`layers.{l}.mlp.gate_up`, `layers.{l}.mlp.down`.
