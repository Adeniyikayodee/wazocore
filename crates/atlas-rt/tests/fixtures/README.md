# Test fixtures

`tiny/`: a random 2-layer Llama-3 (hidden 64, GQA 4/2, head_dim 16, llama3
RoPE scaling) with PyTorch reference logits for 40 tokens. Regenerate with:

```sh
F=crates/atlas-rt/tests/fixtures/tiny
uv run --project atlas-trace --group ref python -m atlas_trace.tiny_reference --out $F
target/release/atlas-convert convert --model-dir $F --out $F/tiny-f32.satl --model-name tiny-random-llama3 --f32
target/release/atlas-convert convert --model-dir $F --out $F/tiny-q4.satl --model-name tiny-random-llama3 --q4-group 32
```
