#!/usr/bin/env bash
# S4: quality of the shipped sparse path, speed curve, and llama.cpp comparison.
set -euo pipefail
cd "$(dirname "$0")/.."
M=data/models/n-atlas-q4g64s.satl; T=data/models/n-atlas-bf16/tokenizer.json
MAN=results/probe/m4-macbook-dev.json; OUT=results/s4; mkdir -p "$OUT"
S=$(mktemp -d)
for l in eng yor hau ibo; do head -40 "data/flores200_dataset/devtest/${l}_Latn.devtest" > "$S/$l.txt"; done
echo "== quality: dense vs sparse decode (30%)" >&2
for l in eng yor hau ibo; do
    target/release/atlas-rt --threads 6 --manifest $MAN compare --profile profiles/en-30.json --model $M --tokenizer $T \
        --text "$S/$l.txt" --label "$l" --tokens 512 --out "$OUT/quality-$l-en30.json" 2>&1 | grep -v "tokens in"
done
echo "== speed curve (paired, P-cores)" >&2
for p in 10 20 30 40; do
    target/release/atlas-rt --threads 6 --manifest $MAN bench --model $M --profile profiles/en-$p.json --tokens 48 --warmup 4 --out "$OUT/speed-pcore-en$p.json"
done
echo "== vs llama.cpp (interleaved rounds)" >&2
uv run --quiet --project atlas-trace python -m atlas_trace.vs_llamacpp --profile profiles/en-30.json --manifest $MAN --out "$OUT/vs-llamacpp-pcore-en30.json"
uv run --quiet --project atlas-trace python -m atlas_trace.vs_llamacpp --profile profiles/en-30.json --ecores --manifest $MAN --out "$OUT/vs-llamacpp-ecore-en30.json"
