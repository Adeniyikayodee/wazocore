#!/usr/bin/env bash
# Paired words/s, original vs vocabulary-expanded N-ATLaS in llama.cpp
# (PROJECT-PLAN-B §14, V4). atlas_trace.paired_vx loads one model at a
# time and alternates which answers each sentence first.
#
#   scripts/paired_vx.sh LABEL THREADS [PROMPTS] [NGL] [EXPANDED.gguf]
set -euo pipefail

label=${1:?label}; threads=${2:?threads}; prompts=${3:-20}; ngl=${4:-0}
root=$(cd "$(dirname "$0")/.." && pwd)
orig="$root/data/models/N-ATLaS.Q4_K_M.gguf"
vx=${5:-$root/data/models/gguf-vx3/N-ATLaS-vx3.Q4_K_M.gguf}
out="$root/results/vocab"; mkdir -p "$out"

meta=$(mktemp)
cat > "$meta" <<JSON
{"label": "$label", "engine": "llama.cpp", "llama_cpp_version": "$(llama-server --version 2>&1 | head -1)",
 "threads": $threads, "ngl": $ngl, "ctx": 2048, "slots": 1, "one_model_in_memory": true,
 "original": "$(shasum -a 256 "$orig" | cut -c1-16)", "expanded": "$(shasum -a 256 "$vx" | cut -c1-16)"}
JSON
uv run --quiet --project "$root/atlas-trace" python -m atlas_trace.paired_vx \
    --original "$orig" --expanded "$vx" --threads "$threads" --ngl "$ngl" \
    --flores "$root/data/flores200_dataset" --prompts "$prompts" --meta "$meta" \
    --out "$out/$label-paired.json"
