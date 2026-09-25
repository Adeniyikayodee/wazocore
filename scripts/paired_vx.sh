#!/usr/bin/env bash
# Paired words/s, original vs vocabulary-expanded N-ATLaS in llama.cpp
# (PROJECT-PLAN-B §14, V4). Both servers use the same binary and flags and
# stay loaded; prompts alternate between them (atlas_trace.paired_vx).
#
#   scripts/paired_vx.sh LABEL THREADS [PROMPTS] [NGL]
set -euo pipefail

label=${1:?label}; threads=${2:?threads}; prompts=${3:-20}; ngl=${4:-0}
root=$(cd "$(dirname "$0")/.." && pwd)
bin=$(dirname "$(command -v llama-server)")
orig="$root/data/models/N-ATLaS.Q4_K_M.gguf"
vx="$root/data/models/gguf-vx/N-ATLaS-vx.Q4_K_M.gguf"
out="$root/results/vocab"; mkdir -p "$out"

start() {  # model port
    "$bin/llama-server" -m "$1" -t "$threads" -c 2048 -np 1 -ngl "$ngl" --no-repack \
        --port "$2" --no-webui > "$out/$label-server-$2.log" 2>&1 &
    echo $!
}
p1=$(start "$orig" 8088); p2=$(start "$vx" 8089)
trap 'kill $p1 $p2 2>/dev/null || true' EXIT
for port in 8088 8089; do
    until curl -sf "http://127.0.0.1:$port/health" > /dev/null; do sleep 1; done
done

meta=$(mktemp)
cat > "$meta" <<JSON
{"label": "$label", "engine": "llama.cpp", "llama_cpp_version": "$("$bin/llama-server" --version 2>&1 | head -1)",
 "threads": $threads, "ngl": $ngl, "ctx": 2048, "flags": "--no-repack -np 1",
 "original": "$(shasum -a 256 "$orig" | cut -c1-16)", "expanded": "$(shasum -a 256 "$vx" | cut -c1-16)"}
JSON
uv run --quiet --project "$root/atlas-trace" python -m atlas_trace.paired_vx \
    --flores "$root/data/flores200_dataset" --prompts "$prompts" --meta "$meta" \
    --out "$out/$label-paired.json"
