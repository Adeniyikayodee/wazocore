#!/usr/bin/env bash
# Real llama.cpp words/s with and without speculative drafting (one config).
#   scripts/spec_experiment.sh LABEL PROMPTS [extra llama-server args...]
set -euo pipefail
label=$1; prompts=$2; shift 2
root=$(cd "$(dirname "$0")/.." && pwd)
out="$root/results/spec"; mkdir -p "$out"
port=8090
log="$out/$label-server.log"
"$root/vendor/llama.cpp/build-cpu/bin/llama-server" -m "$root/data/models/N-ATLaS.Q4_K_M.gguf" \
    -t 6 -c 2048 -np 1 -ngl 0 --port $port --no-webui "$@" > "$log" 2>&1 &
pid=$!
trap 'kill $pid 2>/dev/null || true' EXIT
until curl -sf "http://127.0.0.1:$port/health" > /dev/null; do
    kill -0 $pid 2>/dev/null || { echo "server died; see $log" >&2; exit 1; }
    sleep 1
done
meta=$(mktemp)
printf '{"label": "%s", "server_args": "%s", "hardware": %s}\n' "$label" "$*" "$(cat "$root/results/probe/m4-macbook-dev.json")" > "$meta"
uv run --quiet --project "$root/atlas-trace" python -m atlas_trace.llamacpp_baseline \
    --server "http://127.0.0.1:$port" --flores "$root/data/flores200_dataset" \
    --prompts "$prompts" --meta "$meta" --out "$out/$label.json" 2>&1 | grep -vE "^(en|yo|ha|ig) #"
kill $pid; wait $pid 2>/dev/null || true
