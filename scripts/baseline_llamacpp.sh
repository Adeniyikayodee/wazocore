#!/usr/bin/env bash
# llama.cpp baseline for one device (PROJECT-PLAN-B §6, P0).
#
#   scripts/baseline_llamacpp.sh LABEL MODEL.gguf THREADS [PROMPTS]
#
# 1. llama-bench: synthetic pp512 / tg128, 5 reps, at 1..THREADS-style list.
# 2. llama-server (one slot: a single edge user) + atlas_trace.llamacpp_baseline:
#    per-language FLORES prompts,
#    prefill/decode tok/s, words/s, chars/s; peak RSS from /usr/bin/time.
# Outputs go to results/baseline/LABEL-*.json.
set -euo pipefail

label=${1:?label}; model=${2:?model.gguf}; threads=${3:?threads}; prompts=${4:-10}
root=$(cd "$(dirname "$0")/.." && pwd)
bin="$root/vendor/llama.cpp/build-cpu/bin"
out="$root/results/baseline"; mkdir -p "$out"
port=8088

model_sha=$(shasum -a 256 "$model" | cut -d' ' -f1)
llama_commit=$(git -C "$root/vendor/llama.cpp" rev-parse HEAD)

echo "== llama-bench ($label)" >&2
"$bin/llama-bench" -m "$model" -t "1,$threads" -p 512 -n 128 -r 5 -ngl 0 -o json \
    > "$out/$label-llama-bench.json"

echo "== llama-server ($label, $threads threads)" >&2
case "$(uname)" in
    Darwin) time_flag=-l ;;
    *) time_flag=-v ;;
esac
server_log="$out/$label-server.log"
/usr/bin/time $time_flag "$bin/llama-server" -m "$model" -t "$threads" -c 2048 -np 1 -ngl 0 \
    --port $port --no-webui > "$server_log" 2>&1 &
server_pid=$!
trap 'kill $server_pid 2>/dev/null || true' EXIT
until curl -sf "http://127.0.0.1:$port/health" > /dev/null; do
    kill -0 $server_pid 2>/dev/null || { echo "server died; see $server_log" >&2; exit 1; }
    sleep 1
done

meta=$(mktemp)
cat > "$meta" <<EOF
{
  "label": "$label",
  "engine": "llama.cpp",
  "llama_cpp_commit": "$llama_commit",
  "build": "CPU only: GGML_METAL=OFF GGML_BLAS=OFF GGML_NATIVE=ON",
  "model_path": "$model",
  "model_sha256": "$model_sha",
  "threads": $threads,
  "ctx": 2048,
  "server_slots": 1
}
EOF

uv run --quiet --project "$root/atlas-trace" python -m atlas_trace.llamacpp_baseline \
    --server "http://127.0.0.1:$port" --flores "$root/data/flores200_dataset" \
    --prompts "$prompts" --meta "$meta" --out "$out/$label-languages.json"

# /usr/bin/time's pid is $server_pid; its child is the server.
pkill -INT -P $server_pid llama-server || true
wait $server_pid 2>/dev/null || true
trap - EXIT

python3 - "$out/$label-languages.json" "$server_log" <<'EOF'
import json, re, sys
path, log = sys.argv[1], open(sys.argv[2]).read()
mac = re.search(r"(\d+)\s+maximum resident set size", log)
linux = re.search(r"Maximum resident set size \(kbytes\): (\d+)", log)
rss = int(mac.group(1)) if mac else int(linux.group(1)) * 1024 if linux else None
report = json.load(open(path))
report["server_peak_rss_bytes"] = rss
json.dump(report, open(path, "w"), indent=2, ensure_ascii=False)
print(f"peak RSS: {rss / 2**30:.2f} GiB" if rss else "peak RSS: unavailable", file=sys.stderr)
EOF
