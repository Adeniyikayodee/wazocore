#!/usr/bin/env bash
# Exports a trained expanded model to Q4_K_M GGUF on the pod.
#   [EDGE_TYPE=q8_0] bash /workspace/cloud_gguf.sh MODEL_DIR OUT.gguf LOG_DIR
set -euo pipefail
src=${1:?model dir}; out=${2:?output gguf}; logs=${3:?log dir}
cd /workspace
[ -d llama.cpp ] || git clone -q --depth 1 https://github.com/ggml-org/llama.cpp
[ -x venv-gguf/bin/python ] || { python -m venv venv-gguf && venv-gguf/bin/pip install -q -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt; }
command -v cmake >/dev/null || { apt-get update -qq && apt-get install -y -qq cmake >/dev/null; }
[ -x llama.cpp/build/bin/llama-quantize ] || { cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF >/dev/null \
    && cmake --build llama.cpp/build --target llama-quantize -j 32 >/dev/null; }
# transformers 5 writes a tokenizer class the converter's transformers 4 cannot load.
S=$(mktemp -d); for f in "$src"/*; do ln -s "$f" "$S"/; done
rm "$S"/tokenizer_config.json
sed 's/"TokenizersBackend"/"PreTrainedTokenizerFast"/' "$src"/tokenizer_config.json > "$S"/tokenizer_config.json
bf16="${out%.gguf}.bf16.gguf"
convert() { venv-gguf/bin/python llama.cpp/convert_hf_to_gguf.py "$S" --outtype bf16 --outfile "$bf16" 2>&1 | tee "$logs"/convert.log; }
if ! convert; then
    # New merges change the pre-tokenizer fingerprint; the pre-tokenizer itself
    # is Llama-3's (checked by vocab_expand tests), so map it to llama-bpe.
    h=$(grep -o 'chkhsh: *[0-9a-f]*' "$logs"/convert.log | tail -1 | awk '{print $2}')
    [ -n "$h" ] || { echo "convert failed for another reason" >&2; exit 1; }
    python3 - "$h" <<'PY'
import sys, pathlib
p = pathlib.Path("/workspace/llama.cpp/conversion/base.py"); s = p.read_text()
if sys.argv[1] not in s:
    anchor = "    def get_vocab_base_pre(self, tokenizer) -> str:"
    i = s.index("res = None", s.index(anchor)) + len("res = None")
    s = s[:i] + f'\n        if chkhsh == "{sys.argv[1]}":\n            res = "llama-bpe"  # N-ATLaS-vx: Llama-3 pre-tokenizer, added merges' + s[i:]
    p.write_text(s)
PY
    convert
fi
llama.cpp/build/bin/llama-quantize "$bf16" "$out" Q4_K_M | tail -2
# EDGE_TYPE=q8_0 also writes a file whose four retrained layers keep 8 bits:
# 4-bit rounding costs them about 1.7 chrF++ in Igbo (docs/METHOD.md, section 3).
[ -z "${EDGE_TYPE:-}" ] || llama.cpp/build/bin/llama-quantize --tensor-type "^blk\.(0|1|30|31)\.=$EDGE_TYPE" "$bf16" "${out%.gguf}.edge-$EDGE_TYPE.gguf" Q4_K_M | tail -2
rm "$bf16"; rm -rf "$S"
sha256sum "$out" "$src"/model.safetensors | tee -a "$logs"/sha256.txt
