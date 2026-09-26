#!/usr/bin/env bash
# Whole V3/V4 pipeline on a rented CUDA pod (docs/cloud-vx.md). Run in tmux
# after scripts/cloud_vx_upload.sh and `hf auth login` (read token) on the pod:
#   bash /workspace/cloud_vx_pod.sh
# Inputs under $W: vx-tok/ (expanded tokenizer), data/{wiki,flores200_dataset},
# chat-distill.jsonl and chat-wiki.jsonl (the original model's replies;
# regenerated if absent, ~15 and ~45 min), prior/eval100-original.json.
# Recipe v3 (docs/METHOD.md): 60% distilled replies, 40% Wikipedia.
set -euo pipefail
export HF_HOME=${HF_HOME:-/workspace/.cache/huggingface}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=${W:-/workspace/work}; R=$W/results; mkdir -p "$R"
PY=/workspace/venv/bin/python; F=$W/data/flores200_dataset
log() { echo "== $(date +%H:%M:%S) $*"; }

# Model files are ~16 GB each; use RAM-backed /dev/shm if the disk is small.
if [ "$(df --output=avail -BG "$W" | tail -1 | tr -dc 0-9)" -lt 80 ]; then
    M=/dev/shm/work; mkdir -p $M; log "small disk: models in $M (lost on restart)"
else
    M=$W
fi

log setup
[ -x $PY ] || python -m venv --system-site-packages /workspace/venv
$PY -m pip install -q "transformers==5.17.0" safetensors "huggingface_hub[hf_xet]" sacrebleu
$PY -c "import torch; assert torch.cuda.is_available(), 'no CUDA GPU'"
[ -f $M/n-atlas/config.json ] || /workspace/venv/bin/hf download NCAIR1/N-ATLaS --local-dir $M/n-atlas \
    --exclude "*.gguf" --exclude "original/*"

cd /workspace/wazocore
V="$PY -m wazocore.vx_torch"
if [ ! -s $W/chat-distill.jsonl ]; then
    log distill-flores; $V distill --model $M/n-atlas --flores $F --batch 256 --out $W/chat-distill.jsonl
fi
if [ ! -s $W/chat-wiki.jsonl ]; then
    log distill; $V distill --model $M/n-atlas --flores $F --wiki $W/data/wiki --n 4000 --batch 256 --out $W/chat-wiki.jsonl
fi
cat $W/chat-distill.jsonl $W/chat-wiki.jsonl > $W/chat-all.jsonl
log expand;  $V expand --base $M/n-atlas --vx $W/vx-tok --out $M/vx-init
log train;   $V train --model $M/vx-init --data $W/data --flores $F --out $M/vx-trained \
                 --chat $W/chat-all.jsonl --chat-share 0.6 --eval-every 100
rm -rf $M/vx-init
cp $M/vx-trained/log.jsonl $R/train-log.jsonl
log bpc;     $V bpc --model $M/vx-trained --flores $F --out $R/bpc-trained.json
log sample;  $V sample --model $M/vx-trained --flores $F --n 100 --out $R/eval100-trained.json 2>/dev/null
log score;   $V score $W/prior/eval100-original.json $R/eval100-trained.json --out $R/eval100-summary.json

log gguf
cd /workspace
[ -d llama.cpp ] || git clone -q --depth 1 https://github.com/ggml-org/llama.cpp
[ -x venv-gguf/bin/python ] || { python -m venv venv-gguf && venv-gguf/bin/pip install -q -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt; }
command -v cmake >/dev/null || { apt-get update -qq && apt-get install -y -qq cmake >/dev/null; }
[ -x llama.cpp/build/bin/llama-quantize ] || { cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF >/dev/null \
    && cmake --build llama.cpp/build --target llama-quantize -j 32 >/dev/null; }
# transformers 5 writes a tokenizer class the converter's transformers 4 cannot load.
S=$M/vx-gguf-src; rm -rf $S; mkdir -p $S
for f in $M/vx-trained/*; do ln -s "$f" $S/; done
rm $S/tokenizer_config.json
sed 's/"TokenizersBackend"/"PreTrainedTokenizerFast"/' $M/vx-trained/tokenizer_config.json > $S/tokenizer_config.json
convert() { venv-gguf/bin/python llama.cpp/convert_hf_to_gguf.py $S --outtype bf16 --outfile $M/vx.bf16.gguf 2>&1 | tee $R/convert.log; }
if ! convert; then
    # New merges change the pre-tokenizer fingerprint; the pre-tokenizer itself
    # is Llama-3's (checked by vocab_expand tests), so map it to llama-bpe.
    h=$(grep -o 'chkhsh: *[0-9a-f]*' $R/convert.log | tail -1 | awk '{print $2}')
    [ -n "$h" ] || { log "convert failed for another reason"; exit 1; }
    python3 - "$h" <<'PY'
import sys, pathlib
p = pathlib.Path("/workspace/llama.cpp/conversion/base.py"); s = p.read_text()
anchor = "    def get_vocab_base_pre(self, tokenizer) -> str:"
i = s.index("res = None", s.index(anchor)) + len("res = None")
s = s[:i] + f'\n        if chkhsh == "{sys.argv[1]}":\n            res = "llama-bpe"  # N-ATLaS-vx: Llama-3 pre-tokenizer, added merges' + s[i:]
p.write_text(s)
PY
    convert
fi
llama.cpp/build/bin/llama-quantize $M/vx.bf16.gguf $W/N-ATLaS-vx.Q4_K_M.gguf Q4_K_M | tail -3
rm $M/vx.bf16.gguf
sha256sum $W/N-ATLaS-vx.Q4_K_M.gguf $M/vx-trained/model.safetensors | tee $R/sha256.txt
log ALL-DONE
