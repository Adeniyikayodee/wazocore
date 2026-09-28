#!/usr/bin/env bash
# Is the v4 result real? One pod session (docs/METHOD.md, "Validation"):
#   - a fresh test set never used in development (NTREX-128, news);
#   - a second training seed of the v4 recipe;
# each scored against the original with bootstrap intervals. All three
# models are sampled the same batched way, so their comparison is fair.
#   HF_MODEL=<user>/<private repo holding v4> bash /workspace/cloud_vx_validate.sh
# Inputs as for cloud_vx_pod.sh, plus $W/ntrex/ (scripts/cloud_vx_upload.sh).
set -euo pipefail
export HF_HOME=${HF_HOME:-/workspace/.cache/huggingface}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
: "${HF_MODEL:?set HF_MODEL to the private repo holding the v4 weights}"
W=${W:-/workspace/work}; R=$W/results-validate; mkdir -p "$R"
PY=/workspace/venv/bin/python; F=$W/data/flores200_dataset; N=$W/ntrex
SEED=${SEED:-1}; NTREX_N=${NTREX_N:-500}; FLORES_N=${FLORES_N:-200}
log() { echo "== $(date +%H:%M:%S) $*"; }

log setup
$PY -m pip install -q "transformers==5.17.0" safetensors "huggingface_hub[hf_xet]" sacrebleu
$PY -c "import torch; assert torch.cuda.is_available(), 'no CUDA GPU'"
[ -f $W/n-atlas/config.json ] || /workspace/venv/bin/hf download NCAIR1/N-ATLaS --local-dir $W/n-atlas --exclude "*.gguf" --exclude "original/*"
[ -f $W/v4/config.json ] || /workspace/venv/bin/hf download "$HF_MODEL" --local-dir $W/v4

cd /workspace/wazocore
V="$PY -m wazocore.vx_torch"
cat $W/chat-distill.jsonl $W/chat-wiki.jsonl > $W/chat-all.jsonl
log expand;  $V expand --base $W/n-atlas --vx $W/vx-tok --out $W/vx-init
log train-seed$SEED
$V train --model $W/vx-init --flores $F --out $W/seed$SEED --seed $SEED \
    --data $W/data --chat $W/chat-all.jsonl --chat-share 0.85 --careful-wiki --batch 8 --accum 2 --eval-every 100 \
    --teacher $W/n-atlas --kl-weight 1.0 --kl-batch 4
rm -rf $W/vx-init
cp $W/seed$SEED/log.jsonl $R/train-seed$SEED-log.jsonl

for m in n-atlas v4 seed$SEED; do
    log "sample $m";  $V sample --model $W/$m --flores $F --ntrex $N --n $NTREX_N --out $R/ntrex-$m.json 2>/dev/null
    $V sample --model $W/$m --flores $F --n $FLORES_N --out $R/flores-$m.json 2>/dev/null
done
for m in v4 seed$SEED; do
    log "score $m (NTREX)";  $V score $R/ntrex-n-atlas.json $R/ntrex-$m.json --out $R/ntrex-$m-summary.json
    log "score $m (FLORES)"; $V score $R/flores-n-atlas.json $R/flores-$m.json --out $R/flores-$m-summary.json
done
log ALL-DONE
