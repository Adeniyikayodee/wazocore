#!/usr/bin/env bash
# Copies code and inputs for scripts/cloud_vx_pod.sh to a pod.
#   [SSH_KEY=~/.ssh/key] scripts/cloud_vx_upload.sh HOST PORT [VX_TOKENIZER_DIR]
set -euo pipefail
host=${1:?host}; port=${2:?port}; vx=${3:-data/models/n-atlas-vx2}
root=$(cd "$(dirname "$0")/.." && pwd); cd "$root"
e="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -i ${SSH_KEY:-$HOME/.ssh/id_ed25519} -p $port"
$e root@$host 'mkdir -p /workspace/work/data'
rsync -az -e "$e" --exclude .venv --exclude __pycache__ --exclude .pytest_cache wazocore root@$host:/workspace/
rsync -az -e "$e" scripts/cloud_vx_pod.sh scripts/cloud_vx_validate.sh scripts/cloud_vx_v5.sh scripts/cloud_vx_v6.sh \
    scripts/cloud_vx_v7.sh scripts/cloud_vx_v8.sh scripts/v8_aya.py scripts/africomet.py scripts/cloud_gguf.sh root@$host:/workspace/
echo "$(git rev-parse HEAD)$(git diff --quiet HEAD -- wazocore scripts || echo ' dirty')" | $e root@$host 'cat > /workspace/wazocore/REVISION'
# v7 and v8 on a fresh pod: send back what earlier pods finished, so it is skipped
[ -d results/vocab/v7 ] && rsync -az -e "$e" --exclude 'run-*.log' results/vocab/v7/ root@$host:/workspace/work/results-v7/
[ -d results/vocab/v8 ] && rsync -az -e "$e" --exclude 'run-*.log' results/vocab/v8/ root@$host:/workspace/work/results-v8/
[ -d data/ntrex ] && rsync -az -e "$e" data/ntrex/ root@$host:/workspace/work/ntrex/
# samples of the original and v4 from the validation run, reused by cloud_vx_v5.sh
$e root@$host 'mkdir -p /workspace/work/prior'
rsync -az -e "$e" results/vocab/validate/ntrex-n-atlas.json results/vocab/validate/ntrex-v4.json \
    results/vocab/validate/flores-n-atlas.json results/vocab/validate/flores-v4.json root@$host:/workspace/work/prior/
rsync -az -e "$e" data/wiki data/flores200_dataset root@$host:/workspace/work/data/
rsync -az -e "$e" "$vx"/ root@$host:/workspace/work/vx-tok/
# optional smaller vocabulary for the `small` arm of cloud_vx_v6.sh
[ -d data/models/n-atlas-vx-5k ] && rsync -az -e "$e" data/models/n-atlas-vx-5k/ root@$host:/workspace/work/vx-tok-small/
for f in data/runs/cloud/chat-distill.jsonl data/runs/cloud/chat-wiki.jsonl; do  # optional: regenerated on the pod if absent
    [ -s "$f" ] && rsync -az -e "$e" "$f" root@$host:/workspace/work/
done
# The pod image has no `hf`; the login below needs it before the pipeline runs.
$e root@$host '[ -x /workspace/venv/bin/hf ] || { python -m venv --system-site-packages /workspace/venv && /workspace/venv/bin/pip install -q "transformers==5.17.0" safetensors "huggingface_hub[hf_xet]" sacrebleu; }; /workspace/venv/bin/python -c "import torch; print(\"GPU:\", torch.cuda.get_device_name(0))"'
echo "uploaded; on the pod: /workspace/venv/bin/hf auth login (read token), then tmux new -s vx 'bash /workspace/cloud_vx_pod.sh 2>&1 | tee /workspace/run.log'"
