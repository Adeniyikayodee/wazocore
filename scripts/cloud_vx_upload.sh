#!/usr/bin/env bash
# Copies code and inputs for scripts/cloud_vx_pod.sh to a pod.
#   scripts/cloud_vx_upload.sh HOST PORT [VX_TOKENIZER_DIR]
set -euo pipefail
host=${1:?host}; port=${2:?port}; vx=${3:-data/models/n-atlas-vx2}
root=$(cd "$(dirname "$0")/.." && pwd); cd "$root"
e="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -i $HOME/.ssh/runpod_natlas -p $port"
$e root@$host 'mkdir -p /workspace/work/prior /workspace/work/data'
rsync -az -e "$e" --exclude .venv --exclude __pycache__ --exclude .pytest_cache atlas-trace root@$host:/workspace/
rsync -az -e "$e" scripts/cloud_vx_pod.sh root@$host:/workspace/
rsync -az -e "$e" data/wiki data/flores200_dataset root@$host:/workspace/work/data/
rsync -az -e "$e" "$vx"/ root@$host:/workspace/work/vx-tok/
rsync -az -e "$e" data/runs/cloud/chat-distill.jsonl root@$host:/workspace/work/
rsync -az -e "$e" results/vocab/cloud/bpc-original.json results/vocab/cloud/eval30-original.json root@$host:/workspace/work/prior/
# The pod image has no `hf`; the login below needs it before the pipeline runs.
$e root@$host '[ -x /workspace/venv/bin/hf ] || { python -m venv --system-site-packages /workspace/venv && /workspace/venv/bin/pip install -q "transformers==5.17.0" safetensors "huggingface_hub[hf_xet]" sacrebleu; }; /workspace/venv/bin/python -c "import torch; print(\"GPU:\", torch.cuda.get_device_name(0))"'
echo "uploaded; on the pod: /workspace/venv/bin/hf auth login (read token), then tmux new -s vx 'bash /workspace/cloud_vx_pod.sh 2>&1 | tee /workspace/run.log'"
