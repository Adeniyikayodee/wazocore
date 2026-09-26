# V3 on a rented GPU (vocabulary-expansion training)

Why: local MLX training on the 16 GB laptop swaps (~15 tok/s), so the
V3 run (PROJECT-PLAN-B §14) needs one 80 GB GPU for a few hours.
Script: `wazocore/wazocore/vx_torch.py`.

## Quick path (v2 vocabulary, one script)

1. Deploy the pod (section 1). The key `~/.ssh/runpod_natlas.pub` must be in RunPod Settings **before** deploying.
2. From the laptop: `scripts/cloud_vx_upload.sh HOST PORT`. This uploads the code, data, the v2 tokenizer, the distilled conversations, and the original model's scores from the 25 Sep run.
3. On the pod (web terminal): `/workspace/venv/bin/hf auth login` with a read token (the upload script installs `hf` and prints the GPU name), then
   `tmux new -s vx 'bash /workspace/cloud_vx_pod.sh 2>&1 | tee /workspace/run.log'`.
4. Expected run: ~1.5 h on an A100 80GB (download 2 min, training ~35 min, eval ~20 min, GGUF ~15 min). Outputs land in `/workspace/work/`: `results/`, `N-ATLaS-vx.Q4_K_M.gguf`, and the trained model (in `/dev/shm/work/vx-trained` if the disk is small).
5. Copy back the GGUF, `results/` and the trained model, and check `results/sha256.txt` before deleting the pod.

The steps below are the manual version the script automates.

## 1. Pod

- RunPod, 1× A100 80GB or H100 80GB, template "RunPod PyTorch", volume 150 GB.
- Add the laptop's `~/.ssh/id_ed25519.pub` under Settings → SSH Public Keys.
- Use the "SSH over exposed TCP" command (plain ssh; scp/rsync work with it).

## 2. Upload (from the repo root on the laptop, ~140 MB)

    export POD="root@HOST"; export PORT=PORT
    rsync -az -e "ssh -p $PORT" --exclude .venv --exclude __pycache__ wazocore $POD:/workspace/
    rsync -az -e "ssh -p $PORT" data/wiki data/flores200_dataset $POD:/workspace/work/data/
    rsync -az -e "ssh -p $PORT" data/models/n-atlas-vx/ $POD:/workspace/work/n-atlas-vx-tok/

## 3. On the pod

    cd /workspace/wazocore
    pip install -q "transformers>=5.0" safetensors
    hf auth login          # read-only token; N-ATLaS is gated
    hf download NCAIR1/N-ATLaS --local-dir ../work/n-atlas
    cd /workspace/wazocore && export W=../work F=../work/data/flores200_dataset
    python -m wazocore.vx_torch expand --base $W/n-atlas --vx $W/n-atlas-vx-tok --out $W/vx-init
    python -m wazocore.vx_torch bpc --model $W/n-atlas --flores $F --out $W/bpc-original.json
    python -m wazocore.vx_torch bpc --model $W/vx-init --flores $F --out $W/bpc-init.json

Pilot (catch divergence or out-of-memory in minutes, not hours):

    python -m wazocore.vx_torch train --model $W/vx-init --data $W/data --flores $F \
        --out $W/vx-pilot --tokens 400000 --eval-every 25

Loss must fall and yo/ha/ig BPC must drop below the `bpc-init` values.
Out of memory: `--batch 8`. Then the full run (~5M tokens) in tmux:

    tmux new -s vx
    python -m wazocore.vx_torch train --model $W/vx-init --data $W/data --flores $F --out $W/vx-trained
    python -m wazocore.vx_torch bpc --model $W/vx-trained --flores $F --out $W/bpc-trained.json
    python -m wazocore.vx_torch sample --model $W/vx-trained

Gate (V3): trained BPC within +3% of `bpc-original` per language, both
measured here on the same GPU (the laptop's results/vocab/bpc-original.json
is from the 4-bit MLX model and is not the comparison point).

## 4. GGUF on the pod, then bring back only ~5 GB

    git clone --depth 1 https://github.com/ggml-org/llama.cpp /workspace/llama.cpp
    pip install -q -r /workspace/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
    cmake -S /workspace/llama.cpp -B /workspace/llama.cpp/build && cmake --build /workspace/llama.cpp/build --target llama-quantize -j
    python /workspace/llama.cpp/convert_hf_to_gguf.py $W/vx-trained --outtype bf16 --outfile $W/vx.bf16.gguf
    /workspace/llama.cpp/build/bin/llama-quantize $W/vx.bf16.gguf $W/N-ATLaS-vx.Q4_K_M.gguf Q4_K_M

From the laptop:

    rsync -az -e "ssh -p $PORT" $POD:/workspace/work/{N-ATLaS-vx.Q4_K_M.gguf,bpc-*.json,vx-trained/log.jsonl} data/runs/cloud/

## What actually happened (25 Sep run) and the fixes

- **No volume:** the pod came up without a volume (30 GB container disk only). Model files lived in `/dev/shm` (134 GB, RAM-backed), deleting each one after use. It is lost on restart, so copy results out early.
- **Hugging Face login:** the web terminal's `hf auth login` stores the token under `/workspace/.cache/huggingface`, so SSH commands need `export HF_HOME=/workspace/.cache/huggingface`.
- **Separate environment for llama.cpp:** llama.cpp's converter requirements downgrade transformers (to 4.57) and install a CPU-only torch. Give it its own venv (`/workspace/venv-gguf`) and never install it into the training venv.
- **Tokenizer class:** transformers 5 saves `tokenizer_class: TokenizersBackend`, which the older converter cannot load. Convert from a symlinked copy whose `tokenizer_config.json` says `PreTrainedTokenizerFast`.
- **Pre-tokenizer hash:** the converter rejects the expanded tokenizer because its pre-tokenizer hash is unknown. After checking that the pre-tokenizer is unchanged, add the reported `chkhsh` → `res = "llama-bpe"` in `conversion/base.py`.
- **Recipe:** the winning run adds self-distillation (`distill`, then `train --chat`); see PROJECT-PLAN-B §14.1.

**Stop and delete the pod afterwards**: it bills per hour while it exists.
