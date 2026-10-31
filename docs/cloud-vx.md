# Training on a cloud GPU

Everything except training runs on a laptop. Training, evaluation and GGUF
export need one 80 GB GPU for about two hours. This runbook uses RunPod; any
host with SSH, an NVIDIA A100 or H100 80GB, CUDA PyTorch and 200 GB of disk
works the same way.

## Pipelines

Each pipeline is one script that runs end to end on the GPU host and writes its
results under `/workspace/work/`.

| Script | What it does | Time on an A100 |
|---|---|---|
| `cloud_vx_pod.sh` | Builds WazoCore with the v4 recipe: expansion, self-distillation (reuses cached replies when present), training, BPC, 200-sentence chrF++ gate with bootstrap intervals, GGUF export, and the no-new-tokens control run | about 3 h |
| `cloud_vx_validate.sh` | Retrains with a second seed and scores the original, the released model and the new run on NTREX-128 (500) and FLORES-200 (200) | about 2 h |
| `cloud_vx_v5.sh` | The on-policy distillation experiment (section 5 of `HISTORY.md`) | about 4 h |
| `cloud_vx_v6.sh` | The release (`METHOD.md`). `ARMS=long` builds it: expansion, human-translation data, training, a 700-sentence gate against the original and GGUF export. The default arms `fix`, `gold` and `control` and the optional `small` and `mbr` are the experiments behind it | `long` about 1.5 h; the three default arms 2.6 h; `small` 50 min; `mbr` adds about 2 h (not run) |
| `cloud_vx_v7.sh` | The open questions of `METHOD.md` section 6, as fixed in `PLAN-v7.md`: held-out test, matched control, ablations, the standard recipe, three seeds, IrokoBench, AfriCOMET. Every step is skipped once its result exists. Trains nothing for release | about 20 h |
| `cloud_vx_v8.sh` | Instruction data, as fixed in `PLAN-v8.md`: Aya mixed in at 20% of training (`v8_aya.py` filters it against the test sets), a matched control, release seed 1 on IrokoBench, and an exploratory AfriXNLI check of the release with original tokens. Built like v7 | about 8 h |
| `cloud_gguf.sh` | GGUF export, shared by the pipelines. `IMATRIX=/workspace/work/data/wiki` also writes a file quantised with an importance matrix (not yet run or gated) | about 5 min; the importance matrix is untimed |

## Run it

1. **Start a GPU host.** On RunPod: add your public SSH key under Settings →
   SSH Public Keys *before* deploying, then deploy an A100 80GB with the
   "RunPod PyTorch" template and a 200 GB disk. Note the "SSH over exposed TCP"
   host and port.
2. **Upload code and inputs** from the repository root (about 150 MB):

   ```sh
   SSH_KEY=~/.ssh/id_ed25519 scripts/cloud_vx_upload.sh HOST PORT
   ```

   This copies the code, the pipelines, the Wikipedia and FLORES text, NTREX,
   the expanded tokenizer and any cached distillation data, creates a Python
   environment on the host, and prints the GPU name as a check.
3. **Log in to Hugging Face** on the host (N-ATLaS is gated; a read token is
   enough): `/workspace/venv/bin/hf auth login`.
4. **Start the pipeline** in `tmux`, so it survives a dropped connection:

   ```sh
   tmux new -d -s run 'HF_HOME=/workspace/.cache/huggingface ARMS=long GGUF=long bash /workspace/cloud_vx_v6.sh 2>&1 | tee /workspace/run.log'
   ```

   `cloud_vx_validate.sh` and `cloud_vx_v5.sh` also need
   `HF_MODEL=<user>/<repo>`, the Hugging Face repository holding the trained
   model to evaluate or refine.
5. **Check the run.** The log prints one `== HH:MM:SS` line per stage and
   ends with `ALL-DONE`; the gate result and the exported file should exist:

   ```sh
   tail -1 /workspace/run.log   # == HH:MM:SS ALL-DONE
   ls /workspace/work/results-v6/pooled-long.json /workspace/work/WazoCore-v6-long.Q4_K_M.gguf
   ```

   A finished arm is skipped on a rerun, so after an error fix the cause and
   start the same command again.
6. **Copy results back** with `rsync` as each arm finishes, and check
   `sha256.txt` against the copies before stopping the host. The host bills by
   the hour until it is deleted. Upload the trained model (16 GB) to a private
   Hugging Face repository from the host; it is too large to bring to a laptop.

## Community Cloud

Community pods cost about a third less than Secure ones, but run on other
people's machines: a host can disappear, there is no network volume, and the
host owner has physical access. `cloud_vx_v7.sh` and `cloud_vx_v8.sh` are
built for that. The commands below are for v7; for v8 run
`cloud_vx_v8.sh` and pull with `RUN=v8`.

- **Deploy** an A100 80GB (PCIe or SXM) with a 200 GB container disk and no
  volume, then upload and log in as above, with a fine-grained Hugging Face
  token that can only read gated repositories. Delete the token after the run.
- **Start it** in `tmux` and keep `run.log`, which the pull script reads:

  ```sh
  tmux new -d -s run 'bash /workspace/cloud_vx_v7.sh 2>&1 | tee /workspace/run.log'
  ```

- **Pull results as they land.** On the laptop, this copies
  `results-v7/` into `results/vocab/v7/` every five minutes, prints the
  current step, and with `STOP=1` stops the pod when the run ends:

  ```sh
  STOP=1 caffeinate -i scripts/cloud_pull.sh HOST PORT
  ```

- **If the host disappears,** deploy another pod and run the upload again: it
  sends the pulled results back, and the pipeline skips everything already
  finished. It refuses to continue if the training data's hashes differ from
  the first pod's. A new pod can reuse an old address with a new host key;
  clear it with `ssh-keygen -R "[HOST]:PORT"`.
- **Nothing private leaves the pod.** v7 and v8 download the public release, keep
  weights only while they score them, and write nothing to Hugging Face.

## Outputs

- `results/`: every score, sample and training log, as JSON.
- A Q4_K_M GGUF of the trained model (about 5 GB).
- The trained model in Hugging Face format (about 16 GB).

## Known pitfalls

- **Small disks.** If the host has less than 80 GB free, the pipeline keeps
  model files in RAM-backed `/dev/shm`; copy results out before any restart.
- **Hugging Face login location.** A login made in a web terminal is stored
  under `/workspace/.cache/huggingface`; set `HF_HOME` to that path for SSH
  sessions, as in step 4.
- **llama.cpp's converter** downgrades transformers and installs a CPU-only
  PyTorch. `cloud_gguf.sh` gives it a separate environment; never install its
  requirements into the training environment.
- **Tokenizer class.** transformers 5 saves `tokenizer_class:
  TokenizersBackend`, which the converter cannot read; `cloud_gguf.sh` converts
  from a copy that declares `PreTrainedTokenizerFast`.
- **Pre-tokenizer fingerprint.** The converter identifies tokenizers by a hash
  of sample output, which the new merges change. `cloud_gguf.sh` maps the new
  hash to `llama-bpe`, which is correct because the pre-tokenizer itself is
  unchanged.
- **Memory.** With the teacher model loaded for the KL anchor, training uses
  about 79 of 80 GB at micro-batch 8; set `KL_BATCH=2`, then `BATCH=4 ACCUM=4`,
  if a different GPU runs out of memory.
- **Unreachable SSH port.** If the host's exposed TCP port stops answering,
  move files with `runpodctl send` on the host and `runpodctl receive` on the
  laptop; it needs no inbound connection.
- **Disk without a volume.** If `/workspace` is on the host's own disk and not
  on a network volume, everything is erased when the host stops. Once free
  space falls below 80 GB the pipeline also keeps models in memory.
