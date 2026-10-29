#!/usr/bin/env bash
# v6: aim above the original in all three languages, on the canonical path
# (docs/METHOD.md). One pod session; each arm changes one thing and
# is scored against the original on NTREX-128 (500) + FLORES-200 (200) with
# `vx_torch pooled`. No arm trains on FLORES text.
#   fix      v4's data (the original's greedy replies to Wikipedia requests),
#            with calibrated initialisation, the canonical anchor and penalty.
#   gold     fix + human translations (google/smol).
#   mbr      gold, with the greedy English -> X replies replaced by the MBR
#            choice among $CANDIDATES replies (adds about 2 h of generation).
#   control  the original model on gold's data, no new tokens: separates what
#            the data gives from what the vocabulary costs.
#   small    gold, with a smaller vocabulary (needs $W/vx-tok-small).
#   long     gold, with twice the training (12M tokens): the release recipe.
#   data     no training: download and build the data, then stop.
# Run after scripts/cloud_vx_upload.sh and `hf auth login`:
#   bash /workspace/cloud_vx_v6.sh
# Knobs: ARMS (default "fix gold control"), CANDIDATES (default 9), TOKENS, SEED,
# CANON, KL_BATCH, BATCH and ACCUM (on an out-of-memory error lower KL_BATCH to
# 2, then BATCH to 4 with ACCUM 4; finished arms are skipped on a rerun),
# HF_MODEL (the v4 weights: adds the canonical-decoding diagnostic), GGUF
# (arm to export, default gold).
set -euo pipefail
export HF_HOME=${HF_HOME:-/workspace/.cache/huggingface}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=${W:-/workspace/work}; R=$W/results-v6; mkdir -p "$R"
ARMS=${ARMS:-"fix gold control"}; CANDIDATES=${CANDIDATES:-9}; TOKENS=${TOKENS:-6000000}
CANON=${CANON:-1.0}; KL_BATCH=${KL_BATCH:-4}; BATCH=${BATCH:-8}; ACCUM=${ACCUM:-2}; SEED=${SEED:-0}; GGUF=${GGUF:-gold}
PY=/workspace/venv/bin/python; F=$W/data/flores200_dataset; N=$W/ntrex; P=$W/prior
log() { echo "== $(date +%H:%M:%S) $*"; }

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
[ -f $W/smol/smolsent/en_yo.jsonl ] || /workspace/venv/bin/hf download google/smol --repo-type dataset --local-dir $W/smol \
    smolsent/en_yo.jsonl smolsent/en_ha.jsonl smolsent/en_ig.jsonl smoldoc/en_yo.jsonl smoldoc/en_ha.jsonl smoldoc/en_ig.jsonl

cd /workspace/wazocore
V="$PY -m wazocore.vx_torch"
[ -s $W/chat-wiki.jsonl ] || { log "distill (greedy, both directions)"; $V distill --model $M/n-atlas --flores $F --wiki $W/data/wiki --n 4000 --batch 256 --out $W/chat-wiki.jsonl; }
log smol; $V smol --smol $W/smol --out $W/chat-smol.jsonl
case " $ARMS " in *" mbr "*)
    [ -s $W/chat-mbr.jsonl ] || { log "distill (MBR over $CANDIDATES replies, English -> X)"
        $V distill --model $M/n-atlas --flores $F --wiki $W/data/wiki --n 4000 --batch 256 --forward-only --candidates "$CANDIDATES" --out $W/chat-mbr.jsonl; } ;;
esac
# One shuffled file per data recipe: training takes conversations in file order.
$PY - "$W" <<'PY'
import json, os, random, sys
w = sys.argv[1]; read = lambda f: open(f"{w}/{f}", encoding="utf-8").read().splitlines()
forward = lambda l: json.loads(l)["messages"][0]["content"].startswith("Translate the following English text into")
wiki, smol = read("chat-wiki.jsonl"), read("chat-smol.jsonl")
recipes = {"fix": wiki, "gold": wiki + smol}
if os.path.exists(f"{w}/chat-mbr.jsonl"):
    recipes["mbr"] = read("chat-mbr.jsonl") + [l for l in wiki if not forward(l)] + smol
for name, lines in recipes.items():
    lines = list(lines); random.Random(0).shuffle(lines)
    open(f"{w}/chat-v6-{name}.jsonl", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"chat-v6-{name}: {len(lines)} conversations", file=sys.stderr)
PY

recipe() { echo "--data $W/data --chat $W/chat-v6-$1.jsonl --chat-share 0.85 --careful-wiki --batch $BATCH --accum $ACCUM --eval-every 100 --tokens $TOKENS --seed $SEED --teacher $M/n-atlas --kl-weight 1.0 --kl-batch $KL_BATCH"; }
gate() {  # NAME: sample both test sets, one pooled estimate per language against the original
    log "sample $1"
    $V sample --model $M/$1 --flores $F --ntrex $N --n 500 --out $R/ntrex-$1.json 2>/dev/null
    $V sample --model $M/$1 --flores $F --n 200 --out $R/flores-$1.json 2>/dev/null
    log "gate $1"
    $V pooled --base $P/ntrex-n-atlas.json $P/flores-n-atlas.json --run $R/ntrex-$1.json $R/flores-$1.json --out $R/pooled-$1.json
}
arm() {  # NAME BASE DATA [train flags...]
    name=$1; base=$2; data=$3; shift 3
    [ -s $R/pooled-$name.json ] && { log "$name: done"; return; }
    log "train $name"; $V train --model $base --flores $F --out $M/$name $(recipe $data) "$@"
    cp $M/$name/log.jsonl $R/train-$name-log.jsonl
    gate $name
    [ "$name" = "$GGUF" ] || rm -rf $M/$name  # 16 GB each; only the exported arm is kept
}
init() {  # NAME TOKENIZER: the expanded model with calibrated initialisation
    [ -f $M/$1/config.json ] || { log "expand $1"; $V expand --base $M/n-atlas --vx $2 --out $M/$1 --out-weights chars --calibrate-input-norm; }
}

if [ -n "${HF_MODEL:-}" ] && [ ! -s $R/pooled-v4-canonical.json ]; then
    # Does v4 keep its quality when it may only write the way its tokenizer would?
    [ -f $M/v4/config.json ] || /workspace/venv/bin/hf download "$HF_MODEL" --local-dir $M/v4
    log "diagnostic: v4, canonical decoding"
    $V sample --model $M/v4 --flores $F --ntrex $N --n 500 --canonical --out $R/ntrex-v4-canonical.json 2>/dev/null
    $V pooled --base $P/ntrex-v4.json --run $R/ntrex-v4-canonical.json --out $R/pooled-v4-canonical.json
fi
CANONICAL="--anchor canonical --canon-weight $CANON"
for a in $ARMS; do
    case $a in
    fix)     init init-v6 $W/vx-tok;          arm fix $M/init-v6 fix $CANONICAL ;;
    gold)    init init-v6 $W/vx-tok;          arm gold $M/init-v6 gold $CANONICAL ;;
    mbr)     init init-v6 $W/vx-tok;          arm mbr $M/init-v6 mbr $CANONICAL ;;
    small)   init init-small $W/vx-tok-small; arm small $M/init-small gold $CANONICAL ;;
    long)    init init-v6 $W/vx-tok;          TOKENS=12000000 arm long $M/init-v6 gold $CANONICAL ;;
    control) arm control $M/n-atlas gold ;;
    data)    ;;  # prepare the data only (cloud_vx_v7.sh)
    *) echo "unknown arm: $a" >&2; exit 1 ;;
    esac
done
[ -f $M/$GGUF/config.json ] && { log gguf; EDGE_TYPE=${EDGE_TYPE:-q8_0} bash /workspace/cloud_gguf.sh $M/$GGUF $W/WazoCore-v6-$GGUF.Q4_K_M.gguf $R; }
log ALL-DONE
