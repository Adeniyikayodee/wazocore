#!/usr/bin/env bash
# v8: instruction data, as preregistered in docs/PLAN-v8.md. Built like v7
# for RunPod Community Cloud: every step is skipped once its result exists,
# results are small files the laptop pulls as they land
# (RUN=v8 scripts/cloud_pull.sh), and no weights leave the pod. Run after
# scripts/cloud_vx_upload.sh and `hf auth login`:
#   bash /workspace/cloud_vx_v8.sh
# UNITS, in priority order (default all):
#   inst     exploratory check on the release, then `inst`: 20% Aya instruction data
#   seed     release seed 1 retrained, IrokoBench only (seed spread)
#   control  `control-inst`: inst's data, no new tokens
#   comet    AfriCOMET on the new held-out samples
# On an out-of-memory error lower KL_BATCH to 2, then BATCH to 4 with ACCUM 4.
set -euo pipefail
export HF_HOME=${HF_HOME:-/workspace/.cache/huggingface}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=${W:-/workspace/work}; R=$W/results-v8; M=$W; mkdir -p "$R"
UNITS=${UNITS:-"inst seed control comet"}
KL_BATCH=${KL_BATCH:-4}; BATCH=${BATCH:-8}; ACCUM=${ACCUM:-2}
PY=/workspace/venv/bin/python; HF=/workspace/venv/bin/hf; EV=/workspace/venv-eval/bin
F=$W/data/flores200_dataset; N=$W/ntrex
BASE_REV=e294476928aca9030e924ca27bb8e085e8581273
RELEASE=mkbadeniyi/WazoCore-8B-v6-long; RELEASE_REV=be50a5df45de4abaecd1bd51231987fe24bacf1f
RELEASE_SHA=b4cc2a6891529ff1827ee66699eb82aafbb23588d6f0988e74577644168db381
CANON="--anchor canonical --canon-weight 1.0"
log() { echo "== $(date +%H:%M:%S) $*"; }
die() { echo "!! $*" >&2; exit 1; }

log preflight  # fail in the first minute, not the fifth hour
mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
[ "$mem" -ge 79000 ] || die "need an 80 GB GPU, found $mem MiB"
[ "$(df --output=avail -BG "$W" | tail -1 | tr -dc 0-9)" -ge 150 ] || die "need 150 GB free in $W"
[ -s $W/chat-wiki.jsonl ] || die "upload data/runs/cloud/chat-wiki.jsonl: the release was trained on it"
[ -s $W/results-v7/inputs.sha256 ] || die "upload results/vocab/v7: v8 checks its inputs against v7's"
$HF download NCAIR1/N-ATLaS --revision $BASE_REV --local-dir $M/n-atlas --exclude "*.gguf" --exclude "original/*" >/dev/null \
    || die "no access to N-ATLaS: $HF auth login"
[ -x $EV/lm_eval ] || { python -m venv --system-site-packages /workspace/venv-eval \
    && $EV/pip install -q "lm-eval[hf]==0.4.13" "transformers==5.17.0"; }

log data
# v6's gold data, which must match v7's byte for byte
W=$W ARMS=data GGUF=none bash /workspace/cloud_vx_v6.sh
(cd $W && sha256sum chat-wiki.jsonl chat-v6-gold.jsonl vx-tok/new_tokens.json) > $R/gold.now
cmp -s $R/gold.now $W/results-v7/inputs.sha256 || die "gold data differs from v7's: see $R/gold.now"
rm $R/gold.now
if [ ! -s $R/aya-report.json ]; then
    log "aya: filter and mix"
    $EV/python -I /workspace/v8_aya.py --vx $W/vx-tok --base $M/n-atlas --flores $F --ntrex $N \
        --gold $W/chat-v6-gold.jsonl --out $W/chat-v8-inst.jsonl --report $R/aya-report.json
fi
[ -s $W/chat-v8-inst.jsonl ] || die "chat-v8-inst.jsonl missing: delete $R/aya-report.json and rerun"
(cd $W && sha256sum chat-v8-inst.jsonl) > $R/inputs.now
[ -s $R/inputs.sha256 ] || cp $R/inputs.now $R/inputs.sha256
cmp -s $R/inputs.now $R/inputs.sha256 || die "v8 mixture differs from the first pod's: see $R/inputs.sha256"
rm $R/inputs.now
{ date -u +%FT%TZ; cat /workspace/wazocore/REVISION 2>/dev/null || echo "no REVISION file"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
  $PY -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'transformers', transformers.__version__)"; } >> $R/env.txt

cd /workspace/wazocore
V="$PY -m wazocore.vx_torch"
iroko() {  # NAME MODEL_DIR [TASKS] [MODEL_ARGS]: IrokoBench, zero-shot, prompt 1, chat template
    local name=$1 dir=$2 tasks=${3:-} extra=${4:-}
    [ -f $R/iroko-$name/done ] && return 0
    [ -n "$tasks" ] || tasks=$(for l in yor hau ibo eng; do printf 'afrixnli_%s_prompt_1,afrimmlu_direct_%s_prompt_1,afrimgsm_%s_prompt_1,' $l $l $l; done)
    log "irokobench $name"
    $EV/lm_eval --model hf --model_args pretrained=$dir,dtype=bfloat16$extra --tasks ${tasks%,} \
        --apply_chat_template --batch_size auto --seed 0 --log_samples --output_path $R/iroko-$name
    mkdir -p $R/iroko-$name && touch $R/iroko-$name/done
}
evaluate() {  # NAME: held-out translation both ways, BPC, IrokoBench
    [ -f $R/done-$1 ] && return 0
    local d way
    for d in en-xx xx-en; do
        [ -s $R/heldout-$d-$1.json ] && continue
        way=; [ $d = xx-en ] && way=--reverse
        log "sample $1 $d"
        $V sample --model $M/$1 --flores $F --ntrex $N --start 500 --n 1497 --batch 64 $way --out $R/heldout-$d-$1.json
    done
    [ -s $R/bpc-$1.json ] || { log "bpc $1"; $V bpc --model $M/$1 --flores $F --sentences 1012 --out $R/bpc-$1.json; }
    iroko $1 $M/$1
    touch $R/done-$1
}
irokoonly() { iroko $1 $M/$1; }
init() {  # the expanded, untrained model with the release's initialisation
    [ -f $M/init-v6/.done ] && return 0
    rm -rf "${M:?}/init-v6"; log "expand init-v6"
    $V expand --base $M/n-atlas --vx $W/vx-tok --out $M/init-v6 --out-weights chars --calibrate-input-norm
    touch $M/init-v6/.done
}
train() {  # NAME SEED INIT CHAT EVAL [flags]: the release recipe, then EVAL (evaluate|irokoonly) and free the disk
    local name=$1 seed=$2 from=$3 chat=$4 how=$5; shift 5
    [ -f $R/done-$name ] && { log "$name: done"; return 0; }
    [ $from = n-atlas ] || init
    if [ ! -f $M/$name/.trained ]; then
        rm -rf "${M:?}/$name"; log "train $name"
        $V train --model $M/$from --flores $F --out $M/$name --data $W/data --chat $W/$chat \
            --chat-share 0.85 --careful-wiki --batch $BATCH --accum $ACCUM --eval-every 100 --tokens 12000000 \
            --seed $seed --teacher $M/n-atlas --kl-weight 1.0 --kl-batch $KL_BATCH "$@"
        cp $M/$name/log.jsonl $R/train-$name-log.jsonl
        touch $M/$name/.trained
    fi
    $how $name
    touch $R/done-$name
    rm -rf "${M:?}/$name"
}
explore() {  # the release reading AfriXNLI through the original tokenizer (exploratory)
    [ -f $R/iroko-release-origtok/done ] && return 0
    $HF download $RELEASE --revision $RELEASE_REV --local-dir $M/release --exclude "*.gguf" >/dev/null
    echo "$RELEASE_SHA  $M/release/model.safetensors" | sha256sum -c --quiet || die "release weights differ from v6-long"
    iroko release-origtok $M/release "$(printf 'afrixnli_%s_prompt_1,' yor hau ibo eng)" ",tokenizer=$M/n-atlas"
    rm -rf "${M:?}/release"
}

for u in $UNITS; do
    case $u in
    inst)    explore; train inst 0 init-v6 chat-v8-inst.jsonl evaluate $CANON ;;
    seed)    train long-s1 1 init-v6 chat-v6-gold.jsonl irokoonly $CANON ;;
    control) train control-inst 0 n-atlas chat-v8-inst.jsonl evaluate ;;
    comet)
        [ -x /workspace/venv-comet/bin/python ] || { python -m venv --system-site-packages /workspace/venv-comet \
            && /workspace/venv-comet/bin/pip install -q "unbabel-comet==2.2.7"; }
        log comet; /workspace/venv-comet/bin/python -I /workspace/africomet.py $R $R/heldout-*.json ;;
    *) die "unknown unit: $u" ;;
    esac
done
rm -rf "${M:?}/init-v6"
log ALL-DONE
