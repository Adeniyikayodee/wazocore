#!/usr/bin/env bash
# v7: the evidence preregistered in docs/PLAN-v7.md. Built for RunPod
# Community Cloud: a host can vanish and has no network volume, so every
# step is skipped once its result exists, results are small files the
# laptop pulls as they land (scripts/cloud_pull.sh), and no weights leave
# the pod. Run after scripts/cloud_vx_upload.sh and `hf auth login`:
#   bash /workspace/cloud_vx_v7.sh
# UNITS, in priority order (default all):
#   base     the original and the release, evaluated
#   control  control at the release budget (12M tokens), seed 0
#   fvt      the standard recipe: plain-mean init, plain continued training
#   ablate   the release minus one part each: no-canon, no-kl, old-anchor, no-calib
#   seeds    release seeds 1 and 2, control seeds 1 and 2
#   comet    AfriCOMET on every held-out sample file
# Every model: NTREX-128 lines 500-1996 both ways and BPC on FLORES devtest
# (English included); IrokoBench for the original, the release and control.
# On an out-of-memory error lower KL_BATCH to 2, then BATCH to 4 with ACCUM 4.
set -euo pipefail
export HF_HOME=${HF_HOME:-/workspace/.cache/huggingface}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=${W:-/workspace/work}; R=$W/results-v7; M=$W; mkdir -p "$R"
UNITS=${UNITS:-"base control fvt ablate seeds comet"}
KL_BATCH=${KL_BATCH:-4}; BATCH=${BATCH:-8}; ACCUM=${ACCUM:-2}
PY=/workspace/venv/bin/python; HF=/workspace/venv/bin/hf; F=$W/data/flores200_dataset; N=$W/ntrex
BASE_REV=e294476928aca9030e924ca27bb8e085e8581273
RELEASE=mkbadeniyi/WazoCore-8B-v6-long; RELEASE_REV=be50a5df45de4abaecd1bd51231987fe24bacf1f
RELEASE_SHA=b4cc2a6891529ff1827ee66699eb82aafbb23588d6f0988e74577644168db381
IROKO="n-atlas release control-s0"
CANON="--anchor canonical --canon-weight 1.0"
log() { echo "== $(date +%H:%M:%S) $*"; }
die() { echo "!! $*" >&2; exit 1; }

log preflight  # fail in the first minute, not the fifth hour
mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
[ "$mem" -ge 79000 ] || die "need an 80 GB GPU, found $mem MiB"
[ "$(df --output=avail -BG "$W" | tail -1 | tr -dc 0-9)" -ge 150 ] || die "need 150 GB free in $W"
[ -s $W/chat-wiki.jsonl ] || die "upload data/runs/cloud/chat-wiki.jsonl: the release was trained on it"
$HF download NCAIR1/N-ATLaS --revision $BASE_REV --local-dir $M/n-atlas --exclude "*.gguf" --exclude "original/*" >/dev/null \
    || die "no access to N-ATLaS: $HF auth login"

log data; W=$W ARMS=data GGUF=none bash /workspace/cloud_vx_v6.sh
(cd $W && sha256sum chat-wiki.jsonl chat-v6-gold.jsonl vx-tok/new_tokens.json) > $R/inputs.now
[ -s $R/inputs.sha256 ] || cp $R/inputs.now $R/inputs.sha256
cmp -s $R/inputs.now $R/inputs.sha256 || die "inputs differ from the first pod's: see $R/inputs.sha256"
rm $R/inputs.now
{ date -u +%FT%TZ; cat /workspace/wazocore/REVISION 2>/dev/null; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
  $PY -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'transformers', transformers.__version__)"; } >> $R/env.txt

cd /workspace/wazocore
V="$PY -m wazocore.vx_torch"
iroko() {  # NAME: IrokoBench, zero-shot, prompt 1, chat template
    case " $IROKO " in *" $1 "*) ;; *) return 0 ;; esac
    [ -f $R/iroko-$1/done ] && return 0
    [ -x /workspace/venv-eval/bin/lm_eval ] || { python -m venv --system-site-packages /workspace/venv-eval \
        && /workspace/venv-eval/bin/pip install -q "lm-eval[hf]==0.4.13" "transformers==5.17.0"; }
    local t; t=$(for l in yor hau ibo eng; do printf 'afrixnli_%s_prompt_1,afrimmlu_direct_%s_prompt_1,afrimgsm_%s_prompt_1,' $l $l $l; done)
    log "irokobench $1"
    /workspace/venv-eval/bin/lm_eval --model hf --model_args pretrained=$M/$1,dtype=bfloat16 --tasks ${t%,} \
        --apply_chat_template --batch_size auto --seed 0 --log_samples --output_path $R/iroko-$1
    mkdir -p $R/iroko-$1 && touch $R/iroko-$1/done
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
    iroko $1
    touch $R/done-$1
}
init() {  # NAME: an expanded, untrained model
    [ -f $M/$1/.done ] && return 0
    local how; case $1 in
        init-v6) how="--out-weights chars --calibrate-input-norm" ;;
        init-nocal) how="--out-weights chars" ;;
        init-fvt) how="--out-weights mean" ;;
    esac
    rm -rf "${M:?}/$1"; log "expand $1"
    $V expand --base $M/n-atlas --vx $W/vx-tok --out $M/$1 $how
    touch $M/$1/.done
}
train() {  # NAME SEED INIT [flags]: v6's recipe at the release budget, then evaluate and free the disk
    local name=$1 seed=$2 from=$3; shift 3
    [ -f $R/done-$name ] && { log "$name: done"; return 0; }
    [ $from = n-atlas ] || init $from
    if [ ! -f $M/$name/.trained ]; then
        rm -rf "${M:?}/$name"; log "train $name"
        $V train --model $M/$from --flores $F --out $M/$name --data $W/data --chat $W/chat-v6-gold.jsonl \
            --chat-share 0.85 --careful-wiki --batch $BATCH --accum $ACCUM --eval-every 100 --tokens 12000000 \
            --seed $seed --teacher $M/n-atlas --kl-weight 1.0 --kl-batch $KL_BATCH "$@"
        cp $M/$name/log.jsonl $R/train-$name-log.jsonl
        touch $M/$name/.trained
    fi
    evaluate $name
    rm -rf "${M:?}/$name"
}

for u in $UNITS; do
    case $u in
    base)
        evaluate n-atlas
        if [ ! -f $R/done-release ]; then
            $HF download $RELEASE --revision $RELEASE_REV --local-dir $M/release --exclude "*.gguf" >/dev/null
            echo "$RELEASE_SHA  $M/release/model.safetensors" | sha256sum -c --quiet || die "release weights differ from v6-long"
            evaluate release
            rm -rf $M/release
        fi ;;
    control) train control-s0 0 n-atlas ;;
    fvt)     train fvt 0 init-fvt --anchor old --kl-weight 0 ;;
    ablate)
        train no-canon 0 init-v6 --anchor canonical --canon-weight 0
        train no-kl 0 init-v6 --anchor old --canon-weight 1.0 --kl-weight 0
        train old-anchor 0 init-v6 --anchor old --canon-weight 1.0
        train no-calib 0 init-nocal $CANON ;;
    seeds)
        train long-s1 1 init-v6 $CANON
        train long-s2 2 init-v6 $CANON
        train control-s1 1 n-atlas
        train control-s2 2 n-atlas ;;
    comet)
        [ -x /workspace/venv-comet/bin/python ] || { python -m venv --system-site-packages /workspace/venv-comet \
            && /workspace/venv-comet/bin/pip install -q "unbabel-comet==2.2.7"; }
        log comet; /workspace/venv-comet/bin/python -I /workspace/africomet.py $R $R/heldout-*.json ;;
    *) die "unknown unit: $u" ;;
    esac
done
log ALL-DONE
