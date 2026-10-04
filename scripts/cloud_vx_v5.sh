#!/usr/bin/env bash
# v5: on-policy self-distillation across tokenizers, starting from v4
# (docs/HISTORY.md, section 5). Two seeds, scored on NTREX (500) and FLORES (200)
# against the original and against v4, then exported to GGUF.
#   HF_MODEL=<private repo holding v4> bash /workspace/cloud_vx_v5.sh
# Needs $W/prior/ with the validation run's original and v4 samples
# (scripts/cloud_vx_upload.sh copies them), so they are not regenerated.
set -euo pipefail
export HF_HOME=${HF_HOME:-/workspace/.cache/huggingface}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
: "${HF_MODEL:?set HF_MODEL to the private repo holding the v4 weights}"
W=${W:-/workspace/work}; R=$W/results-v5; mkdir -p "$R"
PY=/workspace/venv/bin/python; F=$W/data/flores200_dataset; N=$W/ntrex; P=$W/prior
STEPS=${STEPS:-300}; SEEDS=${SEEDS:-"0 1"}
log() { echo "== $(date +%H:%M:%S) $*"; }

log setup
$PY -m pip install -q "transformers==5.17.0" safetensors "huggingface_hub[hf_xet]" sacrebleu
$PY -c "import torch; assert torch.cuda.is_available(), 'no CUDA GPU'"
[ -f $W/n-atlas/config.json ] || /workspace/venv/bin/hf download NCAIR1/N-ATLaS --local-dir $W/n-atlas --exclude "*.gguf" --exclude "original/*"
[ -f $W/v4/config.json ] || /workspace/venv/bin/hf download "$HF_MODEL" --local-dir $W/v4
cat $W/chat-distill.jsonl $W/chat-wiki.jsonl > $W/chat-all.jsonl

cd /workspace/wazocore
V="$PY -m wazocore.vx_torch"
for s in $SEEDS; do
    log "onpolicy seed $s"
    $V onpolicy --model $W/v4 --teacher $W/n-atlas --chat $W/chat-all.jsonl --out $W/v5-s$s --steps $STEPS --seed $s
    cp $W/v5-s$s/log.jsonl $R/onpolicy-s$s-log.jsonl
    log "sample v5-s$s"
    $V sample --model $W/v5-s$s --flores $F --ntrex $N --n 500 --out $R/ntrex-v5-s$s.json 2>/dev/null
    $V sample --model $W/v5-s$s --flores $F --n 200 --out $R/flores-v5-s$s.json 2>/dev/null
    for t in ntrex flores; do
        log "score v5-s$s on $t";  $V score $P/$t-n-atlas.json $R/$t-v5-s$s.json --out $R/$t-v5-s$s-vs-original.json
        $V score $P/$t-v4.json $R/$t-v5-s$s.json --out $R/$t-v5-s$s-vs-v4.json
    done
done
log spelling-and-loops
$PY - "$P" "$R" <<'PY'
import json, sys, unicodedata
from pathlib import Path
from wazocore.vx_core import looping
P, R = Path(sys.argv[1]), Path(sys.argv[2])
files = {"original": P / "ntrex-n-atlas.json", "v4": P / "ntrex-v4.json", **{f.stem.replace("ntrex-", ""): f for f in sorted(R.glob("ntrex-v5-s?.json"))}}
out = {}
for name, f in files.items():
    rows = json.load(open(f))["rows"]
    rate = lambda lang, chars: (lambda t: sum(c in chars for c in unicodedata.normalize("NFC", t)) / max(1, len(t)) * 100)("".join(r["output"] for r in rows if r["lang"] == lang))
    out[name] = {"loops": sum(looping(r["output"]) for r in rows), "stopped": sum(r["stopped"] for r in rows), "n": len(rows),
                 "ha_hooked_per_100": round(rate("ha", "ƙɗɓƘƊƁ"), 3), "ig_dots_per_100": round(rate("ig", "ịọụṅỊỌỤṄ"), 3),
                 "yo_marks_per_100": round(rate("yo", "ẹọṣẸỌṢàáèéìíòóùúÀÁÈÉÌÍÒÓÙÚ"), 3)}
    print(name, out[name])
json.dump(out, open(R / "spelling-loops.json", "w"), indent=1)
PY
log gguf; bash /workspace/cloud_gguf.sh $W/v5-s0 $W/WazoCore-8B-v5.Q4_K_M.gguf $R
log ALL-DONE
