#!/usr/bin/env bash
# Pulls a run's results (RUN=v7 or v8, default v7) from the pod every few
# minutes, so a lost Community
# Cloud host costs at most the step it was on. Exits once the run is done;
# with STOP=1 it then stops the pod, which bills until stopped.
#   [SSH_KEY=~/.ssh/key] [RUN=v8] [EVERY=300] [STOP=1] caffeinate -i scripts/cloud_pull.sh HOST PORT
set -euo pipefail
host=${1:?host}; port=${2:?port}
root=$(cd "$(dirname "$0")/.." && pwd); r=${RUN:-v7}; out=$root/results/vocab/$r; run=$out/run-$host-$port.log
mkdir -p "$out"
e="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -i ${SSH_KEY:-$HOME/.ssh/id_ed25519} -p $port"
while :; do
    if rsync -az -e "$e" --exclude '*.part' root@$host:/workspace/work/results-$r/ "$out"/ \
        && rsync -az -e "$e" root@$host:/workspace/run.log "$run"; then
        echo "$(date +%H:%M) $(grep '^== ' "$run" | tail -1)"
    else
        echo "$(date +%H:%M) pod unreachable; results so far are in $out"
    fi
    # the v7 and v8 data steps run v6, which logs its own ALL-DONE first
    ends=$(grep -c ALL-DONE "$run" 2>/dev/null) || true
    if [ "${ends:-0}" -ge 2 ] && tail -1 "$run" | grep -q ALL-DONE; then
        echo "done: results in $out"
        [ "${STOP:-0}" = 1 ] || exit 0
        # runpodctl and its pod-scoped key live in the container's environment, not the SSH session's
        $e root@$host 'set -a; . <(tr "\0" "\n" </proc/1/environ | grep "^RUNPOD_"); runpodctl stop pod "$RUNPOD_POD_ID"' \
            || echo "could not stop the pod: stop it in the RunPod console"
        exit 0
    fi
    sleep "${EVERY:-300}"
done
