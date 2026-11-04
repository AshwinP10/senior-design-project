#!/bin/bash
# Supervise one imitation sweep to the end: training job -> report job -> push to GitHub main.
# A training or report job that ends in anything but COMPLETED is resubmitted (up to 2 more times; a retry
# starts in a fresh <sweep>-retryN folder). The push runs here on the login node, which can reach GitHub.
# Run it inside tmux on a TACC login node, from the code folder, so it keeps going without a laptop:
#   tmux new-session -d -s striker 'bash scripts/tacc/supervise.sh imitation-hints "H1 H2 H3 H4" ITERS=10,PPO_STEPS=20000000'
# Progress: logs/supervise-<name>.log.   Attach: tmux attach -t striker   (detach: Ctrl-b then d)
set -uo pipefail
NAME="$1"; SWEEP_GROUPS="$2"; EXPORTS="${3:-}"
CODE="$(pwd)"
LOG="$CODE/logs/supervise-$NAME.log"
ACCOUNT="${ACCOUNT:-OTH24028}"
exec >> "$LOG" 2>&1 < /dev/null
say() { echo "$(date '+%F %T') $*"; }
jobid() { grep -oE '^[0-9]+' | tail -1; }                     # sbatch --parsable, minus TACC's banner
state() { sacct -j "$1" -X -n -o State 2>/dev/null | head -1 | tr -d ' +'; }
wait_job() { while squeue -h -j "$1" 2>/dev/null | grep -q .; do sleep 600; done; sleep 60; state "$1"; }

for attempt in 0 1 2; do
    SWEEP="$WORK/airhockey-runs/$NAME$([ "$attempt" = 0 ] || echo "-retry$attempt")"
    if [ -f "$SWEEP/report/summary.json" ]; then
        say "report already exists in $SWEEP"
    else
        if [ -f "$SWEEP/TRAINING_DONE" ]; then
            say "training already done in $SWEEP"
        else
            J=$(sbatch --parsable -A "$ACCOUNT" -p normal -t 24:00:00 \
                --export=ALL,SWEEP_DIR="$SWEEP",SWEEP_GROUPS="$SWEEP_GROUPS"${EXPORTS:+,$EXPORTS} \
                scripts/tacc/imitation_sweep.slurm | jobid)
            say "training job $J submitted (attempt $attempt, $SWEEP)"
            S=$(wait_job "$J"); say "training job $J ended: $S"
            [ "$S" = COMPLETED ] || continue
            date > "$SWEEP/TRAINING_DONE"
        fi
        for r in 1 2 3; do
            R=$(sbatch --parsable -A "$ACCOUNT" -p development -t 02:00:00 --export=ALL,SWEEP_DIR="$SWEEP" \
                scripts/tacc/imitation_report.slurm | jobid)
            say "report job $R submitted"
            S=$(wait_job "$R"); say "report job $R ended: $S"
            [ -f "$SWEEP/report/summary.json" ] && break
        done
    fi
    if [ -f "$SWEEP/report/summary.json" ]; then
        for p in 1 2 3; do
            if bash scripts/tacc/publish_results.sh "$SWEEP" "$(basename "$SWEEP")"; then
                say "published $SWEEP to GitHub main"; date > "$SWEEP/SUPERVISOR_DONE"; exit 0
            fi
            sleep 900
        done
        say "publishing failed 3 times; results stay in $SWEEP"; exit 1
    fi
done
say "gave up after 3 training attempts"
exit 1
