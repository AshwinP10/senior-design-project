#!/bin/bash
# Login-node backup for publish_results.slurm (compute nodes may have no internet). Checks every 15 minutes;
# once a sweep's report exists and its report job has ended, publishes it unless it is already published.
# Start detached:  nohup setsid bash scripts/tacc/publish_watch.sh REPORT_JOB:SWEEP_DIR:NAME ... > logs/publish_watch.log 2>&1 &
cd "$(dirname "$0")/../.."
end=$(( $(date +%s) + 5 * 24 * 3600 ))          # give up after 5 days
while [ "$(date +%s)" -lt "$end" ]; do
    pending=0
    for spec in "$@"; do
        IFS=: read -r job sweep name <<< "$spec"
        [ -f "$sweep/PUBLISHED" ] && continue
        pending=1
        if ! squeue -h -j "$job" 2>/dev/null | grep -q . && [ -d "$sweep/report" ]; then
            sleep 900                                   # give the compute-node publish job its chance first
            [ -f "$sweep/PUBLISHED" ] || bash scripts/tacc/publish_results.sh "$sweep" "$name"
        fi
    done
    [ "$pending" = 0 ] && { echo "all published $(date)"; exit 0; }
    sleep 900
done
echo "watcher stopped after 5 days $(date)"
