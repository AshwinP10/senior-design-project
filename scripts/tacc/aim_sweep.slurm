#!/bin/bash
# Pure-RL training on the strict task plus the aim reward, 4 nodes (groups AA AB AC AR, see aim_node.sh).
# Submit from the repository root:
#   sbatch -A OTH24028 -p development --export=ALL,SWEEP_DIR=...,SNAP_RUN=...,STEPS=5000000 scripts/tacc/aim_sweep.slurm
#SBATCH --job-name=striker-aim
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/striker-aim-%j.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
OUT="${SWEEP_DIR:-$WORK/airhockey-runs/aim-$SLURM_JOB_ID}"
mkdir -p "$OUT"
"${TACC_PYTHON:-$WORK/venvs/precision-striker-cpu/bin/python}" -m unittest discover -s tests 2>&1 | tail -2
groups=(AA AB AC AR)
for i in 0 1 2 3; do
    ibrun -n 1 -o "$i" bash scripts/tacc/aim_node.sh "${groups[$i]}" "$OUT" &
done
wait
echo "Finished. Results: $OUT"
