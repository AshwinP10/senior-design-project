#!/bin/bash
# Train RL policies for obstacles placed anywhere (live demo: a user places the block), 4 nodes at once.
#   node 0: W1 DAgger, 1 obstacle     node 1: W3 DAgger, 1-3 obstacles
#   node 2: WF PPO from BEST_RUN       node 3: WB DAgger, 1 obstacle, more data
# Submit from the repository root:
#   sbatch -A OTH24028 -p development --export=ALL,SWEEP_DIR=...,BEST_RUN=... scripts/tacc/wide_sweep.slurm
#SBATCH --job-name=striker-wide
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/striker-wide-%j.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
OUT="${SWEEP_DIR:-$WORK/airhockey-runs/wide-$SLURM_JOB_ID}"
mkdir -p "$OUT"
"${TACC_PYTHON:-$WORK/venvs/precision-striker-cpu/bin/python}" -m unittest discover -s tests 2>&1 | tail -2
groups=(W1 W3 WF WB)
for i in 0 1 2 3; do
    ibrun -n 1 -o "$i" bash scripts/tacc/wide_node.sh "${groups[$i]}" "$OUT" &
done
wait
echo "Finished. Results: $OUT"
