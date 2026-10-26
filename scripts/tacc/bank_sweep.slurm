#!/bin/bash
# Continue the bank-shot run on the strict task, 4 nodes (groups BA BB BC BD, see bank_node.sh).
# Submit from the repository root:
#   sbatch -A OTH24028 -p development --export=ALL,SWEEP_DIR=...,BASE_RUN=...,STEPS=5000000 scripts/tacc/bank_sweep.slurm
#SBATCH --job-name=striker-bank
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/striker-bank-%j.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
OUT="${SWEEP_DIR:-$WORK/airhockey-runs/bank-$SLURM_JOB_ID}"
mkdir -p "$OUT"
"${TACC_PYTHON:-$WORK/venvs/precision-striker-cpu/bin/python}" -m unittest discover -s tests 2>&1 | tail -2
groups=(BA BB BC BD)
for i in 0 1 2 3; do
    ibrun -n 1 -o "$i" bash scripts/tacc/bank_node.sh "${groups[$i]}" "$OUT" &
done
wait
echo "Finished. Results: $OUT"
