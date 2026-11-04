#!/bin/bash
# Imitation (DAgger from the sim-corrected physics striker) vs imitation + PPO, 4 nodes (groups D1-D4, see imitation_node.sh).
# Submit from the repository root:
#   sbatch -A OTH24028 -p normal --export=ALL,SWEEP_DIR=...,PPO_STEPS=20000000 scripts/tacc/imitation_sweep.slurm
#SBATCH --job-name=striker-imitation
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/striker-imitation-%j.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
OUT="${SWEEP_DIR:-$WORK/airhockey-runs/imitation-$SLURM_JOB_ID}"
mkdir -p "$OUT"
"${TACC_PYTHON:-$WORK/venvs/precision-striker-cpu/bin/python}" -m unittest discover -s tests 2>&1 | tail -2
groups=(${SWEEP_GROUPS:-D1 D2 D3 D4})   # SWEEP_GROUPS="H1 H2 H3 H4" for the hint study
for i in 0 1 2 3; do
    ibrun -n 1 -o "$i" bash scripts/tacc/imitation_node.sh "${groups[$i]}" "$OUT" &
done
wait
echo "Finished. Results: $OUT"
