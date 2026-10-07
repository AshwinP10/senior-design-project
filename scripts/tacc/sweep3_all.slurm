#!/bin/bash
# Sweep 3 on four nodes at once (fits the 2-hour development queue):
#   node 0: J shot-choice PPO   node 1: K shot-choice SAC + L residual PPO
#   node 2: M imitation -> PPO  node 3: strike-planner tuning/validation/test
# RL runs use PLANNER_JSON (strike controller settings chosen on validation shots).
# Submit from the repository root:
#   sbatch -A OTH24028 -p development --export=ALL,PLANNER_JSON='{...}' scripts/tacc/sweep3_all.slurm
#SBATCH --job-name=striker-sweep3
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/striker-sweep3-%j.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
OUT="${SWEEP_DIR:-$WORK/airhockey-runs/sweep3-$SLURM_JOB_ID}"
mkdir -p "$OUT"
# Planner settings come from PLANNER_FILE (JSON) because sbatch --export splits on commas.
if [[ -z "${PLANNER_JSON:-}" && -f "${PLANNER_FILE:-planner_settings.json}" ]]; then
    export PLANNER_JSON="$(cat "${PLANNER_FILE:-planner_settings.json}")"
fi
echo "Planner settings: ${PLANNER_JSON:-defaults}"
"${TACC_PYTHON:-$WORK/venvs/precision-striker-cpu/bin/python}" -m unittest discover -s tests 2>&1 | tail -3
groups=(J KL M P)
for i in 0 1 2 3; do
    ibrun -n 1 -o "$i" bash scripts/tacc/node_task.sh "${groups[$i]}" "$OUT" &
done
wait
echo "Finished. Results: $OUT"
