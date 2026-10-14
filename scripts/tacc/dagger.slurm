#!/bin/bash
# DAgger then PPO fine-tuning: 3 independent runs side by side on one node.
#   1. scripts/train_dagger.py   : iterative imitation of the strike controller (best round on validation)
#   2. train_precision_striker   : PPO fine-tuning from the DAgger network (FT_STEPS)
#   3. eval_precision_striker    : the 200 test shots (seed 30000)
# The controller settings come from planner_settings.json in the repository root.
# Submit from the repository root:
#   sbatch -A OTH24028 -p normal --export=ALL,SWEEP_DIR=$WORK/airhockey-runs/dagger1 scripts/tacc/dagger.slurm
#SBATCH --job-name=striker-dagger
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=04:00:00
#SBATCH --output=logs/striker-dagger-%j.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
OUT="${SWEEP_DIR:-$WORK/airhockey-runs/dagger-$SLURM_JOB_ID}"
mkdir -p "$OUT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
PY="${TACC_PYTHON:-$WORK/venvs/precision-striker-cpu/bin/python}"
PLANNER="$(cat "${PLANNER_FILE:-planner_settings.json}")"
"$PY" -m unittest discover -s tests 2>&1 | tail -2
for s in 1 2 3; do
    {
        "$PY" -u -m scripts.train_dagger --output "$OUT/D-seed$s-dagger" --seed "$s" --workers 34 \
            --iterations "${ITERATIONS:-8}" --steps-per-iter "${STEPS_PER_ITER:-400000}" \
            --planner "$PLANNER" --extra-obs --goal-shaping &&
        "$PY" -u -m scripts.train_precision_striker --envs 8 --vector subproc --rollout 512 --device cpu \
            --seed "$s" --require-hit --timeout-penalty 10 --obstacle-bounce --extra-obs --goal-shaping \
            --steps "${FT_STEPS:-8000000}" --obstacles 1 --resume "$OUT/D-seed$s-dagger" --output "$OUT/D-seed$s" &&
        "$PY" -u -m scripts.eval_precision_striker --run "$OUT/D-seed$s" --output "$OUT/D-seed$s/evaluation" \
            --seed 30000 --layouts 10 --shots 20 --obstacles 1 &&
        "$PY" -u -m scripts.eval_precision_striker --run "$OUT/D-seed$s-dagger" --output "$OUT/D-seed$s-dagger/evaluation" \
            --seed 30000 --layouts 10 --shots 20 --obstacles 1
    } > "$OUT/D-seed$s.log" 2>&1 &
done
wait
echo "Finished. Results in $OUT"
