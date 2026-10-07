#!/bin/bash
# Follow-up to sweep 3: keep training the strategies whose learning curves were still rising.
#   M (imitation -> PPO): resume each run for M_EXTRA more steps.
#   L (residual PPO on the strike controller): resume each run for L_EXTRA more steps.
# Every resumed run ends with the standard 200-shot test (seed 30000).
# Submit from the repository root after sweep 3 finishes:
#   sbatch -A OTH24028 -p normal --dependency=afterany:<sweep3 job> \
#          --export=ALL,SWEEP_DIR=$WORK/airhockey-runs/sweep3-dev scripts/tacc/followup3.slurm
#SBATCH --job-name=striker-followup3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=04:00:00
#SBATCH --output=logs/striker-followup3-%j.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
: "${SWEEP_DIR:?Set SWEEP_DIR}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
PY="${TACC_PYTHON:-$WORK/venvs/precision-striker-cpu/bin/python}"
BOUNCE=(--require-hit --timeout-penalty 10 --obstacle-bounce)
resume() {  # name source extra_steps envs task-flags...
    local name="$1" src="$2" steps="$3" envs="$4"; shift 4
    local out="$SWEEP_DIR/$name"
    {
        "$PY" -u -m scripts.train_precision_striker --envs "$envs" --vector subproc --rollout 512 --device cpu \
            --seed 7 "${BOUNCE[@]}" "$@" --steps "$steps" --obstacles 1 --resume "$src" --output "$out" &&
        "$PY" -u -m scripts.eval_precision_striker --run "$out" --output "$out/evaluation" \
            --seed 30000 --layouts 10 --shots 20 --obstacles 1
    } > "$out.log" 2>&1 &
}
PLANNER="$(cat "${PLANNER_FILE:-planner_settings.json}")"
for s in 1 2 3; do
    resume "M-seed$s-more" "$SWEEP_DIR/M-seed$s" "${M_EXTRA:-8000000}" 8 --extra-obs --goal-shaping
    if [[ -f "$SWEEP_DIR/L-seed$s/policy.zip" ]]; then
        resume "L-seed$s-more" "$SWEEP_DIR/L-seed$s" "${L_EXTRA:-5000000}" 16 --hybrid residual --planner "$PLANNER"
    fi
done
wait
echo "Finished follow-up. Results in $SWEEP_DIR/*-more"
