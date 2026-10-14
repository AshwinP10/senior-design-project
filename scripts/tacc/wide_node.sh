#!/bin/bash
# One node's share of wide_sweep.slurm (obstacles anywhere a user may place them).
# Usage: wide_node.sh GROUP OUT_DIR
#   W1  DAgger, 1 obstacle anywhere, seeds 1-3        W3  DAgger, 1-3 obstacles anywhere, seeds 1-3
#   WF  PPO from the 79.5% network on wide placements  WB  DAgger, 1 obstacle, more data, seeds 4-5
set -uo pipefail
GROUP="$1"; OUT="$2"
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
PY="${TACC_PYTHON:-$WORK/venvs/precision-striker-cpu/bin/python}"
TEACHER="$(cat planner_tuned.json)"
BASE=(--require-hit --timeout-penalty 10 --obstacle-bounce --extra-obs --goal-shaping --wide-obstacles)
FT_STEPS="${FT_STEPS:-4000000}"
echo "$(hostname) group $GROUP start $(date)"

evaluate() {  # run_dir
    "$PY" -u -m scripts.eval_precision_striker --run "$1" --output "$1/evaluation" \
        --seed 30000 --layouts 10 --shots 20 --obstacles 1
}
dagger_then_ppo() {  # name seed workers steps_per_iter extra_flags...
    local name="$1" seed="$2" workers="$3" per="$4"; shift 4
    {
        "$PY" -u -m scripts.train_dagger --output "$OUT/$name-dagger" --seed "$seed" --workers "$workers" \
            --iterations 6 --steps-per-iter "$per" --planner "$TEACHER" --extra-obs --goal-shaping --wide-obstacles "$@" &&
        evaluate "$OUT/$name-dagger" &&
        "$PY" -u -m scripts.train_precision_striker --envs 8 --vector subproc --rollout 512 --device cpu \
            --seed "$seed" "${BASE[@]}" "$@" --steps "$FT_STEPS" --obstacles 1 \
            --resume "$OUT/$name-dagger" --output "$OUT/$name" &&
        evaluate "$OUT/$name"
    } > "$OUT/$name.log" 2>&1 &
}
case "$GROUP" in
    W1) for s in 1 2 3; do dagger_then_ppo "W1-seed$s" "$s" 34 400000; done ;;
    W3) for s in 1 2 3; do dagger_then_ppo "W3-seed$s" "$s" 34 400000 --obstacles-max 3; done ;;
    WB) for s in 4 5; do dagger_then_ppo "WB-seed$s" "$s" 50 600000; done ;;
    WF) for s in 1 2 3; do
            {
                "$PY" -u -m scripts.train_precision_striker --envs 16 --vector subproc --rollout 512 --device cpu \
                    --seed "$s" "${BASE[@]}" --steps "${WF_STEPS:-6000000}" --obstacles 1 \
                    --resume "${BEST_RUN:?set BEST_RUN}" --resume-new-task --output "$OUT/WF-seed$s" &&
                evaluate "$OUT/WF-seed$s"
            } > "$OUT/WF-seed$s.log" 2>&1 &
        done ;;
esac
wait
echo "$(hostname) group $GROUP done $(date)"
