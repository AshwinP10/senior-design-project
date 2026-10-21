#!/bin/bash
# One node's share of bank_sweep.slurm: continue from BASE_RUN, the aim-reward run that learned bank shots
# (aim-long/AB-seed3: 85.5% with the straight line blocked, but 62.5% with one random block), and try to
# keep the bank shots while getting straight shots back. Same strict task as aim_node.sh.
# Usage: bank_node.sh GROUP OUT_DIR
#   BA  fine-tune BASE_RUN, aim 5, learning rate 1e-4 decaying to 0, seeds 1-3
#   BB  fine-tune BASE_RUN, aim 5, learning rate 3e-4, seeds 4-6
#   BC  fine-tune BASE_RUN, aim 5, straight line blocked in only 30% of episodes, seeds 7-9
#   BD  learning curve of BASE_RUN: strict suite on its checkpoints (to see when bank shots appeared)
set -uo pipefail
GROUP="$1"; OUT="$2"
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
PY="${TACC_PYTHON:-$WORK/venvs/precision-striker-cpu/bin/python}"
BASE="${BASE_RUN:?set BASE_RUN}"
STEPS="${STEPS:-5000000}"
TASK=(--require-hit --timeout-penalty 10 --obstacle-bounce --extra-obs --goal-shaping --wide-obstacles
      --obstacles 1 --obstacles-max 3 --paddle-restitution 0.8 --single-hit
      --layout-check trajectory --aim-shaping 5)
COMMON=(--envs 32 --vector subproc --rollout 256 --device cpu --checkpoint-every 250000 --steps "$STEPS"
        --resume "$BASE" --resume-new-task --resume-ent-coef 0.005)
echo "$(hostname) group $GROUP start $(date)"

suite() {  # run_dir: strict suite on validation shots
    "$PY" -u -m scripts.eval_suite --run "$1" --seed 50000 --workers 4 --output "$1/suite_val50000.json"
}
train() {  # name seed flags...
    local name="$1" seed="$2"; shift 2
    { "$PY" -u -m scripts.train_precision_striker "${COMMON[@]}" "${TASK[@]}" --seed "$seed" "$@" \
          --output "$OUT/$name" && suite "$OUT/$name"; } > "$OUT/$name.log" 2>&1 &
}
case "$GROUP" in
    BA) for s in 1 2 3; do train "BA-seed$s" "$s" --block-line-prob 0.5 --lr 1e-4 --lr-schedule linear; done ;;
    BB) for s in 4 5 6; do train "BB-seed$s" "$s" --block-line-prob 0.5 --lr 3e-4; done ;;
    BC) for s in 7 8 9; do train "BC-seed$s" "$s" --block-line-prob 0.3; done ;;
    BD) CK="$BASE/checkpoints"; CURVE="$OUT/BD-curve"; mkdir -p "$CURVE"
        steps=$(ls "$CK" | grep -o 'policy_[0-9]*_steps.zip' | grep -o '[0-9]*' | sort -n | awk 'NR % 8 == 0')
        for n in $steps; do
            d="$CURVE/ckpt-$n"; mkdir -p "$d"
            cp "$CK/policy_${n}_steps.zip" "$d/policy.zip"; cp "$CK/policy_vecnormalize_${n}_steps.pkl" "$d/vecnormalize.pkl"
            cp "$BASE/manifest.json" "$d/"
            "$PY" -u -m scripts.eval_suite --run "$d" --seed 50000 --shots 100 --workers 4 \
                --output "$d/suite_val50000.json" > "$d.log" 2>&1 &
            while [ "$(jobs -r | wc -l)" -ge 28 ]; do sleep 5; done
        done ;;
esac
wait
echo "$(hostname) group $GROUP done $(date)"
