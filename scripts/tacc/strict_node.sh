#!/bin/bash
# One node's share of strict_sweep.slurm: pure-RL PPO on the strict one-contact task
# (paddle restitution 0.8, one contact, 1-3 obstacles anywhere, straight line blocked in half the
# episodes, layouts verified makeable with the trajectory calculator).
# Usage: strict_node.sh GROUP OUT_DIR
#   SA  PPO fine-tune of W1-seed2 (DAgger+PPO net), seeds 1-3, entropy 0.005
#   SB  PPO fine-tune of WB-seed5, seeds 1-3, entropy 0.005
#   SC  PPO fine-tune of W1-seed2, lower learning rate 1e-4 with linear decay, seeds 4-6
#   SR  PPO from scratch (no teacher, no imitation at all), 256-wide net, seeds 1-3
set -uo pipefail
GROUP="$1"; OUT="$2"
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
PY="${TACC_PYTHON:-$WORK/venvs/precision-striker-cpu/bin/python}"
PREV="${PREV_DIR:-$WORK/airhockey-runs/wide1}"
STEPS="${STEPS:-5000000}"
TASK=(--require-hit --timeout-penalty 10 --obstacle-bounce --extra-obs --goal-shaping --wide-obstacles
      --obstacles 1 --obstacles-max 3 --paddle-restitution 0.8 --single-hit --block-line-prob 0.5
      --layout-check trajectory)
COMMON=(--envs 32 --vector subproc --rollout 256 --device cpu --checkpoint-every 250000 --steps "$STEPS")
echo "$(hostname) group $GROUP start $(date)"

suite() {  # run_dir: strict suite on validation shots (choose models here; test seed 30000 later)
    "$PY" -u -m scripts.eval_suite --run "$1" --seed 50000 --workers 4 --output "$1/suite_val50000.json"
}
train() {  # name seed flags...
    local name="$1" seed="$2"; shift 2
    { "$PY" -u -m scripts.train_precision_striker "${COMMON[@]}" "${TASK[@]}" --seed "$seed" "$@" \
          --output "$OUT/$name" && suite "$OUT/$name"; } > "$OUT/$name.log" 2>&1 &
}
case "$GROUP" in
    SA) for s in 1 2 3; do train "SA-seed$s" "$s" --resume "$PREV/W1-seed2" --resume-new-task --resume-ent-coef 0.005; done ;;
    SB) for s in 1 2 3; do train "SB-seed$s" "$s" --resume "$PREV/WB-seed5" --resume-new-task --resume-ent-coef 0.005; done ;;
    SC) for s in 4 5 6; do train "SC-seed$s" "$s" --resume "$PREV/W1-seed2" --resume-new-task --resume-ent-coef 0.005 \
                               --lr 1e-4 --lr-schedule linear; done ;;
    SR) for s in 1 2 3; do train "SR-seed$s" "$s" --net 256 --ent-coef 0.005 --lr 3e-4; done ;;
esac
wait
echo "$(hostname) group $GROUP done $(date)"
