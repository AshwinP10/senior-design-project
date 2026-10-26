#!/bin/bash
# One node's share of aim_sweep.slurm: pure-RL PPO on the strict one-contact task (as strict_node.sh)
# plus the aim reward (--aim-shaping): credit, when the puck leaves the paddle, for heading along a clear
# straight or bank line into the goal.
# Usage: aim_node.sh GROUP OUT_DIR
#   AA  fine-tune SNAP_RUN (best strict checkpoint), aim 5, seeds 1-3
#   AB  fine-tune W1-seed2 (DAgger+PPO net), aim 5, seeds 1-3
#   AC  fine-tune SNAP_RUN, aim 10, entropy 0.01, seeds 4-6
#   AR  PPO from scratch (no teacher, no imitation), aim 5, 256-wide net, seeds 1-3
set -uo pipefail
GROUP="$1"; OUT="$2"
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
PY="${TACC_PYTHON:-$WORK/venvs/precision-striker-cpu/bin/python}"
PREV="${PREV_DIR:-$WORK/airhockey-runs/wide1}"
SNAP="${SNAP_RUN:?set SNAP_RUN}"
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
FT=(--resume-new-task --resume-ent-coef 0.005)
case "$GROUP" in
    AA) for s in 1 2 3; do train "AA-seed$s" "$s" --resume "$SNAP" "${FT[@]}" --aim-shaping 5; done ;;
    AB) for s in 1 2 3; do train "AB-seed$s" "$s" --resume "$PREV/W1-seed2" "${FT[@]}" --aim-shaping 5; done ;;
    AC) for s in 4 5 6; do train "AC-seed$s" "$s" --resume "$SNAP" --resume-new-task --resume-ent-coef 0.01 \
                               --aim-shaping 10; done ;;
    AR) for s in 1 2 3; do train "AR-seed$s" "$s" --net 256 --ent-coef 0.005 --aim-shaping 5; done ;;
esac
wait
echo "$(hostname) group $GROUP done $(date)"
