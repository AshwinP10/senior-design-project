#!/bin/bash
# One sweep run: train (optionally in two curriculum stages), then the historical evaluation.
# Usage: run_variant.sh PYTHON VARIANT SEED TOTAL_STEPS CURRICULUM_STEPS OUTPUT_DIR
#   Sweep 1 (collisions end the shot):
#     A  v2 task                          B  v3: goal needs a hit, timeout cost 10
#     C  B + easy stage first
#   Sweep 2 (puck bounces off obstacles; only goals after a hit count):
#     E  bounce rule + easy stage         F  fine-tune sweep-1 C-seed<SEED> on E's task
#     G  E + extra inputs (hit, time) + goal-mouth shaping
#     H  G + 256-wide nets, linear LR decay, gamma 0.995, minibatch 1024, entropy 0.005
#     I  H with 16 simulations per run (8,192 steps per update)
#   Sweep 3 (strike controller; PLANNER_JSON = planner settings as JSON, default {}):
#     J  shot choice with PPO: policy picks aim/speed/interception once per shot (TOTAL_STEPS = shots)
#     K  shot choice with SAC
#     L  residual PPO: policy corrects the controller's 20 Hz command
#     M  imitation: clone the controller into a PPO network, then fine-tune with PPO
#   F needs PREV_SWEEP=<sweep-1 directory>; its TOTAL_STEPS are additional steps.
#
# CPU environment (once, on LS6; Python 3.10/3.11 required):
#   python3.11 -m venv $WORK/venvs/precision-striker-cpu
#   $WORK/venvs/precision-striker-cpu/bin/pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
#   $WORK/venvs/precision-striker-cpu/bin/pip install -r requirements-headless.in
set -euo pipefail
PY="$1"; VARIANT="$2"; SEED="$3"; STEPS="$4"; WARMUP="$5"; OUT="$6"
ENVS=8
BOUNCE=(--require-hit --timeout-penalty 10 --obstacle-bounce)
TUNED=(--net 256 --lr-schedule linear --gamma 0.995 --batch-size 1024 --ent-coef 0.005)
CURRICULUM=0
NO_SETTINGS='{}'
PLANNER="${PLANNER_JSON:-$NO_SETTINGS}"
case "$VARIANT" in
    A) TASK=() ;;
    B) TASK=(--require-hit --timeout-penalty 10) ;;
    C) TASK=(--require-hit --timeout-penalty 10); CURRICULUM=1 ;;
    E) TASK=("${BOUNCE[@]}"); CURRICULUM=1 ;;
    F) TASK=("${BOUNCE[@]}") ;;
    G) TASK=("${BOUNCE[@]}" --extra-obs --goal-shaping); CURRICULUM=1 ;;
    H) TASK=("${BOUNCE[@]}" --extra-obs --goal-shaping "${TUNED[@]}"); CURRICULUM=1 ;;
    I) TASK=("${BOUNCE[@]}" --extra-obs --goal-shaping "${TUNED[@]}"); CURRICULUM=1; ENVS=16 ;;
    J) TASK=("${BOUNCE[@]}" --hybrid shot --planner "$PLANNER" --rollout-override 64 --batch-size 512 --gamma 0.9
             --ent-coef 0.005 --lr-schedule linear); ENVS=32 ;;
    K) TASK=("${BOUNCE[@]}" --hybrid shot --planner "$PLANNER" --algo sac --batch-size 256 --gamma 0.9); ENVS=16 ;;
    L) TASK=("${BOUNCE[@]}" --hybrid residual --planner "$PLANNER" --log-std -2.0 --net 256 --lr 1e-4
             --lr-schedule linear --gamma 0.995 --batch-size 1024 --ent-coef 0.0); ENVS=16 ;;
    M) TASK=("${BOUNCE[@]}" --extra-obs --goal-shaping) ;;
    *) echo "Unknown variant $VARIANT" >&2; exit 2 ;;
esac
ROLLOUT=512
KEPT=()                         # J uses short rollouts: each step is a whole shot
for ((i = 0; i < ${#TASK[@]}; i++)); do
    if [[ "${TASK[$i]}" == --rollout-override ]]; then ROLLOUT="${TASK[$((i + 1))]}"; i=$((i + 1))
    else KEPT+=("${TASK[$i]}"); fi
done
TASK=("${KEPT[@]}")
COMMON=(--envs "$ENVS" --vector subproc --rollout "$ROLLOUT" --seed "$SEED" --device cpu)

if [[ "$CURRICULUM" == 1 ]]; then
    "$PY" -u -m scripts.train_precision_striker "${COMMON[@]}" "${TASK[@]}" \
        --steps "$WARMUP" --obstacles 0 --speed-min 0.3 --speed-max 0.8 --output "$OUT-stage1"
    "$PY" -u -m scripts.train_precision_striker "${COMMON[@]}" "${TASK[@]}" \
        --steps "$((STEPS - WARMUP))" --obstacles 1 --resume "$OUT-stage1" --output "$OUT"
elif [[ "$VARIANT" == M ]]; then
    "$PY" -u -m scripts.train_bc --output "$OUT-bc" --steps "${BC_STEPS:-1000000}" --workers 16 --seed "$SEED"         --planner "$PLANNER" --extra-obs --goal-shaping
    "$PY" -u -m scripts.train_precision_striker "${COMMON[@]}" "${TASK[@]}"         --steps "$STEPS" --obstacles 1 --resume "$OUT-bc" --output "$OUT"
elif [[ "$VARIANT" == F ]]; then
    "$PY" -u -m scripts.train_precision_striker "${COMMON[@]}" "${TASK[@]}" \
        --steps "$STEPS" --obstacles 1 --resume "${PREV_SWEEP:?set PREV_SWEEP}/C-seed$SEED" \
        --resume-new-task --output "$OUT"
else
    "$PY" -u -m scripts.train_precision_striker "${COMMON[@]}" "${TASK[@]}" \
        --steps "$STEPS" --obstacles 1 --output "$OUT"
fi

"$PY" -u -m scripts.eval_precision_striker --run "$OUT" --output "$OUT/evaluation" \
    --seed 30000 --layouts 10 --shots 20 --obstacles 1
echo "Done: $OUT"
