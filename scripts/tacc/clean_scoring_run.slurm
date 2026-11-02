#!/bin/bash
# One clean-scoring training run on Lonestar6, for the strategy in docs/CLEAN_SCORING.md.
# Generalizes clean_scoring.slurm: either mode, any seed, any step budget, so a sweep is
# several submissions of this one file rather than several near-identical copies.
#
#   MODE=finetune TRAIN_SEED=7 TRAIN_STEPS=10000000 \
#     sbatch -A TRA25006 -J clean-ft-s7 -t 24:00:00 scripts/tacc/clean_scoring_run.slurm
#
# MODE=finetune resumes the bundled 34M-step bank model under the corrected rules.
# MODE=scratch trains a fresh policy, pinned to the architecture and PPO settings that
# model actually carries (256-wide, gamma 0.995, rollout 512) rather than the script
# defaults, which differ and would not be a like-for-like comparison.
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
: "${WORK:?TACC WORK storage must be set}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
unset DISPLAY

MODE="${MODE:-finetune}"
PYTHON="${TACC_PYTHON:-$WORK/venvs/precision-striker-cpu/bin/python}"
BASE="${BASE_RUN:-models/tacc-2026-10/AB-seed3-bank}"
SEED="${TRAIN_SEED:-7}"
STEPS="${TRAIN_STEPS:-10000000}"
RUN="${RUN_DIR:-$WORK/airhockey-runs/clean-$MODE-s$SEED-$SLURM_JOB_ID}"
if [[ ! -x "$PYTHON" ]]; then
    echo "Missing Python environment: $PYTHON. Build it with scripts/tacc/setup_env_cpu.sh." >&2
    exit 2
fi

# The corrected objective: obstacle contact ends the episode (no --obstacle-bounce), and
# --prefer-direct withholds aim credit from bank shots whenever a direct shot is open.
# --block-line-prob 0.5 keeps the base model's layout mix, so half of training is open-goal
# and half forces a bank, matching the random_* and blocked_* validation conditions.
COMMON=(--steps "$STEPS" --seed "$SEED" --envs "${TRAIN_ENVS:-32}"
        --vector subproc --device cpu
        --require-hit --timeout-penalty 10 --extra-obs --goal-shaping
        --wide-obstacles --obstacles 1 --obstacles-max 3 --single-hit
        --paddle-restitution 0.8 --aim-shaping 5 --prefer-direct
        --block-line-prob 0.5 --checkpoint-every 250000 --output "$RUN")

case "$MODE" in
    finetune)
        if [[ ! -f "$BASE/policy.zip" || ! -f "$BASE/vecnormalize.pkl" || ! -f "$BASE/manifest.json" ]]; then
            echo "Missing checkpoint files in $BASE. Export BASE_RUN to the complete run directory." >&2
            exit 2
        fi
        ARGS=(--resume "$BASE" --resume-new-task --resume-ent-coef "${ENT_COEF:-0.005}")
        ;;
    scratch)
        # --net, --rollout and --batch-size only apply to a new model; a resumed one keeps
        # whatever its ancestors were built with. These match the base model's real shape.
        ARGS=(--net 256 --rollout 512 --batch-size 1024 --gamma 0.995 --ent-coef "${ENT_COEF:-0.01}")
        ;;
    *)
        echo "MODE must be finetune or scratch, not '$MODE'." >&2
        exit 2
        ;;
esac

"$PYTHON" -c 'import torch, tensorboard; print("PyTorch:", torch.__version__)'
"$PYTHON" -m unittest discover -s tests -q
"$PYTHON" -u -m scripts.train_precision_striker "${ARGS[@]}" "${COMMON[@]}"

# Seeds 50000 and 60000 are the validation pair; 30000 stays the untouched historical test.
for seed in 50000 60000; do
    "$PYTHON" -u -m scripts.eval_suite --run "$RUN" --seed "$seed" \
        --only random_1 random_2 random_3 blocked_1 blocked_1to3 \
        --shots 200 --workers 8 --output "$RUN/validation-$seed.json"
done
"$PYTHON" -m pip freeze > "$RUN/environment.freeze"
echo "Finished. Checkpoint and validation results: $RUN"
