#!/bin/bash
# Submit from the repository root after copying the latest code to Lonestar6:
#   mkdir -p logs
#   sbatch -A YOUR_PROJECT scripts/tacc/clean_scoring.slurm
# Optional exported settings: TACC_PYTHON, BASE_RUN, RUN_DIR, TRAIN_STEPS, TRAIN_SEED, TRAIN_ENVS.
# This CPU job fine-tunes the bundled policy under the corrected scoring rules.
#SBATCH --job-name=striker-clean
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=02:00:00
#SBATCH --output=logs/striker-clean-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
: "${WORK:?TACC WORK storage must be set}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
unset DISPLAY

PYTHON="${TACC_PYTHON:-$WORK/venvs/precision-striker-cpu/bin/python}"
BASE="${BASE_RUN:-models/tacc-2026-10/AB-seed3-bank}"
RUN="${RUN_DIR:-$WORK/airhockey-runs/clean-scoring-$SLURM_JOB_ID}"
if [[ ! -x "$PYTHON" ]]; then
    echo "Missing Python environment: $PYTHON. Export TACC_PYTHON to your existing training Python." >&2
    exit 2
fi
if [[ ! -f "$BASE/policy.zip" || ! -f "$BASE/vecnormalize.pkl" || ! -f "$BASE/manifest.json" ]]; then
    echo "Missing checkpoint files in $BASE. Export BASE_RUN to the complete run directory." >&2
    exit 2
fi
"$PYTHON" -c 'import torch, tensorboard; print("PyTorch:", torch.__version__)'
"$PYTHON" -m unittest discover -s tests -q
"$PYTHON" -u -m scripts.train_precision_striker \
    --resume "$BASE" --resume-new-task \
    --steps "${TRAIN_STEPS:-2000000}" --seed "${TRAIN_SEED:-7}" \
    --envs "${TRAIN_ENVS:-32}" --vector subproc --device cpu \
    --require-hit --timeout-penalty 10 --extra-obs --goal-shaping \
    --wide-obstacles --obstacles 1 --obstacles-max 3 --single-hit \
    --paddle-restitution 0.8 --aim-shaping 5 --prefer-direct \
    --resume-ent-coef 0.005 --checkpoint-every 250000 --output "$RUN"
for seed in 50000 60000; do
    "$PYTHON" -u -m scripts.eval_suite --run "$RUN" --seed "$seed" \
        --only random_1 random_2 random_3 blocked_1 blocked_1to3 \
        --shots 200 --workers 8 --output "$RUN/validation-$seed.json"
done
"$PYTHON" -m pip freeze > "$RUN/environment.freeze"
echo "Finished. Checkpoint and validation results: $RUN"
