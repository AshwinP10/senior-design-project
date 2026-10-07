#!/bin/bash
#SBATCH --job-name=airhockey-smoke
#SBATCH --partition=rtx-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/airhockey-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
: "${WORK:?TACC WORK storage must be set}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
unset DISPLAY

# Select an existing environment. Set TACC_PYTHON at submission if needed.
PYTHON="${TACC_PYTHON:-$WORK/venvs/precision-striker/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
    echo "Missing $PYTHON. Run scripts/tacc/setup_env.sh on a compute node first." >&2
    exit 2
fi
nvidia-smi
"$PYTHON" -c 'import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(), "CUDA is unavailable"; print(torch.cuda.get_device_name(0))'
RUN="${RUN_DIR:-$WORK/airhockey-runs/$SLURM_JOB_ID}"
"$PYTHON" -m unittest discover -s tests -v
"$PYTHON" -u -m scripts.train_precision_striker \
    --device cuda --envs 8 --vector subproc --rollout 512 \
    --steps "${TRAIN_STEPS:-32768}" --obstacles "${OBSTACLES:-1}" \
    --speed-min "${SPEED_MIN:-0.3}" --speed-max "${SPEED_MAX:-1.5}" \
    --seed "${TRAIN_SEED:-0}" --output "$RUN"
"$PYTHON" -u -m scripts.eval_precision_striker \
    --run "$RUN" --output "$RUN/evaluation" --obstacles "${OBSTACLES:-1}" \
    --speed-min "${SPEED_MIN:-0.3}" --speed-max "${SPEED_MAX:-1.5}" \
    --layouts "${EVAL_LAYOUTS:-2}" --shots "${EVAL_SHOTS:-5}"
"$PYTHON" -m pip freeze > "$RUN/environment.freeze"
echo "Finished. Results: $RUN"
echo "Terminal replay: $PYTHON -m scripts.eval_precision_striker --run $RUN --output $RUN/replay --layouts 1 --shots 1 --obstacles ${OBSTACLES:-1} --terminal"
