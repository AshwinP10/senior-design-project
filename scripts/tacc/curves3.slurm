#!/bin/bash
# Learning curves for sweep 3 (all runs use the bounce rule, so one rule set):
#   shot-choice runs (J, K) every 25k shots, residual / imitation runs (L, M) every 0.5M steps.
# Submit from the repository root:
#   sbatch -A OTH24028 -p development --export=ALL,SWEEP_DIR=... scripts/tacc/curves3.slurm
#SBATCH --job-name=striker-curves3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=01:30:00
#SBATCH --output=logs/striker-curves3-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
: "${SWEEP_DIR:?Set SWEEP_DIR}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
PYTHON="${TACC_PYTHON:-$WORK/venvs/precision-striker-cpu/bin/python}"
"$PYTHON" scripts/tacc/curve_eval.py "$SWEEP_DIR" --workers 124 --variants JK --every 25000 --tag shots
"$PYTHON" scripts/tacc/curve_eval.py "$SWEEP_DIR" --workers 124 --variants LM --every 500000 --tag steps
