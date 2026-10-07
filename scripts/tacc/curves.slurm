#!/bin/bash
# Learning curves for a finished sweep (see scripts/tacc/curve_eval.py).
# Submit from the repository root:
#   sbatch -A OTH24028 --export=ALL,SWEEP_DIR=$WORK/airhockey-runs/sweep-<job> scripts/tacc/curves.slurm
#SBATCH --job-name=striker-curves
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=01:00:00
#SBATCH --output=logs/striker-curves-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
: "${SWEEP_DIR:?Set SWEEP_DIR to the sweep to analyse}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
PYTHON="${TACC_PYTHON:-$WORK/venvs/precision-striker-cpu/bin/python}"
"$PYTHON" scripts/tacc/curve_eval.py "$SWEEP_DIR" --workers 120 --bounce --every "${CURVE_EVERY:-500000}"
