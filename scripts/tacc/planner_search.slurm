#!/bin/bash
# Tune / validate / test the hand-coded strike planner (see scripts/tacc/planner_search.py).
# Submit from the repository root:
#   sbatch -A OTH24028 --export=ALL,SEARCH_DIR=$WORK/airhockey-runs/planner-search scripts/tacc/planner_search.slurm
#SBATCH --job-name=striker-planner
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=01:30:00
#SBATCH --output=logs/striker-planner-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
PYTHON="${TACC_PYTHON:-$WORK/venvs/precision-striker-cpu/bin/python}"
"$PYTHON" -m unittest discover -s tests
"$PYTHON" -m scripts.tacc.planner_search "${SEARCH_DIR:?Set SEARCH_DIR}" --workers 124
