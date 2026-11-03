#!/bin/bash
# Charts, GIFs and summary for an imitation_sweep.slurm result (scripts/analysis/imitation_report.py).
# Queue it to start when training ends:
#   sbatch -A OTH24028 -p normal --dependency=afterany:<training job> --export=ALL,SWEEP_DIR=... scripts/tacc/imitation_report.slurm
#SBATCH --job-name=striker-report
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=03:00:00
#SBATCH --output=logs/striker-report-%j.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
"${TACC_PYTHON:-$WORK/venvs/precision-striker-cpu/bin/python}" -u -m scripts.analysis.imitation_report \
    --sweep "${SWEEP_DIR:?set SWEEP_DIR}" --parallel 28
echo "Report: $SWEEP_DIR/report"
