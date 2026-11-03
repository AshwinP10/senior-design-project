#!/bin/bash
# Push a sweep's report to GitHub main once its report job ends (see publish_results.sh).
#   sbatch -A OTH24028 -p normal --dependency=afterany:<report job> --export=ALL,SWEEP_DIR=...,NAME=... scripts/tacc/publish_results.slurm
#SBATCH --job-name=striker-publish
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=logs/striker-publish-%j.out
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
bash scripts/tacc/publish_results.sh "${SWEEP_DIR:?}" "${NAME:?}"
