#!/bin/bash
# Run inside idev or sbatch, not on a shared login node.
set -euo pipefail
: "${SLURM_JOB_ID:?Request a compute allocation with idev first}"
: "${WORK:?TACC WORK storage must be set}"
cd "$(dirname "$0")/../.."
PYTHON="${BOOTSTRAP_PYTHON:-python3}"
"$PYTHON" -c 'import sys; assert (3,10) <= sys.version_info < (3,12), "Select Python 3.10 or 3.11 using module spider python3"'
VENV="${TACC_VENV:-$WORK/venvs/precision-striker}"
if [[ -e "$VENV" ]]; then
    echo "Environment already exists at $VENV; inspect it or choose a new TACC_VENV." >&2
    exit 2
fi
"$PYTHON" -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip
# Conservative CUDA 12.1 wheel for the older RTX nodes; driver >= 525 required.
# Override only after checking nvidia-smi and the official PyTorch wheel matrix.
"$VENV/bin/python" -m pip install 'torch==2.2.2' --index-url https://download.pytorch.org/whl/cu121
"$VENV/bin/python" -m pip install -r requirements-headless.in
"$VENV/bin/python" -c 'import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available()'
"$VENV/bin/python" -m unittest discover -s tests -v
echo "Ready: $VENV/bin/python"
