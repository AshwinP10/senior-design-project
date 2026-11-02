#!/bin/bash
# CPU training environment for the Lonestar6 `normal` partition, built with uv.
# Companion to setup_env.sh, which builds the CUDA environment for GPU nodes.
# Everything lands on $WORK: /scratch is purged after 10 days and takes venvs with it.
set -euo pipefail
: "${WORK:?TACC WORK storage must be set}"
cd "$(dirname "$0")/../.."
UV="${UV_BIN:-$WORK/tools/uv/uv}"
VENV="${TACC_VENV:-$WORK/venvs/precision-striker-cpu}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$WORK/uv-python}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$WORK/uv-cache}"
# The torch wheel is ~180 MB and the login nodes reach PyPI slowly; the 30s default times out.
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"
# Unpacking that wheel in parallel exhausts the per-user memory cgroup and aborts uv,
# so extract one wheel at a time. Run this inside a job, not on a login node.
export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-1}"
export UV_CONCURRENT_INSTALLS="${UV_CONCURRENT_INSTALLS:-1}"
export UV_CONCURRENT_BUILDS="${UV_CONCURRENT_BUILDS:-1}"
if [[ ! -x "$UV" ]]; then
    echo "Missing uv at $UV. Export UV_BIN, or install uv from https://astral.sh/uv." >&2
    exit 2
fi
if [[ -e "$VENV" ]]; then
    echo "Environment already exists at $VENV; inspect it or choose a new TACC_VENV." >&2
    exit 2
fi
# Pin to the interpreter and wheel versions recorded in the bundled checkpoint manifests,
# so a resumed policy loads against the libraries it was written by.
"$UV" venv --python 3.11 "$VENV"
"$UV" pip install --python "$VENV/bin/python" 'torch==2.2.2' --index-url https://download.pytorch.org/whl/cpu
"$UV" pip install --python "$VENV/bin/python" -r requirements-headless.in
"$VENV/bin/python" -c 'import torch, stable_baselines3 as sb3; print(torch.__version__, sb3.__version__)'
"$VENV/bin/python" -m unittest discover -s tests -q
echo "Ready: $VENV/bin/python"
