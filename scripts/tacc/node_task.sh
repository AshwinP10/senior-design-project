#!/bin/bash
# One node's share of sweep3_all.slurm. Usage: node_task.sh GROUP OUT_DIR
#   J  shot-choice PPO x3        KL  shot-choice SAC x3 + residual PPO x3
#   M  imitation -> PPO x3       P   planner search
set -uo pipefail
GROUP="$1"; OUT="$2"
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
PY="${TACC_PYTHON:-$WORK/venvs/precision-striker-cpu/bin/python}"
echo "$(hostname) group $GROUP start $(date)"
launch() {  # variant seed steps
    bash scripts/tacc/run_variant.sh "$PY" "$1" "$2" "$3" 0 "$OUT/$1-seed$2" > "$OUT/$1-seed$2.log" 2>&1 &
}
case "$GROUP" in
    J)  for s in 1 2 3; do launch J "$s" "${J_STEPS:-300000}"; done ;;
    KL) for s in 1 2 3; do launch K "$s" "${K_STEPS:-150000}"; launch L "$s" "${L_STEPS:-5000000}"; done ;;
    M)  export BC_STEPS="${BC_STEPS:-600000}"
        for s in 1 2 3; do launch M "$s" "${M_STEPS:-4000000}"; done ;;
    P)  "$PY" -m scripts.tacc.planner_search "$OUT/planner-search" --workers 124 > "$OUT/planner-search.log" 2>&1 ;;
esac
wait
echo "$(hostname) group $GROUP done $(date)"
