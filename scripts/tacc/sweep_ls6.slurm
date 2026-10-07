#!/bin/bash
# Lonestar6 CPU sweep: 3 variants x 3 training seeds, run side by side on one 128-core node.
#   A  = v2 task, longer training (does more training alone help?)
#   B  = v3 task: goal requires a paddle hit, timeout cost 10
#   C  = B + curriculum (slow puck, no obstacles first, then the full task)
# Each run ends with the historical evaluation protocol (seed 30000, 10 layouts x 20 shots),
# so results are directly comparable with the saved 18/200 raw and 9/200 after-hit record.
#
#   (set VARIANTS="D E F" for the obstacle-bounce sweep; see scripts/tacc/run_variant.sh)
# Submit from the repository root:  sbatch -A OTH24028 scripts/tacc/sweep_ls6.slurm
#SBATCH --job-name=striker-sweep
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=12:00:00
#SBATCH --output=logs/striker-sweep-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
: "${WORK:?TACC WORK storage must be set}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
unset DISPLAY

PYTHON="${TACC_PYTHON:-$WORK/venvs/precision-striker-cpu/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
    echo "Missing $PYTHON. Create the CPU environment first (see scripts/tacc/run_variant.sh header)." >&2
    exit 2
fi
"$PYTHON" -m unittest discover -s tests -v

# Optional: take the strike-controller settings chosen by planner_search (stage3_test.json).
if [[ -n "${PLANNER_FROM:-}" ]]; then
    export PLANNER_JSON="$("$PYTHON" -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1]))["best_planner"]))' "$PLANNER_FROM")"
    echo "Planner settings: $PLANNER_JSON"
fi
STEPS="${TRAIN_STEPS:-5000000}"
WARMUP="${CURRICULUM_STEPS:-1000000}"
SWEEP="${SWEEP_DIR:-$WORK/airhockey-runs/sweep-$SLURM_JOB_ID}"
mkdir -p "$SWEEP"
cp "$0" "$SWEEP/sweep.slurm" 2>/dev/null || true

for seed in 1 2 3; do
    for variant in ${VARIANTS:-A B C}; do
        bash scripts/tacc/run_variant.sh "$PYTHON" "$variant" "$seed" "$STEPS" "$WARMUP" \
            "$SWEEP/$variant-seed$seed" > "$SWEEP/$variant-seed$seed.log" 2>&1 &
    done
done
wait || true   # one failed run must not discard the others

"$PYTHON" - "$SWEEP" <<'EOF'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
rows = []
for ev in sorted(root.glob("*/evaluation/evaluation.json")):
    s = json.loads(ev.read_text())
    rows.append({"run": ev.parents[1].name, "task_version": s["task_version"],
                 "episodes": s["episodes"], "raw_goals": s["raw_goals"],
                 "goals_after_hit": s["goals_after_hit"], "success_rate": s["scoring_rate"],
                 "goals_off_obstacle": s.get("goals_off_obstacle"),
                 "collision_episodes": s["collision_episodes"], "outcomes": s["outcomes"]})
(root / "sweep_summary.json").write_text(json.dumps(rows, indent=2))
for r in rows:
    print(f'{r["run"]:10s} raw {r["raw_goals"]:3d}/{r["episodes"]}  after-hit {r["goals_after_hit"]:3d}  '
          f'collisions {r["collision_episodes"]:3d}  {r["outcomes"]}')
EOF
"$PYTHON" -m pip freeze > "$SWEEP/environment.freeze"
echo "Finished. Results: $SWEEP"
