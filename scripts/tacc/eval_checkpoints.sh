#!/bin/bash
# Evaluate every sweep run's checkpoint at STEP on the historical protocol (seed 30000, 10 x 20),
# while training continues. Run inside the sweep allocation:
#   srun --jobid=<sweep job> --overlap -N1 -n1 -c 24 bash scripts/tacc/eval_checkpoints.sh SWEEP_DIR STEP
set -uo pipefail
SWEEP="$1"; STEP="$2"
PY="${TACC_PYTHON:-$WORK/venvs/precision-striker-cpu/bin/python}"
OUT="$SWEEP/checkpoint-eval-$STEP"
mkdir -p "$OUT"
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1
for run in "$SWEEP"/[ABC]-seed[0-9]; do
    name=$(basename "$run")
    src="$run"
    # Variant C before its switch: use the finished easy-task stage.
    if [[ ! -e "$run/checkpoints/policy_${STEP}_steps.zip" && -e "$run-stage1/policy.zip" ]]; then
        src="$run-stage1"
    fi
    dest="$OUT/$name"
    mkdir -p "$dest"
    cp "$src/manifest.json" "$dest/"
    if [[ -e "$src/checkpoints/policy_${STEP}_steps.zip" ]]; then
        cp "$src/checkpoints/policy_${STEP}_steps.zip" "$dest/policy.zip"
        cp "$src/checkpoints/policy_vecnormalize_${STEP}_steps.pkl" "$dest/vecnormalize.pkl"
    elif [[ "$src" == *-stage1 ]]; then
        cp "$src/policy.zip" "$src/vecnormalize.pkl" "$dest/"
        echo "stage1-final" > "$dest/NOTE"
    else
        echo "$name: no checkpoint at $STEP yet" | tee "$dest/MISSING"; continue
    fi
    "$PY" -m scripts.eval_precision_striker --run "$dest" --output "$dest/evaluation" \
        --seed 30000 --layouts 10 --shots 20 --obstacles 1 > "$dest/eval.log" 2>&1 &
done
wait
"$PY" - "$OUT" <<'EOF'
import json, sys
from pathlib import Path
root = Path(sys.argv[1]); rows = []
for ev in sorted(root.glob("*/evaluation/evaluation.json")):
    s = json.loads(ev.read_text()); run = ev.parents[1]
    rows.append({"run": run.name, "note": (run / "NOTE").read_text().strip() if (run / "NOTE").exists() else "",
                 "raw_goals": s["raw_goals"], "goals_after_hit": s["goals_after_hit"],
                 "collision_episodes": s["collision_episodes"], "episodes": s["episodes"], "outcomes": s["outcomes"]})
(root / "summary.json").write_text(json.dumps(rows, indent=2))
for r in rows:
    print(f'{r["run"]:8s} raw {r["raw_goals"]:3d}/{r["episodes"]} after-hit {r["goals_after_hit"]:3d} '
          f'collisions {r["collision_episodes"]:3d} {r["note"]} {r["outcomes"]}')
EOF
