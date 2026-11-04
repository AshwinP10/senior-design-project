#!/bin/bash
# Publish an imitation sweep's report to GitHub main: charts, GIFs, summaries, small result files and the chosen
# models. Runs from TACC with the repository deploy key (~/.ssh/github_striker, host alias github-striker in
# ~/.ssh/config). Uses its own clone in $WORK/striker-publish, which is only ever used for publishing.
# Usage: publish_results.sh SWEEP_DIR NAME        (e.g. $WORK/airhockey-runs/imitation-long2 imitation-long2)
# On failure it leaves $SWEEP_DIR/PUBLISH_FAILED so a retry (publish_watch.sh on a login node) can take over.
set -uo pipefail
exec < /dev/null                      # git/ssh must not read the caller's input
SWEEP="$1"; NAME="$2"
REPO="$WORK/striker-publish"
DEST="results/tacc/$NAME"
fail() { echo "publish failed: $1"; date > "$SWEEP/PUBLISH_FAILED"; echo "$1" >> "$SWEEP/PUBLISH_FAILED"; exit 1; }
[ -d "$SWEEP/report" ] || fail "no report in $SWEEP"
if [ ! -d "$REPO/.git" ]; then
    git clone -q git@github-striker:AshwinP10/senior-design-project.git "$REPO" || fail "clone"
fi
cd "$REPO" || fail "cd"
git fetch -q origin || fail "fetch"
git checkout -q main && git reset -q --hard origin/main || fail "checkout"
mkdir -p "$DEST/report/animations" "$DEST/run_files"   # "gifs/" and "runs/" are gitignored
cp "$SWEEP"/report/*.png "$SWEEP"/report/*.md "$SWEEP"/report/*.json "$DEST/report/" 2>/dev/null
# GIFs up to 8 MB each (GitHub warns above 50 MB per file; keep the repository light).
find "$SWEEP/report/gifs" -name '*.gif' -size -8M -exec cp {} "$DEST/report/animations/" \; 2>/dev/null
cp "$SWEEP"/teacher_suite_val50000.json "$DEST/run_files/" 2>/dev/null
for d in "$SWEEP"/D*-imitation "$SWEEP"/D*-ppo-seed*; do
    [ -d "$d" ] || continue
    n=$(basename "$d"); mkdir -p "$DEST/run_files/$n"
    cp "$d"/manifest.json "$d"/suite_val50000.json "$DEST/run_files/$n/" 2>/dev/null
    for f in "$d"/curve/ckpt-*/suite_val50000.json; do
        [ -f "$f" ] || continue
        c=$(basename "$(dirname "$f")"); mkdir -p "$DEST/run_files/$n/curve/$c"; cp "$f" "$DEST/run_files/$n/curve/$c/"
    done
done
cp "$SWEEP"/*-imitation.log "$DEST/run_files/" 2>/dev/null
# The models the report chose on validation (imitation and imitation + PPO), so others can run them.
python3 - "$SWEEP/report/summary.json" "$REPO/models/tacc-2026-11/$NAME" <<'PY'
import json, shutil, sys
from pathlib import Path
summary, dest = json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2])
for label, run in summary.get("chosen", {}).items():
    if not run:
        continue
    out = dest / label.replace(" ", "_").replace("+", "plus")
    out.mkdir(parents=True, exist_ok=True)
    for f in ("policy.zip", "vecnormalize.pkl", "manifest.json"):
        if (Path(run) / f).exists():
            shutil.copy(Path(run) / f, out / f)
    (out / "SOURCE.txt").write_text(f"{run}\n")
PY
git add -f "$DEST" "models/tacc-2026-11/$NAME" 2>/dev/null   # -f: *.gif, *.log are gitignored repo-wide
if git diff --cached --quiet; then echo "nothing new to publish"; rm -f "$SWEEP/PUBLISH_FAILED"; exit 0; fi
MSG="Add TACC results: $NAME

Charts, GIFs, summary and chosen models from $SWEEP (published automatically from TACC).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git -c user.name="Precision Striker TACC" -c user.email="ap63726@users.noreply.github.com" commit -q -m "$MSG" || fail "commit"
for attempt in 1 2 3; do
    if git push -q origin main; then
        echo "published $NAME to main: $(git rev-parse --short HEAD)"; rm -f "$SWEEP/PUBLISH_FAILED"; date > "$SWEEP/PUBLISHED"; exit 0
    fi
    git pull -q --rebase origin main || fail "rebase"
done
fail "push"
