"""Learning curves for a finished sweep: evaluate saved checkpoints on the historical 200-shot test.

For each run and each target step, the nearest checkpoint at or below that step is evaluated
(curriculum runs use their easy-stage checkpoints before the switch). Each checkpoint is scored
under the rules it was trained with and, with --bounce, also under v4 obstacle-bounce rules.
Run on a compute node:  python scripts/tacc/curve_eval.py SWEEP_DIR --workers 120 --bounce
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

STEP_RE = re.compile(r"policy_(\d+)_steps\.zip$")


def checkpoints(run: Path):
    """(steps, zip, pkl, manifest) for a run, including its easy stage if it has one."""
    found = []
    for directory in (Path(f"{run}-stage1"), run):
        for z in (directory / "checkpoints").glob("policy_*_steps.zip"):
            steps = int(STEP_RE.search(z.name).group(1))
            pkl = z.with_name(f"policy_vecnormalize_{steps}_steps.pkl")
            if pkl.exists():
                found.append((steps, z, pkl, directory / "manifest.json"))
    manifest = json.loads((run / "manifest.json").read_text())
    found.append((manifest["completed_timesteps"], run / "policy.zip", run / "vecnormalize.pkl",
                  run / "manifest.json"))
    return sorted(found, key=lambda item: item[0])


def evaluate(job):
    work, zip_path, pkl_path, manifest, bounce, seed = job
    work.mkdir(parents=True, exist_ok=True)
    shutil.copy(zip_path, work / "policy.zip")
    shutil.copy(pkl_path, work / "vecnormalize.pkl")
    shutil.copy(manifest, work / "manifest.json")
    out = work / ("eval-bounce" if bounce else "eval")
    cmd = [sys.executable, "-m", "scripts.eval_precision_striker", "--run", str(work), "--output", str(out),
           "--seed", str(seed), "--layouts", "10", "--shots", "20", "--obstacles", "1"]
    if bounce:
        cmd.append("--obstacle-bounce")
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    s = json.loads((out / "evaluation.json").read_text())
    return {k: s[k] for k in ("raw_goals", "goals_after_hit", "collision_episodes", "outcomes", "episodes")}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("sweep", type=Path)
    p.add_argument("--every", type=int, default=500_000)
    p.add_argument("--workers", type=int, default=100)
    p.add_argument("--bounce", action="store_true")
    p.add_argument("--variants", default="", help="Only these variant letters, e.g. JK (default: all)")
    p.add_argument("--tag", default="", help="Suffix for the output file and folder")
    p.add_argument("--seed", type=int, default=30000,
                   help="Test-shot set: 30000 is the reported test; use another (e.g. 50000) to choose checkpoints")
    args = p.parse_args()
    tag = ("" if args.seed == 30000 else f"-seed{args.seed}") + (f"-{args.tag}" if args.tag else "")
    out_root = args.sweep / f"curves{tag}"
    out_root.mkdir(exist_ok=True)
    jobs, meta = [], []
    for run in sorted(d for d in args.sweep.iterdir() if re.fullmatch(r"[A-Z]-seed\d+", d.name)):
        if args.variants and run.name[0] not in args.variants:
            continue
        cps = checkpoints(run)
        final = cps[-1][0]
        targets = list(range(args.every, final, args.every)) + [final]
        for target in targets:
            eligible = [c for c in cps if c[0] <= target]
            if not eligible:
                continue
            steps, z, pkl, manifest = eligible[-1]
            for bounce in ([False, True] if args.bounce else [False]):
                work = out_root / run.name / str(steps) / ("bounce" if bounce else "trained")
                jobs.append((work, z, pkl, manifest, bounce, args.seed))
                meta.append({"run": run.name, "variant": run.name[0], "seed": int(run.name.split("seed")[1]),
                             "steps": steps, "rules": "bounce" if bounce else "trained", "eval_seed": args.seed,
                             "easy_stage": "-stage1" in str(z)})
    print(f"{len(jobs)} evaluations", flush=True)
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(evaluate, jobs))
    rows = [{**m, **r} for m, r in zip(meta, results)]
    (args.sweep / f"curves{tag}.json").write_text(json.dumps(rows, indent=2))
    print(f"Wrote {args.sweep / f'curves{tag}.json'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
