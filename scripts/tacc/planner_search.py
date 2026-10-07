"""Tune the strike planner on validation shots, confirm on a second validation set, then test once.

Stage 1: every setting combination on validation seed 50000 (200 shots).
Stage 2: the top 10 on an independent validation seed 60000 (200 shots).
Stage 3: the best by combined validation accuracy on the reported test (seed 30000), plus stress
         tests (2 and 3 obstacles, faster unseen pucks), all under the bounce rule.
Run on a compute node:  python scripts/tacc/planner_search.py OUTPUT_DIR --workers 120
"""
from __future__ import annotations

import argparse
import itertools
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from scripts.eval_planner import run

GRID = {"kp_lat": [4.0, 8.0, 12.0, 16.0, 24.0], "min_slack": [0.0, 0.1, 0.2],
        "out_speed": [2.0, 2.4, 2.8], "kp_track": [4.0, 8.0], "tau": [0.05, 0.08, 0.12]}


def evaluate(job):
    cfg, kwargs = job
    summary, _ = run(planner_kwargs=cfg, **kwargs)
    return {"planner": cfg, **{k: summary[k] for k in ("goals_after_hit", "episodes", "outcomes", "accuracy")}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--workers", type=int, default=120)
    p.add_argument("--quick", action="store_true", help="Pipeline test: 2 settings, 4 shots per evaluation")
    args = p.parse_args()
    small = {"layouts": 1, "shots": 4} if args.quick else {}
    args.output.mkdir(parents=True, exist_ok=True)
    configs = [dict(zip(GRID, values)) for values in itertools.product(*GRID.values())]
    if args.quick:
        configs = configs[:2]
    with ProcessPoolExecutor(args.workers) as pool:
        stage1 = list(pool.map(evaluate, [(c, {"seed": 50000, **small}) for c in configs]))
        stage1.sort(key=lambda r: -r["accuracy"])
        (args.output / "stage1_val50000.json").write_text(json.dumps(stage1, indent=2))
        top = [r["planner"] for r in stage1[:10]]
        stage2 = list(pool.map(evaluate, [(c, {"seed": 60000, **small}) for c in top]))
        for r2 in stage2:
            r2["val1_accuracy"] = next(r["accuracy"] for r in stage1 if r["planner"] == r2["planner"])
            r2["combined_val"] = (r2["accuracy"] + r2["val1_accuracy"]) / 2
        stage2.sort(key=lambda r: -r["combined_val"])
        (args.output / "stage2_val60000.json").write_text(json.dumps(stage2, indent=2))
        best = stage2[0]["planner"]
        tests = {"test_seed30000": {"seed": 30000},
                 "two_obstacles": {"seed": 30000, "obstacles": 2},
                 "three_obstacles": {"seed": 30000, "obstacles": 3},
                 "fast_unseen_1.5_2.0": {"seed": 30000, "speed_min": 1.5, "speed_max": 2.0},
                 "default_planner_test": {"seed": 30000}}
        jobs = [((best if name != "default_planner_test" else {}), {**kw, **small}) for name, kw in tests.items()]
        stage3 = dict(zip(tests, pool.map(evaluate, jobs)))
    (args.output / "stage3_test.json").write_text(json.dumps({"best_planner": best, **stage3}, indent=2))
    for name, r in stage3.items():
        print(f"{name:24s} {r['goals_after_hit']}/{r['episodes']} = {r['accuracy']:.1%}  {r['outcomes']}")
    print("best settings:", best)


if __name__ == "__main__":
    main()
