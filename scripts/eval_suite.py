"""Evaluation suite for the "one contact" task: 4 conditions x 200 shots, same shots for every policy.

Rules for every condition: a goal counts only after a paddle hit, one paddle contact per shot (a second
touch or the puck coming back ends the shot), paddle-puck restitution 0.8, obstacles anywhere a user may
place them, puck bounces off obstacles.

    python -m scripts.eval_suite --controller                 # strike controller (tuned settings)
    python -m scripts.eval_suite --run runs/some-model        # trained PPO policy
    python -m scripts.eval_suite --run ... --seed 50000       # validation shots (choose models on these)
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from scripts.precision_striker_env import PrecisionStrikerEnv
from scripts.strike_planner import TUNED, StrikePlanner

RULES = dict(require_hit=True, timeout_penalty=10.0, obstacle_bounce=True, single_hit=True,
             paddle_restitution=0.8, wide_obstacles=True)
CONDITIONS = {
    "random_1": dict(obstacles=1),
    "blocked_1": dict(obstacles=1, block_line_prob=1.0),
    "random_1to3": dict(obstacles=1, obstacles_max=3),
    "blocked_1to3": dict(obstacles=1, obstacles_max=3, block_line_prob=1.0),
}


def _policy(run, controller, overrides=None):
    overrides = {k: v for k, v in (overrides or {}).items() if k != "layout_check"}
    if controller:
        planner = StrikePlanner(**{**TUNED, "restitution": RULES["paddle_restitution"], **overrides})
        return planner.act, planner.reset, {}
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    torch.set_num_threads(1)
    manifest = json.loads((Path(run) / "manifest.json").read_text())
    obs_opts = {k: True for k in ("extra_obs", "goal_shaping") if manifest.get("task_options", {}).get(k)}
    probe = PrecisionStrikerEnv(obstacles=1, **{**RULES, **obs_opts})
    norm = VecNormalize.load(str(Path(run) / "vecnormalize.pkl"), DummyVecEnv([lambda: probe]))
    norm.training = False
    model = PPO.load(str(Path(run) / "policy.zip"), device="cpu")
    return (lambda o: model.predict(norm.normalize_obs(o[None, :]), deterministic=True)[0][0]), (lambda: None), obs_opts


def evaluate(job):
    name, run, controller, seed, shots, overrides = job
    act, reset, obs_opts = _policy(run, controller, overrides)
    layout = {"layout_check": overrides["layout_check"]} if overrides and "layout_check" in overrides else {}
    env = PrecisionStrikerEnv(seed=seed, randomize=False, **{**RULES, **obs_opts, **CONDITIONS[name], **layout})
    outcomes, goals = Counter(), 0
    for i in range(shots):
        obs, _ = env.reset(seed=seed + 1000 + i)
        reset()
        done = False
        while not done:
            obs, _, terminated, truncated, info = env.step(act(obs))
            done = terminated or truncated
        good = info["outcome"] == "goal" and info["hit"]
        goals += good
        outcomes[info["outcome"]] += 1
    env.close()
    return name, {"goals": goals, "shots": shots, "accuracy": goals / shots, "outcomes": dict(outcomes)}


def run_suite(run=None, controller=False, seed=30000, shots=200, workers=4, overrides=None, conditions=None):
    jobs = [(name, run, controller, seed, shots, overrides) for name in (conditions or CONDITIONS)]
    with ProcessPoolExecutor(min(workers, len(jobs))) as pool:
        results = dict(pool.map(evaluate, jobs))
    results["mean_accuracy"] = float(np.mean([results[n]["accuracy"] for n in (conditions or CONDITIONS)]))
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path)
    p.add_argument("--controller", action="store_true")
    p.add_argument("--seed", type=int, default=30000, help="30000 = test shots, 50000 = validation shots")
    p.add_argument("--shots", type=int, default=200)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--output", type=Path)
    p.add_argument("--set", nargs="*", default=[], help="Controller setting overrides, e.g. mass_factor=1.0")
    p.add_argument("--only", nargs="*", help="Run only these conditions")
    p.add_argument("--layout-check", choices=["geometry", "trajectory"], default="geometry",
                   help="trajectory: layouts verified with the trajectory calculator (different shots)")
    args = p.parse_args()
    overrides = {k: float(v) for k, v in (s.split("=") for s in args.set)}
    if args.layout_check != "geometry":
        overrides["layout_check"] = args.layout_check
    if not args.controller and args.run is None:
        p.error("give --run or --controller")
    res = run_suite(args.run, args.controller, args.seed, args.shots, args.workers, overrides, args.only)
    for name in (args.only or CONDITIONS):
        r = res[name]
        print(f"{name:13s} {r['goals']:3d}/{r['shots']} = {100 * r['accuracy']:5.1f}%   {r['outcomes']}")
    print(f"mean accuracy {100 * res['mean_accuracy']:.1f}%")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"run": str(args.run), "controller": args.controller, "overrides": overrides,
                                           "seed": args.seed, **res}, indent=2))


if __name__ == "__main__":
    main()
