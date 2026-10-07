"""Evaluate the hand-coded strike planner on the historical 200-shot protocol (seed 30000)."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np

from scripts.precision_striker_env import PrecisionStrikerEnv
from scripts.strike_planner import StrikePlanner


def run(layouts=10, shots=20, seed=30000, bounce=True, obstacles=1, planner_kwargs=None, verbose=False,
        speed_min=0.3, speed_max=1.5):
    env = PrecisionStrikerEnv(seed=seed, obstacles=obstacles, randomize=False, require_hit=True,
                              timeout_penalty=10, obstacle_bounce=bounce, speed_min=speed_min, speed_max=speed_max)
    planner = StrikePlanner(**(planner_kwargs or {}))
    records = []
    try:
        for layout in range(layouts):
            blocks = env.sample_layout(seed + layout)
            for shot in range(shots):
                ep_seed = seed + 1000 + layout * shots + shot
                obs, info = env.reset(seed=ep_seed, options={"blocks": blocks})
                planner.reset()
                done, steps = False, 0
                while not done:
                    obs, _, terminated, truncated, info = env.step(planner.act(obs))
                    steps += 1
                    done = terminated or truncated
                records.append({"layout": layout, "shot": shot, "seed": ep_seed, "outcome": info["outcome"],
                                "hit": info["hit"], "success": info["is_success"], "steps": steps,
                                "obstacle_contact": info["obstacle_collision"]})
                if verbose:
                    print(json.dumps(records[-1]), flush=True)
    finally:
        env.close()
    n = len(records)
    summary = {"episodes": n, "goals_after_hit": sum(r["outcome"] == "goal" and r["hit"] for r in records),
               "outcomes": dict(Counter(r["outcome"] for r in records)),
               "hits": sum(r["hit"] for r in records), "rules": "bounce" if bounce else "collision-ends"}
    summary["accuracy"] = summary["goals_after_hit"] / n
    return summary, records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layouts", type=int, default=10)
    p.add_argument("--shots", type=int, default=20)
    p.add_argument("--seed", type=int, default=30000)
    p.add_argument("--obstacles", type=int, default=1)
    p.add_argument("--no-bounce", action="store_true", help="Collisions end the shot (v3 rules)")
    p.add_argument("--set", nargs="*", default=[], help="Planner overrides, e.g. out_speed=2.5 tau=0.2")
    p.add_argument("--tuned", action="store_true", help="Start from strike_planner.TUNED settings")
    p.add_argument("--output", type=Path)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    kwargs = {k: float(v) if "." in v or v.isdigit() else v for k, v in (s.split("=") for s in args.set)}
    if args.tuned:
        from scripts.strike_planner import TUNED
        kwargs = {**TUNED, **kwargs}
    if "x_window" in kwargs:
        kwargs["x_window"] = tuple(float(x) for x in str(kwargs["x_window"]).split(","))
    started = time.perf_counter()
    summary, records = run(args.layouts, args.shots, args.seed, not args.no_bounce, args.obstacles,
                           kwargs, args.verbose)
    summary.update(planner=kwargs, seconds=round(time.perf_counter() - started, 1))
    print(json.dumps(summary, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({**summary, "records": records}, indent=2))


if __name__ == "__main__":
    main()
