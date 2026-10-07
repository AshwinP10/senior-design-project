"""Reproducible held-out evaluation, terminal playback and optional OpenCV viewer."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from scripts.hybrid_envs import make_env
from scripts.precision_striker_env import task_version


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--layouts", type=int, default=10)
    p.add_argument("--shots", type=int, default=20)
    p.add_argument("--seed", type=int, default=10000)
    p.add_argument("--obstacles", type=int, choices=range(4), default=1)
    p.add_argument("--terminal", action="store_true")
    p.add_argument("--render", action="store_true")
    p.add_argument("--puck", type=float, nargs=2, metavar=("X", "Y"))
    p.add_argument("--velocity", type=float, nargs=2, help="Override randomized incoming velocity")
    p.add_argument("--speed-min", type=float, default=0.3)
    p.add_argument("--speed-max", type=float, default=1.5)
    p.add_argument("--blocks", type=str, help='Fixed layout, e.g. "-0.4,0;-0.6,0.2"')
    p.add_argument("--obstacle-bounce", action="store_true",
                   help="Evaluate under v4 rules (puck bounces off obstacles) regardless of training task")
    args = p.parse_args()
    if min(args.layouts, args.shots) < 1:
        p.error("layouts and shots must be positive")
    manifest = json.loads((args.run / "manifest.json").read_text())
    options = manifest.get("task_options", {"require_hit": False, "timeout_penalty": 2.0})
    if manifest["task_version"] != task_version(**options):
        p.error("Checkpoint task version does not match")
    if args.obstacle_bounce:
        options = {**options, "obstacle_bounce": True}
    TASK_VERSION = task_version(**options)
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    hybrid = manifest.get("hybrid", "none")
    extra = {"scale": manifest.get("residual_scale", 0.5)} if hybrid == "residual" else {}
    if hybrid != "none":
        extra["planner"] = json.loads(manifest.get("planner", "{}"))
    env = make_env(hybrid, seed=args.seed, obstacles=args.obstacles, randomize=False,
                   speed_min=args.speed_min, speed_max=args.speed_max, **options, **extra)
    base_env = getattr(env, "env", env)          # hybrid wrappers hold the simulator in .env
    vector = DummyVecEnv([lambda: env])
    normalizer = VecNormalize.load(str(args.run / "vecnormalize.pkl"), vector)
    normalizer.training = False
    normalizer.norm_reward = False
    model = (SAC if manifest.get("algo") == "sac" else PPO).load(str(args.run / "policy.zip"), device="cpu")
    renderer = None
    if args.render:
        import cv2
        from airhockey.renderers import AirHockeyRenderer
        renderer = AirHockeyRenderer(base_env, orientation="vertical", show_target_position=False)
    fixed_blocks = None
    if args.blocks is not None:
        fixed_blocks = [tuple(map(float, item.split(","))) for item in args.blocks.split(";") if item]
    records = []
    quit_requested = False
    try:
        for layout in range(args.layouts):
            blocks = base_env.sample_layout(args.seed + layout) if fixed_blocks is None else fixed_blocks
            for shot in range(args.shots):
                seed = args.seed + 1000 + layout * args.shots + shot
                options = {"blocks": blocks}
                if args.velocity is not None:
                    options["puck_velocity"] = args.velocity
                if args.puck is not None:
                    options["puck_position"] = args.puck
                obs, reset_info = env.reset(seed=seed, options=options)
                observations, actions, rewards, latencies = [obs.copy()], [], [], []
                done = False
                info = {}
                while not done:
                    before = time.perf_counter()
                    normalized = normalizer.normalize_obs(obs[None, :])
                    action, _ = model.predict(normalized, deterministic=True)
                    latencies.append((time.perf_counter() - before) * 1000)
                    obs, reward, terminated, truncated, info = env.step(action[0])
                    observations.append(obs.copy())
                    actions.append(action[0].copy())
                    rewards.append(reward)
                    done = terminated or truncated
                    if args.terminal and (len(actions) % 10 == 0 or done):
                        print(f"\nLayout {layout + 1} shot {shot + 1} step {len(actions)} [{info['outcome']}]\n"
                              + base_env.terminal_frame(), flush=True)
                        time.sleep(0.05)
                    if renderer is not None:
                        cv2.imshow("Precision striker (q to quit)", renderer.get_frame())
                        if cv2.waitKey(20) & 0xFF == ord("q"):
                            quit_requested = True
                            break
                if quit_requested:
                    break
                trajectory = f"layout_{layout:02d}_shot_{shot:02d}.npz"
                np.savez_compressed(args.output / trajectory, observations=observations,
                                    actions=actions, rewards=rewards, inference_ms=latencies)
                record = {"layout": layout, "shot": shot, "seed": seed, "blocks": blocks,
                          "outcome": info["outcome"], "success": info["is_success"],
                          "initial_velocity": reset_info["initial_velocity"], "hit": info["hit"],
                          "scorable": reset_info.get("scorable"),
                          "obstacle_collision": info["obstacle_collision"], "steps": len(actions),
                          "trajectory": trajectory, "inference_p95_ms": float(np.percentile(latencies, 95))}
                records.append(record)
                print(json.dumps(record), flush=True)
            if quit_requested:
                break
    finally:
        normalizer.close()
        if renderer is not None:
            cv2.destroyAllWindows()
    total = len(records)
    successes = sum(r["success"] for r in records)
    collisions = sum(r["obstacle_collision"] for r in records)
    rate = successes / total if total else 0
    # Wilson interval communicates uncertainty; acceptance remains the report's observed-rate test.
    z = 1.96
    denom = 1 + z*z / total if total else 1
    center = (rate + z*z/(2*total)) / denom if total else 0
    half = z * np.sqrt(rate*(1-rate)/total + z*z/(4*total*total)) / denom if total else 0
    summary = {"task_version": TASK_VERSION, "run": str(args.run), "seed": args.seed,
               "task_options": options,
               "episodes": total, "successes": successes, "scoring_rate": rate,
               # Comparable across v2/v3: goal outcome with paddle contact at any earlier step.
               "raw_goals": sum(r["outcome"] == "goal" for r in records),
               "goals_after_hit": sum(r["outcome"] == "goal" and r["hit"] for r in records),
               "goals_off_obstacle": sum(r["outcome"] == "goal" and r["obstacle_collision"] for r in records),
               "unscorable_shots": sum(r.get("scorable") is False for r in records),
               "wilson_95_interval": [max(0.0, center-half), min(1.0, center+half)], "collision_episodes": collisions,
               "outcomes": dict(Counter(r["outcome"] for r in records)),
               "completed": total == args.layouts * args.shots,
               "report_protocol": args.layouts == 10 and args.shots == 20 and args.puck is None
                                  and fixed_blocks is None and args.obstacles > 0 and args.velocity is None,
               "records": records}
    summary["meets_report_target"] = bool(summary["report_protocol"] and total == 200
                                           and rate >= 0.8 and collisions == 0)
    (args.output / "evaluation.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
