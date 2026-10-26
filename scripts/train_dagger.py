"""DAgger: iterative imitation of the strike controller, producing a pure neural PPO policy.

Plain imitation (train_bc.py) only shows the network states the controller itself visits; once the
network makes a small error it reaches states it never saw, and errors compound. DAgger fixes this:

  iteration 0   the controller plays; record (state, controller action)            [= imitation]
  iteration k   the network plays (the controller acts with probability beta**k); the controller
                still computes its action in every visited state, which becomes the label
  after each    retrain the network on all labels so far; score it on validation shots
                (seed 50000, never the test seed) and keep the best iteration

The result is saved like a PPO run (policy.zip, vecnormalize.pkl, manifest.json), so
train_precision_striker.py can --resume it for PPO fine-tuning, and eval_precision_striker.py can test it.
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from scripts.precision_striker_env import PrecisionStrikerEnv, task_version
from scripts.strike_planner import StrikePlanner

TASK = {"require_hit": True, "timeout_penalty": 10.0, "obstacle_bounce": True}


def _load(run):
    model = PPO.load(str(Path(run) / "policy.zip"), device="cpu")
    probe = PrecisionStrikerEnv(obstacles=1, **TASK, **json.loads((Path(run) / "obs_options.json").read_text()))
    norm = VecNormalize.load(str(Path(run) / "vecnormalize.pkl"), DummyVecEnv([lambda: probe]))
    norm.training = False
    return model, norm


def collect(job):
    """Roll out the current network (or the controller with probability beta); label with the controller."""
    seed, steps, run, beta, obs_options, planner_kwargs, gamma = job
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    env = PrecisionStrikerEnv(seed=seed, obstacles=1, **TASK, **obs_options)
    planner = StrikePlanner(**planner_kwargs)
    model, norm = (None, None) if beta >= 1.0 else _load(run)
    X, Y, R, wins, episodes = [], [], [], 0, 0
    while len(X) < steps:
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        planner.reset()
        rewards, done = [], False
        while not done:
            expert = planner.act(obs)                  # always computed: it is the label
            X.append(obs); Y.append(expert)
            if model is None or rng.random() < beta:
                action = expert
            else:
                action = model.predict(norm.normalize_obs(obs[None, :]), deterministic=False)[0][0]
            obs, r, terminated, truncated, info = env.step(action)
            rewards.append(r)
            done = terminated or truncated
        g = 0.0
        returns = []
        for r in reversed(rewards):
            g = r + gamma * g
            returns.append(g)
        R += returns[::-1]
        wins += bool(info.get("is_success")); episodes += 1
    env.close()
    return (np.array(X, np.float32), np.array(Y, np.float32), np.array(R, np.float32), wins, episodes)


def validate(job):
    """Deterministic network on part of the validation shots (seed 50000): goals after a hit."""
    run, layouts, obs_options = job
    torch.set_num_threads(1)
    model, norm = _load(run)
    env = PrecisionStrikerEnv(seed=50000, obstacles=1, randomize=False, **TASK, **obs_options)
    goals = shots = 0
    for layout in layouts:
        blocks = env.sample_layout(50000 + layout)
        for shot in range(20):
            obs, _ = env.reset(seed=51000 + layout * 20 + shot, options={"blocks": blocks})
            done = False
            while not done:
                obs, _, te, tr, info = env.step(model.predict(norm.normalize_obs(obs[None, :]), deterministic=True)[0][0])
                done = te or tr
            goals += int(info["outcome"] == "goal" and info["hit"]); shots += 1
    env.close()
    return goals, shots


def fit(model, norm, X, Y, R, epochs, lr, value_weight=0.01):
    policy = model.policy
    x = torch.as_tensor(np.clip((X - norm.obs_rms.mean) / np.sqrt(norm.obs_rms.var + 1e-8), -10, 10), dtype=torch.float32)
    ya, yr = torch.as_tensor(Y), torch.as_tensor(R)
    opt = torch.optim.Adam([p for n, p in policy.named_parameters() if n != "log_std"], lr=lr)
    n = len(x)
    for _ in range(epochs):
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, 4096):
            idx = perm[i:i + 4096]
            latent_pi, latent_vf = policy.mlp_extractor(policy.extract_features(x[idx]))
            loss_a = torch.mean((policy.action_net(latent_pi) - ya[idx]) ** 2)
            loss_v = torch.mean((policy.value_net(latent_vf).squeeze(-1) - yr[idx]) ** 2)
            opt.zero_grad(); (loss_a + value_weight * loss_v).backward(); opt.step()
            total += loss_a.item() * len(idx)
    return total / n


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--iterations", type=int, default=8)
    p.add_argument("--steps-per-iter", type=int, default=400_000)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--beta-decay", type=float, default=0.5, help="Controller acts with prob decay**k at iteration k")
    p.add_argument("--net", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--lr", type=float, default=1e-4, help="PPO learning rate stored for later fine-tuning")
    p.add_argument("--log-std", type=float, default=-1.6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--planner", type=str, default="{}", help="JSON StrikePlanner settings (the teacher)")
    p.add_argument("--extra-obs", action="store_true")
    p.add_argument("--goal-shaping", action="store_true")
    p.add_argument("--val-layouts", type=int, default=10)
    p.add_argument("--wide-obstacles", action="store_true", help="Obstacles anywhere a user may place one")
    p.add_argument("--obstacles-max", type=int, help="Random obstacle count per episode: 1..this")
    args = p.parse_args()
    # Environment options beyond the fixed task rules (saved with every iteration and in the manifest).
    obs_options = {"extra_obs": args.extra_obs, "goal_shaping": args.goal_shaping}
    if args.wide_obstacles:
        obs_options["wide_obstacles"] = True
    if args.obstacles_max:
        obs_options["obstacles_max"] = args.obstacles_max
    planner_kwargs = json.loads(args.planner)
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    history, data = [], None
    model = norm = None
    best = (-1, None)
    with Pool(args.workers) as pool:
        for k in range(args.iterations):
            started = time.perf_counter()
            beta = 1.0 if k == 0 else args.beta_decay ** k
            it_dir = args.output / f"iter{k}"
            it_dir.mkdir()
            prev = args.output / f"iter{k - 1}" if k else None
            per = -(-args.steps_per_iter // args.workers)
            parts = pool.map(collect, [(args.seed * 100_000 + k * 1000 + i, per, prev, beta, obs_options,
                                        planner_kwargs, args.gamma) for i in range(args.workers)])
            X = np.concatenate([q[0] for q in parts]); Y = np.concatenate([q[1] for q in parts])
            R = np.concatenate([q[2] for q in parts])
            rollout_success = sum(q[3] for q in parts) / max(1, sum(q[4] for q in parts))
            data = (X, Y, R) if data is None else tuple(np.concatenate([a, b]) for a, b in zip(data, (X, Y, R)))
            if model is None:                         # normalisation fixed from the controller's own states
                venv = VecNormalize(DummyVecEnv([lambda: PrecisionStrikerEnv(obstacles=1, **TASK, **obs_options)]),
                                    norm_obs=True, norm_reward=False)
                venv.obs_rms.mean = X.mean(0).astype(np.float64)
                venv.obs_rms.var = X.var(0).astype(np.float64) + 1e-8
                venv.obs_rms.count = float(len(X))
                norm = venv
                model = PPO("MlpPolicy", venv, n_steps=512, batch_size=1024, learning_rate=args.lr, gamma=args.gamma,
                            ent_coef=0.0, n_epochs=10, gae_lambda=0.95, clip_range=0.2, vf_coef=0.5,
                            max_grad_norm=0.5, seed=args.seed, device="cpu",
                            policy_kwargs=dict(net_arch=dict(pi=[args.net, args.net], vf=[args.net, args.net]),
                                               activation_fn=torch.nn.Tanh, log_std_init=args.log_std))
            mse = fit(model, norm, *data, epochs=args.epochs if k else 3 * args.epochs, lr=1e-3 if k == 0 else 3e-4)
            model.save(str(it_dir / "policy")); norm.save(str(it_dir / "vecnormalize.pkl"))
            (it_dir / "obs_options.json").write_text(json.dumps(obs_options))
            chunks = [[l] for l in range(args.val_layouts)]
            vals = pool.map(validate, [(it_dir, c, obs_options) for c in chunks])
            val = sum(g for g, _ in vals) / max(1, sum(s for _, s in vals))
            history.append({"iteration": k, "beta": beta, "new_steps": int(len(X)), "total_steps": int(len(data[0])),
                            "rollout_success": rollout_success, "action_mse": mse, "validation_accuracy": val,
                            "seconds": round(time.perf_counter() - started)})
            print(json.dumps(history[-1]), flush=True)
            if val > best[0]:
                best = (val, k)
    k = best[1]
    src = args.output / f"iter{k}"
    for name in ("policy.zip", "vecnormalize.pkl"):
        (args.output / name).write_bytes((src / name).read_bytes())
    task_options = {**TASK, **obs_options}
    manifest = {"task_version": task_version(**task_options), "task_options": task_options, "algo": "ppo",
                "hybrid": "none", "completed_timesteps": 0, "dagger": True, "best_iteration": k,
                "best_validation_accuracy": best[0], "history": history, "planner": planner_kwargs,
                "net": args.net, "obstacles": 1, "speed_min": 0.3, "speed_max": 1.5}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Best iteration {k}: validation {best[0]:.3f}. Saved to {args.output}")


if __name__ == "__main__":
    main()
