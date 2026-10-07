"""Behaviour cloning from the strike planner into a standard PPO policy (imitation, then RL).

1. Roll out the hand-coded StrikePlanner in the training distribution and record
   (observation, planner action, discounted return).
2. Fit observation normalisation (VecNormalize statistics) to the data.
3. Train the PPO actor mean to reproduce the planner's actions and the critic to predict returns.
4. Save policy.zip + vecnormalize.pkl + manifest.json, which train_precision_striker.py can --resume
   to fine-tune with PPO. The result is a pure neural policy: no planner is used at run time.
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


def collect(job):
    seed, steps, task_options, planner_kwargs, gamma = job
    env = PrecisionStrikerEnv(seed=seed, obstacles=1, **task_options)
    planner = StrikePlanner(**planner_kwargs)
    obs_buf, act_buf, ret_buf, successes, episodes = [], [], [], 0, 0
    rng = np.random.RandomState(seed)
    while len(obs_buf) < steps:
        obs, _ = env.reset(seed=int(rng.randint(0, 2**31 - 1)))
        planner.reset()
        ep_obs, ep_act, ep_rew = [], [], []
        done = False
        while not done:
            a = planner.act(obs)
            ep_obs.append(obs); ep_act.append(a)
            obs, r, terminated, truncated, info = env.step(a)
            ep_rew.append(r)
            done = terminated or truncated
        g, rets = 0.0, []
        for r in reversed(ep_rew):
            g = r + gamma * g
            rets.append(g)
        obs_buf += ep_obs; act_buf += ep_act; ret_buf += rets[::-1]
        successes += bool(info.get("is_success")); episodes += 1
    env.close()
    return np.array(obs_buf, np.float32), np.array(act_buf, np.float32), np.array(ret_buf, np.float32), successes, episodes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=1_000_000, help="Demonstration environment steps")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--net", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--lr", type=float, default=1e-4, help="PPO learning rate stored for fine-tuning")
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--log-std", type=float, default=-1.6, help="Initial exploration noise for fine-tuning")
    p.add_argument("--planner", type=str, default="{}", help="JSON planner settings")
    p.add_argument("--extra-obs", action="store_true")
    p.add_argument("--goal-shaping", action="store_true")
    args = p.parse_args()
    task_options = {"require_hit": True, "timeout_penalty": 10.0, "obstacle_bounce": True,
                    "extra_obs": args.extra_obs, "goal_shaping": args.goal_shaping}
    planner_kwargs = json.loads(args.planner)
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    started = time.perf_counter()
    per = -(-args.steps // args.workers)
    with Pool(args.workers) as pool:
        parts = pool.map(collect, [(args.seed * 1000 + i, per, task_options, planner_kwargs, args.gamma)
                                   for i in range(args.workers)])
    obs = np.concatenate([x[0] for x in parts]); act = np.concatenate([x[1] for x in parts])
    ret = np.concatenate([x[2] for x in parts])
    demo_success = sum(x[3] for x in parts) / max(1, sum(x[4] for x in parts))
    print(f"Collected {len(obs)} steps, planner success {demo_success:.3f}, "
          f"{time.perf_counter() - started:.0f}s", flush=True)

    venv = VecNormalize(DummyVecEnv([lambda: PrecisionStrikerEnv(obstacles=1, **task_options)]),
                        norm_obs=True, norm_reward=False)
    venv.obs_rms.mean = obs.mean(0).astype(np.float64)
    venv.obs_rms.var = obs.var(0).astype(np.float64) + 1e-8
    venv.obs_rms.count = float(len(obs))
    model = PPO("MlpPolicy", venv, n_steps=512, batch_size=1024, learning_rate=args.lr, gamma=args.gamma,
                ent_coef=args.ent_coef, n_epochs=10, gae_lambda=0.95, clip_range=0.2, vf_coef=0.5,
                max_grad_norm=0.5, seed=args.seed, device="cpu",
                policy_kwargs=dict(net_arch=dict(pi=[args.net, args.net], vf=[args.net, args.net]),
                                   activation_fn=torch.nn.Tanh, log_std_init=args.log_std))
    x = torch.as_tensor(np.clip((obs - venv.obs_rms.mean) / np.sqrt(venv.obs_rms.var), -10, 10), dtype=torch.float32)
    y_act = torch.as_tensor(act); y_ret = torch.as_tensor(ret)
    policy = model.policy
    params = [p for n, p in policy.named_parameters() if n != "log_std"]
    opt = torch.optim.Adam(params, lr=1e-3)
    n = len(x)
    history = []
    for epoch in range(args.epochs):
        perm = torch.randperm(n)
        tot_a = tot_v = 0.0
        for i in range(0, n, 4096):
            idx = perm[i:i + 4096]
            latent_pi, latent_vf = policy.mlp_extractor(policy.extract_features(x[idx]))
            mean = policy.action_net(latent_pi)
            value = policy.value_net(latent_vf).squeeze(-1)
            loss_a = torch.mean((mean - y_act[idx]) ** 2)
            loss_v = torch.mean((value - y_ret[idx]) ** 2)
            opt.zero_grad(); (loss_a + 0.01 * loss_v).backward(); opt.step()
            tot_a += loss_a.item() * len(idx); tot_v += loss_v.item() * len(idx)
        history.append({"epoch": epoch, "action_mse": tot_a / n, "value_mse": tot_v / n})
        print(json.dumps(history[-1]), flush=True)
    model.save(str(args.output / "policy"))
    venv.save(str(args.output / "vecnormalize.pkl"))
    manifest = {"task_version": task_version(**task_options), "task_options": task_options,
                "algo": "ppo", "hybrid": "none", "completed_timesteps": 0, "bc": True,
                "demo_steps": int(len(obs)), "demo_planner_success": demo_success,
                "planner": planner_kwargs, "net": args.net, "history": history,
                "speed_min": 0.3, "speed_max": 1.5, "obstacles": 1}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Saved behaviour-cloned policy to {args.output}")


if __name__ == "__main__":
    main()
