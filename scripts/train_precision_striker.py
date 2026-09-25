"""Headless PPO training. Run from the repository root with python -m scripts..."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import torch
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from scripts.precision_striker_env import PrecisionStrikerEnv, TASK_VERSION


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--envs", type=int, default=8)
    p.add_argument("--obstacles", type=int, choices=range(4), default=1)
    p.add_argument("--speed-min", type=float, default=0.3)
    p.add_argument("--speed-max", type=float, default=1.5)
    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    p.add_argument("--vector", choices=["dummy", "subproc"], default="subproc")
    p.add_argument("--rollout", type=int, default=512)
    p.add_argument("--checkpoint-every", type=int, default=50_000)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume", type=Path, help="Previous run directory (additional --steps)")
    args = p.parse_args()
    if min(args.steps, args.envs, args.rollout, args.checkpoint_every) < 1 or args.rollout * args.envs < 2:
        p.error("steps, envs, rollout and checkpoint-every must be positive; rollout batch >= 2")
    if args.device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA requested but unavailable; use a GPU allocation and CUDA PyTorch")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)

    def factory(rank):
        def build():
            return Monitor(PrecisionStrikerEnv(seed=args.seed + rank, obstacles=args.obstacles,
                                              speed_min=args.speed_min, speed_max=args.speed_max),
                           str(args.output / f"worker_{rank}"),
                           info_keywords=("is_success", "obstacle_collision", "outcome", "hit"))
        return build

    vector_type = SubprocVecEnv if args.vector == "subproc" else DummyVecEnv
    venv = vector_type([factory(i) for i in range(args.envs)])
    if args.resume:
        previous = json.loads((args.resume / "manifest.json").read_text())
        if previous["task_version"] != TASK_VERSION:
            raise ValueError("Resume task version does not match")
        venv = VecNormalize.load(str(args.resume / "vecnormalize.pkl"), venv)
        venv.training = True
        model = PPO.load(str(args.resume / "policy.zip"), env=venv, device=args.device)
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=False)
        batch = min(256, args.rollout * args.envs)
        while (args.rollout * args.envs) % batch:
            batch -= 1
        model = PPO("MlpPolicy", venv, n_steps=args.rollout, batch_size=batch,
                    learning_rate=3e-4, gamma=0.99, device=args.device, seed=args.seed,
                    ent_coef=0.01, n_epochs=10, gae_lambda=0.95, clip_range=0.2,
                    vf_coef=0.5, max_grad_norm=0.5,
                    policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128]),
                                       activation_fn=torch.nn.Tanh, log_std_init=-0.7),
                    verbose=1, tensorboard_log=str(args.output / "tensorboard"))
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    manifest = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "task_version": TASK_VERSION, "git_commit": commit,
                "torch": torch.__version__, "sb3": stable_baselines3.__version__,
                "python": platform.python_version(), "actual_device": str(model.device)}
    root = Path(__file__).resolve().parents[1]
    manifest["source_sha256"] = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ["scripts/precision_striker_env.py", "scripts/train_precision_striker.py",
                     "airhockey/sims/airhockey_box2d.py", "configs/flat_table_play.yaml"]}
    manifest["actual_rollout"] = model.n_steps
    manifest["actual_batch_size"] = model.batch_size
    manifest["policy_architecture"] = str(model.policy)
    manifest["ppo"] = {"learning_rate": 3e-4, "gamma": model.gamma, "gae_lambda": model.gae_lambda,
                       "n_epochs": model.n_epochs, "ent_coef": model.ent_coef,
                       "vf_coef": model.vf_coef, "max_grad_norm": model.max_grad_norm}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    callback = CheckpointCallback(save_freq=max(args.checkpoint_every // args.envs, 1),
                                  save_path=str(args.output / "checkpoints"),
                                  name_prefix="policy", save_vecnormalize=True)
    print(json.dumps(manifest, indent=2), flush=True)
    try:
        model.learn(total_timesteps=args.steps, callback=callback,
                    reset_num_timesteps=args.resume is None, progress_bar=False)
    finally:
        model.save(str(args.output / "policy"))
        venv.save(str(args.output / "vecnormalize.pkl"))
        manifest["completed_timesteps"] = model.num_timesteps
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
        venv.close()
    print(f"Saved policy and normalization to {args.output}", flush=True)


if __name__ == "__main__":
    main()
