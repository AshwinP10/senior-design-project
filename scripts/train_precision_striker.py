"""Headless PPO/SAC training, optionally on top of the strike controller (--hybrid). Run from the repository root with python -m scripts..."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import torch
import stable_baselines3
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from scripts.hybrid_envs import make_env
from scripts.precision_striker_env import task_version


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
    p.add_argument("--require-hit", action="store_true",
                   help="v3 task: a goal earns reward and counts as success only after paddle contact")
    p.add_argument("--timeout-penalty", type=float, default=2.0,
                   help="Cost of reaching the time limit (v2 uses 2)")
    p.add_argument("--obstacle-bounce", action="store_true",
                   help="v4 task: the puck bounces off obstacles; contact is neither terminal nor penalized")
    p.add_argument("--resume-new-task", action="store_true",
                   help="Allow --resume from a run trained on different task options (fine-tuning)")
    p.add_argument("--extra-obs", action="store_true",
                   help="v5 task: also observe [paddle has hit, fraction of time used] (22 inputs)")
    p.add_argument("--goal-shaping", action="store_true",
                   help="After the hit, reward progress toward the goal mouth instead of -x progress")
    # PPO optimization settings (defaults reproduce the earlier experiments).
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-schedule", choices=["constant", "linear"], default="constant",
                   help="linear: decay the learning rate to 0 over this run's --steps")
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--net", type=int, default=128, help="Width of both hidden layers (actor and critic)")
    p.add_argument("--hybrid", choices=["none", "shot", "residual"], default="none",
                   help="shot: policy picks each shot for the strike controller; residual: policy corrects it")
    p.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    p.add_argument("--residual-scale", type=float, default=0.5)
    p.add_argument("--log-std", type=float, default=-0.7, help="Initial log std of the Gaussian policy")
    p.add_argument("--planner", type=str, default="{}", help="JSON StrikePlanner settings for --hybrid")
    args = p.parse_args()
    planner_kwargs = json.loads(args.planner)
    task_options = {"require_hit": args.require_hit, "timeout_penalty": args.timeout_penalty,
                    "obstacle_bounce": args.obstacle_bounce, "extra_obs": args.extra_obs,
                    "goal_shaping": args.goal_shaping}
    TASK_VERSION = task_version(**task_options)
    if min(args.steps, args.envs, args.rollout, args.checkpoint_every) < 1 or args.rollout * args.envs < 2:
        p.error("steps, envs, rollout and checkpoint-every must be positive; rollout batch >= 2")
    if args.device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA requested but unavailable; use a GPU allocation and CUDA PyTorch")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)

    def factory(rank):
        def build():
            extra = {"scale": args.residual_scale} if args.hybrid == "residual" else {}
            if args.hybrid != "none":
                extra["planner"] = planner_kwargs
            return Monitor(make_env(args.hybrid, seed=args.seed + rank, obstacles=args.obstacles,
                                    speed_min=args.speed_min, speed_max=args.speed_max,
                                    **task_options, **extra),
                           str(args.output / f"worker_{rank}"),
                           info_keywords=("is_success", "obstacle_collision", "outcome", "hit"))
        return build

    vector_type = SubprocVecEnv if args.vector == "subproc" else DummyVecEnv
    venv = vector_type([factory(i) for i in range(args.envs)])
    if args.resume:
        previous = json.loads((args.resume / "manifest.json").read_text())
        previous_options = {"require_hit": False, "timeout_penalty": 2.0, "obstacle_bounce": False,
                            "extra_obs": False, "goal_shaping": False, **previous.get("task_options", {})}
        if previous_options["extra_obs"] != args.extra_obs:
            raise ValueError("Cannot resume across observation sizes (--extra-obs must match)")
        if not args.resume_new_task and (previous["task_version"] != TASK_VERSION
                                         or previous_options != task_options):
            raise ValueError("Resume task version/options do not match; pass the same task flags "
                             "or --resume-new-task to fine-tune on a different task")
        venv = VecNormalize.load(str(args.resume / "vecnormalize.pkl"), venv)
        venv.training = True
        Algo = SAC if previous.get("algo", "ppo") == "sac" else PPO
        model = Algo.load(str(args.resume / "policy.zip"), env=venv, device=args.device)
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=False)
        batch = min(args.batch_size, args.rollout * args.envs)
        while (args.rollout * args.envs) % batch:
            batch -= 1
        # SB3 calls the schedule with progress_remaining, 1 -> 0 over this learn() call.
        lr = args.lr if args.lr_schedule == "constant" else (lambda remaining, base=args.lr: base * remaining)
        if args.algo == "sac":
            model = SAC("MlpPolicy", venv, learning_rate=lr, buffer_size=1_000_000, batch_size=args.batch_size,
                        gamma=args.gamma, learning_starts=5_000, train_freq=1,
                        gradient_steps=max(1, args.envs // 2), device=args.device, seed=args.seed,
                        policy_kwargs=dict(net_arch=[args.net, args.net]),
                        verbose=1, tensorboard_log=str(args.output / "tensorboard"))
        else:
            model = PPO("MlpPolicy", venv, n_steps=args.rollout, batch_size=batch,
                        learning_rate=lr, gamma=args.gamma, device=args.device, seed=args.seed,
                        ent_coef=args.ent_coef, n_epochs=args.n_epochs, gae_lambda=0.95, clip_range=0.2,
                        vf_coef=0.5, max_grad_norm=0.5,
                        policy_kwargs=dict(net_arch=dict(pi=[args.net, args.net], vf=[args.net, args.net]),
                                           activation_fn=torch.nn.Tanh, log_std_init=args.log_std),
                        verbose=1, tensorboard_log=str(args.output / "tensorboard"))
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    manifest = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "task_version": TASK_VERSION, "git_commit": commit,
                "task_options": task_options,
                "torch": torch.__version__, "sb3": stable_baselines3.__version__,
                "python": platform.python_version(), "actual_device": str(model.device)}
    root = Path(__file__).resolve().parents[1]
    manifest["source_sha256"] = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ["scripts/precision_striker_env.py", "scripts/train_precision_striker.py",
                     "airhockey/sims/airhockey_box2d.py", "configs/flat_table_play.yaml"]}
    manifest["actual_rollout"] = getattr(model, "n_steps", None)
    manifest["actual_batch_size"] = model.batch_size
    manifest["policy_architecture"] = str(model.policy)
    if isinstance(model, PPO):
        manifest["ppo"] = {"learning_rate": args.lr, "lr_schedule": args.lr_schedule,
                           "gamma": model.gamma, "gae_lambda": model.gae_lambda,
                           "n_epochs": model.n_epochs, "ent_coef": model.ent_coef,
                           "vf_coef": model.vf_coef, "max_grad_norm": model.max_grad_norm}
    else:
        manifest["sac"] = {"learning_rate": args.lr, "gamma": model.gamma, "batch_size": model.batch_size,
                           "buffer_size": model.buffer_size, "gradient_steps": model.gradient_steps}
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
