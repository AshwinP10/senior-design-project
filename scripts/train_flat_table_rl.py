"""
Train PPO (Stable-Baselines3) on AirHockeyFlatTableEnv with shaped rewards and VecNormalize.
"""
from __future__ import annotations

import argparse
import os
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

_repo_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
_scripts = os.path.join(_repo_root, "scripts")
for _p in (_repo_root, _scripts):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import flat_table_rl_wrappers as ftw  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        type=str,
        default=os.path.join(_repo_root, "configs", "flat_table_play.yaml"),
        help="YAML config with air_hockey.flat_table settings",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--curriculum-steps", type=int, default=250_000, help="0 disables curriculum")
    parser.add_argument(
        "--blocks",
        type=str,
        default=None,
        help='Optional "x,y;x2,y2" user blocks preserved for curriculum phase 2',
    )
    parser.add_argument("--tensorboard", type=str, default=os.path.join(_repo_root, "tensorboard_logs", "ppo_flat_table"))
    parser.add_argument("--models-dir", type=str, default=os.path.join(_repo_root, "models"))
    args = parser.parse_args()

    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.tensorboard, exist_ok=True)

    user_blocks = ftw.parse_user_blocks_arg(args.blocks)
    curric = int(args.curriculum_steps)

    def make_env(rank: int):
        def _init():
            e = ftw.build_flat_table_train_env(
                args.cfg,
                seed=args.seed + rank,
                curriculum_steps=curric,
                user_blocks=user_blocks if user_blocks else None,
            )
            return Monitor(e, filename=os.path.join(args.models_dir, f"monitor_{rank}"), allow_early_resets=True)

        return _init

    venv = DummyVecEnv([make_env(i) for i in range(args.n_envs)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)

    model = PPO(
        "MlpPolicy",
        venv,
        learning_rate=3e-4,
        gamma=0.99,
        batch_size=256,
        n_steps=2048,
        verbose=1,
        tensorboard_log=args.tensorboard,
        device="auto",
        seed=args.seed,
    )

    ckpt_kwargs = dict(
        save_freq=50_000,
        save_path=args.models_dir,
        name_prefix="ppo_flat_table",
    )
    try:
        ckpt = CheckpointCallback(save_vecnormalize=True, **ckpt_kwargs)
    except TypeError:
        ckpt = CheckpointCallback(**ckpt_kwargs)

    try:
        model.learn(total_timesteps=args.total_timesteps, callback=ckpt, progress_bar=True)
    except ImportError:
        model.learn(total_timesteps=args.total_timesteps, callback=ckpt, progress_bar=False)

    final_base = os.path.join(args.models_dir, "ppo_flat_table")
    model.save(final_base)
    venv.save(final_base + "_vecnormalize.pkl")
    print("Saved", final_base + ".zip and", final_base + "_vecnormalize.pkl")


if __name__ == "__main__":
    main()
