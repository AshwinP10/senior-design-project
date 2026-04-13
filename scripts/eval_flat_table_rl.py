"""
Evaluate a trained PPO policy on the flat table with OpenCV rendering; supports user-placed blocks.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

_repo_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
_scripts = os.path.join(_repo_root, "scripts")
for _p in (_repo_root, _scripts):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import flat_table_rl_wrappers as ftw  # noqa: E402
from airhockey.renderers import AirHockeyRenderer  # noqa: E402


def _build_eval_env(cfg_path: str, seed: int, user_blocks):
    import copy

    base = ftw.load_flat_table_config(cfg_path)
    base["seed"] = int(seed)
    env = ftw.AirHockeyFlatTableRandEnv(
        puck_position_noise=0.0,
        paddle_position_noise=0.0,
        damping_noise_frac=0.0,
        **base,
    )
    if user_blocks:
        env.user_block_positions = copy.deepcopy(list(user_blocks))
    env = ftw.ActionClipWrapper(env)
    env = ftw.ObsFloat32Wrapper(env)
    return env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=os.path.join(_repo_root, "models", "ppo_flat_table.zip"))
    parser.add_argument(
        "--vecnorm",
        type=str,
        default=None,
        help="VecNormalize stats (default: models/ppo_flat_table_vecnormalize.pkl next to --model)",
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default=os.path.join(_repo_root, "configs", "flat_table_play.yaml"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--blocks", type=str, default=None, help='Optional "x,y;x2,y2" static blocks')
    args = parser.parse_args()

    vecnorm_path = args.vecnorm
    if vecnorm_path is None:
        root, _ = os.path.splitext(args.model)
        candidate = root + "_vecnormalize.pkl"
        if os.path.isfile(candidate):
            vecnorm_path = candidate

    user_blocks = ftw.parse_user_blocks_arg(args.blocks)

    def make_mon():
        return Monitor(_build_eval_env(args.cfg, args.seed, user_blocks), allow_early_resets=True)

    venv = DummyVecEnv([make_mon])
    if vecnorm_path and os.path.isfile(vecnorm_path):
        venv = VecNormalize.load(vecnorm_path, venv)
        venv.training = False
        venv.norm_reward = False
    else:
        print("No VecNormalize file; using raw observations (only valid if training ran without VecNormalize).")

    model = PPO.load(args.model, env=venv, device="auto")

    ah = venv.venv.envs[0].unwrapped
    renderer = AirHockeyRenderer(ah, orientation="vertical", robosuite_view="")
    renderer.show_target_position = False
    win = "PPO flat table eval (q to quit)"
    cv2.namedWindow(win)

    obs = venv.reset()
    ep = 0
    while ep < args.episodes:
        frame = renderer.get_frame()
        cv2.imshow(win, frame)
        key = cv2.waitKey(1)
        if key in (ord("q"), ord("Q"), 27):
            break

        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, _ = venv.step(action)
        if dones[0]:
            ep += 1
            obs = venv.reset()
        time.sleep(0.001)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
