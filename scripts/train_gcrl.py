"""
Goal-conditioned RL policy for air hockey (puck_goal_position task).

The agent observes [paddle_x, paddle_y, puck_x, puck_y, goal_x, goal_y] (6D flat)
and outputs a 2D paddle velocity action in [-1, 1].

PPO is used with n_epochs=1 (single gradient epoch per rollout).

Run:
    python scripts/train_gcrl.py
    python scripts/train_gcrl.py --render          # live OpenCV window during training
    python scripts/train_gcrl.py --steps 200000
    python scripts/train_gcrl.py --cfg configs/baseline_configs/box2d/gcrl_pos.yaml
"""
from __future__ import annotations

import argparse
import copy
import os
import sys

import numpy as np
import yaml
import gymnasium as gym

_repo_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from airhockey import AirHockeyEnv
from airhockey.renderers import AirHockeyRenderer


# ---------------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------------

class ObsFloat32Wrapper(gym.ObservationWrapper):
    """Cast observations to float32 for SB3 compatibility."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        sp = env.observation_space
        self.observation_space = gym.spaces.Box(
            low=sp.low.astype(np.float32),
            high=sp.high.astype(np.float32),
            dtype=np.float32,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return np.asarray(obs, dtype=np.float32)


class ActionClipWrapper(gym.ActionWrapper):
    """Clip actions to the environment's action bounds."""

    def action(self, act: np.ndarray) -> np.ndarray:
        return np.clip(
            act, self.env.action_space.low, self.env.action_space.high
        ).astype(np.float64)


# ---------------------------------------------------------------------------
# Live render callback
# ---------------------------------------------------------------------------

class LiveRenderCallback(BaseCallback):
    """
    Every `render_freq` training steps: run one eval episode in a fresh copy
    of the environment and show each frame in an OpenCV window.

    Press Q in the window to skip the current render early.
    """

    WINDOW = "Air Hockey — Training Live View"

    def __init__(self, params: dict, render_freq: int = 5_000, verbose: int = 0):
        super().__init__(verbose)
        self._params = params
        self._render_freq = render_freq
        self._last_render = 0
        import cv2
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW, 400, 720)

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_render < self._render_freq:
            return True
        self._last_render = self.num_timesteps
        self._render_episode()
        return True

    def _render_episode(self):
        import cv2

        cfg = copy.deepcopy(self._params)
        cfg["seed"] = 0
        env = AirHockeyEnv(cfg)
        renderer = AirHockeyRenderer(env, orientation="vertical", robosuite_view="")

        obs, _ = env.reset()
        done = False
        step = 0
        ep_reward = 0.0

        while not done:
            # Normalise obs the same way VecNormalize does (use running stats)
            obs_t = self.model.policy.obs_to_tensor(
                obs.reshape(1, -1).astype(np.float32)
            )[0]
            with __import__("torch").no_grad():
                action = self.model.policy._predict(obs_t, deterministic=True)
            action_np = action.cpu().numpy().flatten()

            obs, reward, terminated, truncated, _ = env.step(action_np)
            ep_reward += float(reward)
            done = terminated or truncated
            step += 1

            frame = renderer.get_frame()

            # HUD overlay
            label = (f"Step {self.num_timesteps:,} | "
                     f"Ep step {step} | "
                     f"Reward {ep_reward:.2f}")
            cv2.putText(frame, label, (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
            cv2.putText(frame, label, (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            cv2.imshow(self.WINDOW, frame)
            key = cv2.waitKey(16)  # ~60 fps cap
            if key in (ord("q"), ord("Q"), 27):
                break

        env.close()

    def _on_training_end(self) -> None:
        import cv2
        cv2.destroyWindow(self.WINDOW)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(cfg_path: str):
    with open(cfg_path, "r") as f:
        full = yaml.safe_load(f)

    params = copy.deepcopy(full["air_hockey"])

    # Flat obs: env concatenates [state | desired_goal] instead of a Dict space
    params["return_goal_obs"] = False
    params["n_training_steps"] = full.get("n_training_steps", 1_000_000)

    seed_raw = full.get("seed", 0)
    seed = int(seed_raw[0]) if isinstance(seed_raw, list) else int(seed_raw)

    return params, full, seed


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_env(params: dict, seed: int):
    def _thunk():
        cfg = copy.deepcopy(params)
        cfg["seed"] = seed
        env = AirHockeyEnv(cfg)
        env = ActionClipWrapper(env)
        env = ObsFloat32Wrapper(env)
        env = Monitor(env)
        return env
    return _thunk


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train a goal-conditioned PPO policy for air hockey.")
    parser.add_argument(
        "--cfg",
        default=os.path.join(_repo_root, "configs", "baseline_configs", "box2d", "gcrl_pos.yaml"),
        help="Path to YAML config (default: configs/baseline_configs/box2d/gcrl_pos.yaml)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Total environment steps to train (overrides config value)",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(_repo_root, "models", "ppo_gcrl"),
        help="Base path for saved model (no extension). Default: models/ppo_gcrl",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Show a live OpenCV render window during training (every 5k steps)",
    )
    parser.add_argument(
        "--render-freq",
        type=int,
        default=5_000,
        help="How often (in env steps) to show a rendered eval episode (default: 5000)",
    )
    args = parser.parse_args()

    params, full_cfg, seed = load_config(args.cfg)

    total_steps = args.steps if args.steps is not None else full_cfg.get("n_training_steps", 1_000_000)
    gamma       = float(full_cfg.get("gamma", 0.99))

    print("=" * 50)
    print(f"  Task       : {params['task']}")
    print(f"  Obs type   : {params['obs_type']}")
    print(f"  Simulator  : {params['simulator']}")
    print(f"  Steps      : {total_steps:,}")
    print(f"  Gamma      : {gamma}")
    print(f"  Seed       : {seed}")
    print("=" * 50)

    # Single vectorised env
    venv = DummyVecEnv([make_env(params, seed)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, gamma=gamma)

    print(f"\n  Obs space  : {venv.observation_space}")
    print(f"  Act space  : {venv.action_space}")

    # PPO — n_epochs=1 means one gradient epoch per rollout (single epoch)
    model = PPO(
        policy="MlpPolicy",
        env=venv,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=1,
        gamma=gamma,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        seed=seed,
        tensorboard_log=os.path.join(_repo_root, "runs"),
    )

    checkpoint_dir = os.path.join(_repo_root, "models", "checkpoints_gcrl")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    checkpoint_cb = CheckpointCallback(
        save_freq=10_000,
        save_path=checkpoint_dir,
        name_prefix="ppo_gcrl",
        save_vecnormalize=True,
    )

    callbacks = [checkpoint_cb]
    if args.render:
        callbacks.append(LiveRenderCallback(params, render_freq=args.render_freq))
        print(f"  Live render : ON (every {args.render_freq:,} steps)\n")
    else:
        print("  Live render : OFF (use --render to enable)\n")

    print("\nStarting training...\n")
    model.learn(
        total_timesteps=total_steps,
        callback=callbacks,
        progress_bar=False,
    )

    model.save(args.out)
    venv.save(args.out + "_vecnormalize.pkl")

    print(f"\nModel saved    : {args.out}.zip")
    print(f"VecNorm saved  : {args.out}_vecnormalize.pkl")
    venv.close()


if __name__ == "__main__":
    main()
