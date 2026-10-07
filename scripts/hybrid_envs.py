"""RL formulations that build on the hand-coded strike controller.

ShotEnv     The policy chooses each shot once (aim angle, outgoing speed, interception x); the
            controller executes it. One RL step per shot; reward 1 for a goal after a paddle hit.
ResidualEnv The policy outputs a correction added to the controller's 20 Hz velocity command.
            Observation = environment observation + the controller's command (2 values).
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box

from scripts.precision_striker_env import PrecisionStrikerEnv
from scripts.strike_planner import StrikePlanner

ANGLE_SPAN = 1.2                  # rad either side of straight at the goal (-x)
SPEED_MID, SPEED_SPAN = 2.4, 0.9  # outgoing puck speed 1.5 .. 3.3 m/s
XPREF_MID, XPREF_SPAN = 0.45, 0.30


def decode_shot(action):
    a = np.clip(np.asarray(action, dtype=float), -1, 1)
    return np.pi + ANGLE_SPAN * a[0], SPEED_MID + SPEED_SPAN * a[1], XPREF_MID + XPREF_SPAN * a[2]


class ShotEnv(gym.Env):
    metadata = {}

    def __init__(self, max_inner_steps=400, planner=None, **env_kwargs):
        self.env = PrecisionStrikerEnv(**env_kwargs)
        self.planner = StrikePlanner(**(planner or {}))
        self.observation_space = self.env.observation_space
        self.action_space = Box(-1.0, 1.0, (3,), dtype=np.float32)
        self.max_inner_steps = max_inner_steps
        self._obs = None

    def reset(self, *, seed=None, options=None):
        self._obs, info = self.env.reset(seed=seed, options=options)
        self.planner.reset()
        return self._obs, info

    def step(self, action):
        self.planner.set_command(*decode_shot(action))
        obs, info, steps = self._obs, {}, 0
        terminated = truncated = False
        while steps < self.max_inner_steps:
            obs, _, terminated, truncated, info = self.env.step(self.planner.act(obs))
            steps += 1
            if terminated or truncated or self.planner.needs_decision:
                break
        self._obs = obs
        reward = 1.0 if (terminated or truncated) and info.get("is_success") else 0.0
        info = {**info, "inner_steps": steps}
        return obs, reward, terminated, truncated, info

    def close(self):
        self.env.close()


class ResidualEnv(gym.Env):
    metadata = {}

    def __init__(self, scale=0.5, planner=None, **env_kwargs):
        self.env = PrecisionStrikerEnv(**env_kwargs)
        self.planner = StrikePlanner(**(planner or {}))
        self.scale = float(scale)
        base = self.env.observation_space
        self.observation_space = Box(np.concatenate([base.low, [-1, -1]]).astype(np.float32),
                                     np.concatenate([base.high, [1, 1]]).astype(np.float32))
        self.action_space = Box(-1.0, 1.0, (2,), dtype=np.float32)
        self._base_action = np.zeros(2, dtype=np.float32)
        self._obs = None

    def _augment(self, obs):
        self._base_action = self.planner.act(obs)
        return np.concatenate([obs, self._base_action]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.planner.reset()
        return self._augment(obs), info

    def step(self, action):
        final = np.clip(self._base_action + self.scale * np.asarray(action, dtype=np.float32), -1, 1)
        obs, reward, terminated, truncated, info = self.env.step(final)
        return self._augment(obs), reward, terminated, truncated, info

    def close(self):
        self.env.close()


def make_env(hybrid, **env_kwargs):
    if hybrid == "shot":
        return ShotEnv(**env_kwargs)
    if hybrid == "residual":
        return ResidualEnv(**env_kwargs)
    return PrecisionStrikerEnv(**env_kwargs)
