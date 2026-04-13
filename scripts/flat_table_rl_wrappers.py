"""
Stable-Baselines3 helpers for AirHockeyFlatTableEnv: domain randomization, PPO reward, curriculum, obs dtype.
"""
from __future__ import annotations

import copy
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np

_repo_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from airhockey.airhockey_simple_tasks import AirHockeyFlatTableEnv


class AirHockeyFlatTableRandEnv(AirHockeyFlatTableEnv):
    """
    Domain randomization: paddle/puck linear damping scaled per episode; slight noise on puck and paddle spawn.
    Damping jitter runs inside create_world_objects (after rng is set in reset).
    """

    def __init__(
        self,
        puck_position_noise: float = 0.012,
        paddle_position_noise: float = 0.035,
        damping_noise_frac: float = 0.12,
        **kwargs: Any,
    ) -> None:
        # Base __init__ calls reset() → create_world_objects; set fields first.
        self.puck_position_noise = float(puck_position_noise)
        self.paddle_position_noise = float(paddle_position_noise)
        self.damping_noise_frac = float(damping_noise_frac)
        sp = kwargs.get("simulator_params", {})
        if isinstance(sp, dict):
            self._nominal_paddle_damping = float(sp.get("paddle_damping", 3.0))
            self._nominal_puck_damping = float(sp.get("puck_damping", 0.5))
        else:
            self._nominal_paddle_damping = float(getattr(sp, "paddle_damping", 3.0))
            self._nominal_puck_damping = float(getattr(sp, "puck_damping", 0.5))
        super().__init__(**kwargs)
        self._nominal_paddle_damping = float(self.simulator.paddle_damping)
        self._nominal_puck_damping = float(self.simulator.puck_damping)

    @staticmethod
    def from_dict(state_dict: Dict[str, Any]) -> "AirHockeyFlatTableRandEnv":
        return AirHockeyFlatTableRandEnv(**state_dict)

    def create_world_objects(self) -> None:
        f = self.damping_noise_frac
        self.simulator.paddle_damping = self._nominal_paddle_damping * (1.0 + self.rng.uniform(-f, f))
        self.simulator.puck_damping = self._nominal_puck_damping * (1.0 + self.rng.uniform(-f, f))
        super().create_world_objects()

    def get_puck_configuration(self, bad_regions: Any = None) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        pos, vel = super().get_puck_configuration(bad_regions)
        n = self.puck_position_noise
        px = pos[0] + float(self.rng.uniform(-n, n))
        py = pos[1] + float(self.rng.uniform(-n, n))
        m = self.puck_radius + 0.02
        px = float(np.clip(px, self.table_x_top + m, self.table_x_bot - m))
        py = float(np.clip(py, self.table_y_left + m, self.table_y_right - m))
        return (px, py), vel

    def get_paddle_configuration(self, name: str) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        pos, vel = super().get_paddle_configuration(name)
        if name != "paddle_ego":
            return pos, vel
        n = self.paddle_position_noise
        px = pos[0] + float(self.rng.uniform(-n, n))
        py = pos[1] + float(self.rng.uniform(-n, n))
        px = float(np.clip(px, self.paddle_x_min + self.paddle_radius, self.paddle_x_max - self.paddle_radius))
        py = float(np.clip(py, self.paddle_y_min + self.paddle_radius, self.paddle_y_max - self.paddle_radius))
        return (px, py), vel


class ObsFloat32Wrapper(gym.ObservationWrapper):
    """SB3-friendly float32 observations."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        obs_sp = env.observation_space
        self.observation_space = gym.spaces.Box(
            low=obs_sp.low.astype(np.float32),
            high=obs_sp.high.astype(np.float32),
            dtype=np.float32,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return np.asarray(obs, dtype=np.float32)


class ActionClipWrapper(gym.ActionWrapper):
    """Clip continuous actions to the environment box."""

    def action(self, act: np.ndarray) -> np.ndarray:
        a = np.asarray(act, dtype=np.float64)
        return np.clip(a, self.env.action_space.low, self.env.action_space.high).astype(np.float64)


class FlatTablePPORewardWrapper(gym.Wrapper):
    """
    Dense shaping for PPO. Goal semantics match airhockey_base.has_finished:
    puck_within_ego_goal = you scored (puck into opponent / far -x side)
    puck_within_alt_goal = conceded (puck into your / far +x side)

    Hit bonus uses paddle_puck_collision_count delta; fallback uses puck speed delta near paddle.
    """

    def __init__(
        self,
        env: gym.Env,
        goal_score_reward: float = 100.0,
        goal_concede_penalty: float = 100.0,
        hit_bonus: float = 5.0,
        toward_goal_vel_scale: float = 0.1,
        time_penalty: float = 0.01,
        hit_speed_jump_thresh: float = 0.35,
        hit_proximity: float = 0.09,
    ) -> None:
        super().__init__(env)
        u = env.unwrapped
        self.paddle_radius = float(getattr(u, "paddle_radius", 0.0508))
        self.puck_radius = float(getattr(u, "puck_radius", 0.03175))
        self.goal_score_reward = goal_score_reward
        self.goal_concede_penalty = goal_concede_penalty
        self.hit_bonus = hit_bonus
        self.toward_goal_vel_scale = toward_goal_vel_scale
        self.time_penalty = time_penalty
        self.hit_speed_jump_thresh = hit_speed_jump_thresh
        self.hit_proximity = hit_proximity
        self._prev_coll_count = 0
        self._prev_puck_speed = 0.0

    def reset(self, **kwargs: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
        self._prev_coll_count = 0
        self._prev_puck_speed = 0.0
        return self.env.reset(**kwargs)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, _, terminated, truncated, info = self.env.step(action)
        base = info.get("paddle_puck_collision_count", 0)
        n = int(base) if base is not None else 0
        coll_hit = n > self._prev_coll_count
        self._prev_coll_count = n

        puck_vx = float(obs[6]) if obs.shape[0] >= 8 else 0.0
        puck_vy = float(obs[7]) if obs.shape[0] >= 8 else 0.0
        px = float(obs[4]) if obs.shape[0] >= 8 else 0.0
        py = float(obs[5]) if obs.shape[0] >= 8 else 0.0
        ppx = float(obs[0])
        ppy = float(obs[1])
        dist = float(np.hypot(px - ppx, py - ppy))
        speed = float(np.hypot(puck_vx, puck_vy))
        speed_jump = speed - self._prev_puck_speed
        self._prev_puck_speed = speed
        approx_hit = (not coll_hit) and dist < self.paddle_radius + self.puck_radius + self.hit_proximity and speed_jump > self.hit_speed_jump_thresh
        hit = coll_hit or approx_hit

        toward = max(0.0, -puck_vx)
        r = self.toward_goal_vel_scale * toward
        if hit:
            r += self.hit_bonus
        r -= self.time_penalty
        if info.get("puck_within_ego_goal"):
            r += self.goal_score_reward
        if info.get("puck_within_alt_goal"):
            r -= self.goal_concede_penalty

        return obs, float(r), terminated, truncated, info


class CurriculumUserBlocksWrapper(gym.Wrapper):
    """Phase 1: train with empty user blocks; phase 2: restore saved block list."""

    def __init__(self, env: gym.Env, blocks_enabled_after_steps: int) -> None:
        super().__init__(env)
        self.blocks_enabled_after_steps = int(blocks_enabled_after_steps)
        self._env_steps = 0
        e = env.unwrapped
        self._saved_blocks: List[Tuple[float, float]] = copy.deepcopy(
            list(getattr(e, "user_block_positions", []))
        )

    def reset(self, **kwargs: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
        e = self.env.unwrapped
        if self._env_steps < self.blocks_enabled_after_steps and hasattr(e, "user_block_positions"):
            e.user_block_positions = []
        obs, info = self.env.reset(**kwargs)
        if hasattr(e, "user_block_positions"):
            e.user_block_positions = list(self._saved_blocks)
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self._env_steps += 1
        return self.env.step(action)


def load_flat_table_config(cfg_path: str) -> Dict[str, Any]:
    import yaml

    with open(cfg_path, "r") as f:
        full = yaml.safe_load(f)
    params = copy.deepcopy(full["air_hockey"])
    params["n_training_steps"] = full.get("n_training_steps", 1)
    params["seed"] = full.get("seed", 0)
    params["return_goal_obs"] = False
    params["use_reward_shaping"] = False
    params["task"] = "flat_table"
    return params


def build_flat_table_train_env(
    cfg_path: str,
    seed: int = 0,
    curriculum_steps: int = 0,
    user_blocks: Optional[Sequence[Tuple[float, float]]] = None,
) -> gym.Env:
    base = load_flat_table_config(cfg_path)
    base["seed"] = int(seed)
    env = AirHockeyFlatTableRandEnv.from_dict(base)
    if user_blocks:
        env.user_block_positions = copy.deepcopy(list(user_blocks))
    if curriculum_steps > 0:
        env = CurriculumUserBlocksWrapper(env, curriculum_steps)
    env = FlatTablePPORewardWrapper(env)
    env = ActionClipWrapper(env)
    env = ObsFloat32Wrapper(env)
    return env


def parse_user_blocks_arg(s: Optional[str]) -> List[Tuple[float, float]]:
    if not s:
        return []
    out: List[Tuple[float, float]] = []
    for part in s.replace(";", " ").split():
        part = part.strip()
        if not part:
            continue
        xy = part.split(",")
        if len(xy) != 2:
            continue
        out.append((float(xy[0]), float(xy[1])))
    return out
