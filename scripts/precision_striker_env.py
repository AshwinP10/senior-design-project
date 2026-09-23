"""Versioned, state-based scoring task using the repository's Box2D physics.

Coordinates are metres: +x is the robot half, -x is the scoring end.
This is a planar controller prototype, not a validated UR5e dynamics model.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import yaml
from gymnasium.spaces import Box

from airhockey.airhockey_simple_tasks import AirHockeyFlatTableEnv

ROOT = Path(__file__).resolve().parents[1]
TASK_VERSION = "precision-striker-v1"
MAX_OBSTACLES = 3


class PrecisionStrikerEnv(AirHockeyFlatTableEnv):
    def __init__(self, seed=0, obstacles=1, max_steps=200, randomize=True):
        if not 0 <= obstacles <= MAX_OBSTACLES or max_steps < 1:
            raise ValueError("Require 0..3 obstacles and positive max_steps")
        self.obstacle_count = int(obstacles)
        self.randomize = bool(randomize)
        self._options = {}
        self.user_block_positions = []
        self._spawn_puck = (0.4, 0.0)
        self._spawn_velocity = (0.0, 0.0)
        self._spawn_paddle = (0.6, 0.0)
        with (ROOT / "configs/flat_table_play.yaml").open() as stream:
            params = copy.deepcopy(yaml.safe_load(stream)["air_hockey"])
        params.update(seed=int(seed), n_training_steps=1, return_goal_obs=False,
                      use_reward_shaping=False, max_timesteps=int(max_steps),
                      terminate_on_out_of_bounds=False,
                      paddle_bounds=[0.0508, 0.88, -0.35, 0.35])
        params["simulator_params"].update(goal_width=0.24, gravity=0, max_paddle_vel=2.0)
        super().__init__(**params)

    def initialize_spaces(self, obs_type):
        super().initialize_spaces(obs_type)
        # 8 paddle/puck state values + 3 fixed slots [x, y, width, present].
        low = list(self.observation_space.low) + [-1, -0.5, 0, 0] * MAX_OBSTACLES
        high = list(self.observation_space.high) + [1, 0.5, 0.2, 1] * MAX_OBSTACLES
        # Terminal puck positions may be just outside the table.
        low[4:6], high[4:6] = [-2, -1], [2, 1]
        self.observation_space = self.single_observation_space = Box(
            np.array(low, dtype=np.float32), np.array(high, dtype=np.float32))

    def get_observation(self, state_info, obs_type="vel", **kwargs):
        state = super().get_observation(state_info, obs_type="vel", **kwargs)
        slots = np.zeros((MAX_OBSTACLES, 4), dtype=np.float32)
        for i, pos in enumerate(self.user_block_positions):
            slots[i] = (*pos, self.block_width, 1)
        return np.concatenate([state, slots.ravel()]).astype(np.float32)

    def get_puck_configuration(self, bad_regions=None):
        return self._spawn_puck, self._spawn_velocity

    def get_paddle_configuration(self, name):
        return self._spawn_paddle, (0.0, 0.0)

    def sample_layout(self, seed):
        rng = np.random.RandomState(seed)
        positions = []
        for _ in range(self.obstacle_count):
            for _attempt in range(100):
                pos = (float(rng.uniform(-0.65, -0.15)), float(rng.uniform(-0.28, 0.28)))
                if all(np.linalg.norm(np.subtract(pos, other)) > 0.13 for other in positions):
                    positions.append(pos)
                    break
            else:
                raise RuntimeError("Could not sample separated obstacles")
        return sorted(positions)

    def reset(self, *, seed=None, options=None, **kwargs):
        self._options = options or {}
        self._hit = False
        self._collision = False
        self._steps = 0
        obs, info = super().reset(seed=seed, **kwargs)
        self._previous_potential = self._potential(obs)
        info.update(task_version=TASK_VERSION, blocks=list(self.user_block_positions))
        return obs, info

    def create_world_objects(self):
        opts = self._options
        puck = np.asarray(opts.get("puck_position", [self.rng.uniform(0.23, 0.53),
                                                     self.rng.uniform(-0.23, 0.23)]), dtype=float)
        velocity = np.asarray(opts.get("puck_velocity", [0, 0]), dtype=float)
        if (puck.shape != (2,) or not np.isfinite(puck).all()
                or not (0.18 <= puck[0] <= 0.60 and abs(puck[1]) <= 0.25)):
            raise ValueError("puck_position requires reachable x in [0.18,0.60], y in [-0.25,0.25]")
        if velocity.shape != (2,) or not np.isfinite(velocity).all() or np.linalg.norm(velocity) > 2:
            raise ValueError("puck_velocity must be a finite 2D vector with speed <= 2 m/s")
        self._spawn_puck, self._spawn_velocity = tuple(puck), tuple(velocity)
        self._spawn_paddle = (float(puck[0] + 0.18),
                              float(np.clip(puck[1] + self.rng.uniform(-0.08, 0.08), -0.29, 0.29)))
        blocks = opts.get("blocks")
        if blocks is None:
            blocks = self.sample_layout(int(self.rng.randint(0, 2**31 - 1)))
        if len(blocks) > MAX_OBSTACLES:
            raise ValueError("At most 3 obstacles supported")
        checked = []
        for block in blocks:
            p = np.asarray(block, dtype=float)
            if (p.shape != (2,) or not np.isfinite(p).all()
                    or not (-0.8 <= p[0] <= -0.1 and abs(p[1]) <= 0.32)):
                raise ValueError("Blocks require opponent-half x in [-0.8,-0.1], y in [-0.32,0.32]")
            if any(np.max(np.abs(p - np.asarray(other))) < self.block_width + 0.02 for other in checked):
                raise ValueError("Obstacles must not overlap")
            checked.append(tuple(p))
        self.user_block_positions = sorted(checked)
        self.simulator.puck_damping = 0.5 * (self.rng.uniform(0.88, 1.12) if self.randomize else 1)
        super().create_world_objects()
        # Continuous collision detection for fast shots against narrow obstacles.
        self.simulator.pucks["puck_0"].bullet = True

    def _potential(self, obs):
        target = obs[4:6] + np.array([self.paddle_radius + self.puck_radius + 0.015, 0])
        return -2.0 * float(np.linalg.norm(obs[:2] - target)) if not self._hit else 0.0

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError("Action must be a finite 2D vector")
        # Use the base transition directly to retain correct Gymnasium timeout semantics.
        obs, _, _, _, info = self.single_agent_step(np.clip(action, -1, 1))
        self._steps += 1
        events = self.simulator.collision_listener.collision_forces
        hit = any({str(e["bodyA"]), str(e["bodyB"])} == {"paddle_ego", "puck_0"} for e in events)
        collision = any(any(str(e[k]).startswith("user_block_") for k in ("bodyA", "bodyB"))
                        and any(str(e[k]) in ("puck_0", "paddle_ego") for k in ("bodyA", "bodyB"))
                        for e in events)
        events.clear()  # The legacy listener otherwise retains a whole episode's contacts.
        reward = -0.01 + (5.0 if hit and not self._hit else 0)
        self._hit |= hit
        self._collision |= collision
        x, y = obs[4:6]
        in_goal_lane = abs(float(y)) + self.puck_radius <= self.simulator.goal_width / 2
        score = bool(x < self.table_x_top - self.puck_radius and in_goal_lane)
        conceded = bool(x > self.table_x_bot + self.puck_radius and in_goal_lane)
        workspace = bool(obs[0] < self.paddle_radius - 1e-4 or obs[0] > 0.88 or abs(obs[1]) > 0.35)
        escaped = bool(abs(x) > self.length / 2 + 0.2 or abs(y) > self.width / 2 + 0.1)
        terminated = bool(score or conceded or collision or workspace or escaped)
        truncated = bool(self._steps >= self.max_timesteps and not terminated)
        potential = 0.0 if terminated else self._potential(obs)
        reward += 0.99 * potential - self._previous_potential
        self._previous_potential = potential
        reward += 100 * score - 100 * collision - 50 * conceded - 20 * workspace
        success = score and not self._collision and not workspace
        outcome = ("obstacle_collision" if collision else "workspace_violation" if workspace else
                   "goal" if score else "conceded" if conceded else "escaped" if escaped else
                   "timeout" if truncated else "running")
        info.update(is_success=bool(success), success=bool(success), outcome=outcome,
                    obstacle_collision=bool(self._collision), hit=bool(self._hit),
                    puck_within_ego_goal=score, puck_within_alt_goal=conceded)
        return obs, float(reward), terminated, truncated, info

    def terminal_frame(self, columns=31, rows=17):
        grid = [[" " for _ in range(columns)] for _ in range(rows)]
        def place(x, y, mark):
            row = int(np.clip((x / self.length + 0.5) * (rows - 1), 0, rows - 1))
            col = int(np.clip((y / self.width + 0.5) * (columns - 1), 0, columns - 1))
            grid[row][col] = mark
        for pos in self.user_block_positions:
            place(*pos, "#")
        place(*self.current_state["pucks"][0]["position"], "o")
        place(*self.current_state["paddles"]["paddle_ego"]["position"], "P")
        return "       opponent goal (-x)\n+" + "-" * columns + "+\n" + "\n".join(
            "|" + "".join(row) + "|" for row in grid) + "\n+" + "-" * columns + "+\nP=paddle o=puck #=obstacle"
