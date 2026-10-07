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
from scripts import scoring_geometry as sg
from scripts.scoring_geometry import scorable

ROOT = Path(__file__).resolve().parents[1]
TASK_VERSION = "precision-striker-v2-incoming"
# v3 changes only the reward/success rule (goal requires a paddle hit, configurable timeout cost).
# Observations, actions and physics are identical to v2.
TASK_VERSION_V3 = "precision-striker-v3-hit-required"
# v4: obstacles act as walls. A puck-obstacle contact neither ends the episode nor is penalized;
# success is a goal (after a paddle hit when require_hit). Same observations/actions/physics.
TASK_VERSION_V4 = "precision-striker-v4-obstacle-bounce"
# v5: v4 plus two observation features the reward depends on: [paddle has hit, fraction of time used].
# Observation length 22 instead of 20, so v5 checkpoints are not interchangeable with v2-v4.
TASK_VERSION_V5 = "precision-striker-v5-bounce-extra-obs"
MAX_OBSTACLES = 3


def task_version(require_hit=False, timeout_penalty=2.0, obstacle_bounce=False,
                 extra_obs=False, goal_shaping=False):
    if extra_obs:
        return TASK_VERSION_V5
    if obstacle_bounce:
        return TASK_VERSION_V4
    return TASK_VERSION if not require_hit and timeout_penalty == 2.0 and not goal_shaping else TASK_VERSION_V3


class PrecisionStrikerEnv(AirHockeyFlatTableEnv):
    def __init__(self, seed=0, obstacles=1, max_steps=200, randomize=True,
                 speed_min=0.3, speed_max=1.5, require_hit=False, timeout_penalty=2.0,
                 obstacle_bounce=False, extra_obs=False, goal_shaping=False):
        if not 0 <= obstacles <= MAX_OBSTACLES or max_steps < 1:
            raise ValueError("Require 0..3 obstacles and positive max_steps")
        if timeout_penalty < 0:
            raise ValueError("timeout_penalty is a cost and must be >= 0")
        self.require_hit = bool(require_hit)
        self.timeout_penalty = float(timeout_penalty)
        self.obstacle_bounce = bool(obstacle_bounce)
        self.extra_obs = bool(extra_obs)
        # After the hit, reward progress toward the goal mouth instead of plain -x progress.
        self.goal_shaping = bool(goal_shaping)
        self.task_version = task_version(self.require_hit, self.timeout_penalty, self.obstacle_bounce,
                                         self.extra_obs, self.goal_shaping)
        self.obstacle_count = int(obstacles)
        if not 0 < speed_min <= speed_max <= 2.0:
            raise ValueError("Require 0 < speed_min <= speed_max <= 2 m/s")
        self.speed_min, self.speed_max = float(speed_min), float(speed_max)
        self.control_dt, self.physics_dt = 0.05, 0.005
        self.acceleration_limit = 10.0  # Prototype assumption; calibrate before hardware transfer.
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
        geometry = (self.length, self.width, self.simulator.goal_width, self.puck_radius, self.block_width)
        if not np.allclose(geometry, (sg.LENGTH, sg.WIDTH, sg.GOAL_WIDTH, sg.PUCK_R, sg.BLOCK_W)):
            raise ValueError("scripts/scoring_geometry.py constants do not match the simulated table")

    def initialize_spaces(self, obs_type):
        super().initialize_spaces(obs_type)
        # 8 paddle/puck state values + 3 fixed slots [x, y, width, present].
        low = list(self.observation_space.low) + [-1, -0.5, 0, 0] * MAX_OBSTACLES
        high = list(self.observation_space.high) + [1, 0.5, 0.2, 1] * MAX_OBSTACLES
        # Terminal puck positions may be just outside the table.
        low[4:6], high[4:6] = [-2, -1], [2, 1]
        if self.extra_obs:
            low, high = low + [0, 0], high + [1, 1]
        self.observation_space = self.single_observation_space = Box(
            np.array(low, dtype=np.float32), np.array(high, dtype=np.float32))

    def get_observation(self, state_info, obs_type="vel", **kwargs):
        state = super().get_observation(state_info, obs_type="vel", **kwargs)
        slots = np.zeros((MAX_OBSTACLES, 4), dtype=np.float32)
        for i, pos in enumerate(self.user_block_positions):
            slots[i] = (*pos, self.block_width, 1)
        parts = [state, slots.ravel()]
        if self.extra_obs:
            elapsed = min(1.0, getattr(self, "_steps", 0) / self.max_timesteps)
            parts.append(np.array([float(getattr(self, "_hit", False)), elapsed], dtype=np.float32))
        return np.concatenate(parts).astype(np.float32)

    def get_puck_configuration(self, bad_regions=None):
        return self._spawn_puck, self._spawn_velocity

    def get_paddle_configuration(self, name):
        return self._spawn_paddle, (0.0, 0.0)

    def sample_layout(self, seed):
        rng = np.random.RandomState(seed)
        positions = []
        for _ in range(self.obstacle_count):
            for _attempt in range(100):
                pos = (float(rng.uniform(-0.65, -0.25)), float(rng.uniform(-0.28, 0.28)))
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
        self._previous_distance = self._intercept_distance(obs)
        info.update(task_version=self.task_version, blocks=list(self.user_block_positions),
                    initial_velocity=list(self._spawn_velocity), scorable=self._scorable)
        return obs, info

    def create_world_objects(self):
        opts = self._options
        puck = np.asarray(opts.get("puck_position", [self.rng.uniform(-0.12, 0.02),
                                                     self.rng.uniform(-0.23, 0.23)]), dtype=float)
        speed = self.rng.uniform(self.speed_min, self.speed_max)
        angle = self.rng.uniform(-0.25, 0.25)
        velocity = np.asarray(opts.get("puck_velocity", [speed * np.cos(angle), speed * np.sin(angle)]), dtype=float)
        if (puck.shape != (2,) or not np.isfinite(puck).all()
                or not (-0.15 <= puck[0] <= 0.60 and abs(puck[1]) <= 0.25)):
            raise ValueError("puck_position requires x in [-0.15,0.60], y in [-0.25,0.25]")
        if velocity.shape != (2,) or not np.isfinite(velocity).all() or np.linalg.norm(velocity) > 2:
            raise ValueError("puck_velocity must be a finite 2D vector with speed <= 2 m/s")
        self._spawn_puck, self._spawn_velocity = tuple(puck), tuple(velocity)
        self._spawn_paddle = (0.67, float(self.rng.uniform(-0.08, 0.08)))
        blocks = opts.get("blocks")
        if blocks is None:
            blocks = self.sample_layout(int(self.rng.randint(0, 2**31 - 1)))
            # Guarantee a geometrically scorable shot. An audit of 12,000 random episodes (1-3
            # obstacles) found none unscorable, so this redraw is a safeguard that normally never runs.
            for _ in range(50):
                if scorable(puck, velocity, blocks):
                    break
                blocks = self.sample_layout(int(self.rng.randint(0, 2**31 - 1)))
            else:
                raise RuntimeError("Could not sample a scorable obstacle layout")
        self._scorable = bool(scorable(puck, velocity, blocks))
        if len(blocks) > MAX_OBSTACLES:
            raise ValueError("At most 3 obstacles supported")
        checked = []
        for block in blocks:
            p = np.asarray(block, dtype=float)
            if (p.shape != (2,) or not np.isfinite(p).all()
                    or not (-0.8 <= p[0] <= -0.22 and abs(p[1]) <= 0.32)):
                raise ValueError("Blocks require opponent-half x in [-0.8,-0.22], y in [-0.32,0.32]")
            if any(np.max(np.abs(p - np.asarray(other))) < self.block_width + 0.02 for other in checked):
                raise ValueError("Obstacles must not overlap")
            checked.append(tuple(p))
        self.user_block_positions = sorted(checked)
        self.simulator.puck_damping = 0.15 * (self.rng.uniform(0.8, 1.2) if self.randomize else 1)
        super().create_world_objects()
        # Continuous collision detection for fast shots against narrow obstacles.
        self.simulator.pucks["puck_0"].bullet = True

    def _crossed_mouth(self, p0, p1, line):
        """True if the puck centre crossed x = line between samples within the goal opening."""
        d0, d1 = float(p0[0]) - line, float(p1[0]) - line
        if d0 == d1 or d0 * d1 > 0:
            return False
        y = float(p0[1]) + d0 / (d0 - d1) * (float(p1[1]) - float(p0[1]))
        return abs(y) <= self.simulator.goal_width / 2

    def _intercept_distance(self, obs):
        # Dense training feedback only: this target is never fed to the action controller.
        target = obs[4:6] + np.array([self.paddle_radius + self.puck_radius, 0])
        target[0] = np.clip(target[0], 0.16, 0.70)
        return float(np.linalg.norm(obs[:2] - target))

    def _advance_physics(self, action):
        """Velocity servo at 200 Hz. Policy decisions remain at 20 Hz.

        Applies acceleration/speed limits and projects onto the robot workspace.
        No interception, shot planning or obstacle avoidance is coded into this servo.
        """
        sim = self.simulator
        paddle = sim.paddles["paddle_ego"]
        target = np.clip(action, -1, 1).astype(float)
        target *= self.max_paddle_vel / max(1.0, float(np.linalg.norm(target)))
        self.last_command = target.copy()
        target_box = np.asarray(sim.base_coord_to_box2d(target))
        clipped = False
        for _ in range(round(self.control_dt / self.physics_dt)):
            velocity = np.array(paddle.linearVelocity, dtype=float)
            delta = target_box - velocity
            limit = self.acceleration_limit * self.physics_dt
            delta *= min(1.0, limit / (np.linalg.norm(delta) + 1e-12))
            paddle.linearVelocity = tuple(velocity + delta)
            sim.world.Step(self.physics_dt, 8, 3)
            actual_velocity = np.array(paddle.linearVelocity)
            actual_velocity *= min(1.0, self.max_paddle_vel / (np.linalg.norm(actual_velocity) + 1e-12))
            paddle.linearVelocity = tuple(actual_velocity)
            pos = np.asarray(sim._box2d_to_base_coords(paddle.position))
            bounded = np.clip(pos, [self.paddle_radius, -0.35], [0.88, 0.35])
            if not np.allclose(pos, bounded, atol=1e-7, rtol=0):
                clipped = True
                v = np.asarray(sim._box2d_to_base_coords(paddle.linearVelocity))
                v[pos != bounded] = 0
                paddle.position = sim.base_coord_to_box2d(bounded)
                paddle.linearVelocity = sim.base_coord_to_box2d(v)
        self.current_state = sim.get_current_state()
        return self.get_observation(self.current_state), clipped

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError("Action must be a finite 2D vector")
        previous = self.get_observation(self.current_state)
        obs, workspace_clip = self._advance_physics(action)
        info = {}
        self._steps += 1
        events = self.simulator.collision_listener.collision_forces
        hit = any({str(e["bodyA"]), str(e["bodyB"])} == {"paddle_ego", "puck_0"} for e in events)
        collision = any(any(str(e[k]).startswith("user_block_") for k in ("bodyA", "bodyB"))
                        and any(str(e[k]) in ("puck_0", "paddle_ego") for k in ("bodyA", "bodyB"))
                        for e in events)
        events.clear()  # The legacy listener otherwise retains a whole episode's contacts.
        reward = -0.005 + (1.0 if hit and not self._hit else 0)
        # Progress terms telescope; waiting far from the puck does not earn idle reward.
        distance = self._intercept_distance(obs)
        if not self._hit:
            reward += 2.0 * (self._previous_distance - distance)
        if self._hit or hit:
            if self.goal_shaping:
                goal = np.array([self.table_x_top, 0.0])
                reward += 2.0 * float(np.linalg.norm(previous[4:6] - goal) - np.linalg.norm(obs[4:6] - goal))
            else:
                reward += 2.0 * float(previous[4] - obs[4])
        self._previous_distance = distance
        reward -= 0.05 * workspace_clip
        self._hit |= hit
        self._collision |= collision
        if self.extra_obs:  # refresh after this step's contact and clock updates
            obs = obs.copy()
            obs[-2:] = (float(self._hit), min(1.0, self._steps / self.max_timesteps))
        x, y = obs[4:6]
        in_goal_lane = abs(float(y)) + self.puck_radius <= self.simulator.goal_width / 2
        score = bool(x < self.table_x_top - self.puck_radius and in_goal_lane)
        conceded = bool(x > self.table_x_bot + self.puck_radius and in_goal_lane)
        if self.obstacle_bounce:
            # v4+: judge the goal where the puck crossed the end line. A fast diagonal puck can cross
            # inside the mouth and drift sideways behind the line before the next 20 Hz sample.
            score = score or self._crossed_mouth(previous[4:6], obs[4:6], self.table_x_top)
            conceded = conceded or self._crossed_mouth(previous[4:6], obs[4:6], self.table_x_bot)
        workspace = bool(obs[0] < self.paddle_radius - 1e-4 or obs[0] > 0.8801 or abs(obs[1]) > 0.3501)
        escaped = bool(abs(x) > self.length / 2 + 0.2 or abs(y) > self.width / 2 + 0.1)
        # v4: the obstacle is just another wall for the puck.
        ends_on_collision = collision and not self.obstacle_bounce
        terminated = bool(score or conceded or ends_on_collision or workspace or escaped)
        truncated = bool(self._steps >= self.max_timesteps and not terminated)
        # v3: a goal without any paddle contact ends the episode but earns nothing.
        credited = score and (self._hit or not self.require_hit)
        reward += (20 * credited - 20 * ends_on_collision - 10 * conceded - 10 * workspace
                   - self.timeout_penalty * truncated)
        success = credited and (self.obstacle_bounce or not self._collision) and not workspace
        outcome = ("obstacle_collision" if ends_on_collision else "workspace_violation" if workspace else
                   "goal" if score else "conceded" if conceded else "escaped" if escaped else
                   "timeout" if truncated else "running")
        info.update(is_success=bool(success), success=bool(success), outcome=outcome,
                    obstacle_collision=bool(self._collision), hit=bool(self._hit),
                    workspace_clip=bool(workspace_clip), command=self.last_command.tolist(),
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
