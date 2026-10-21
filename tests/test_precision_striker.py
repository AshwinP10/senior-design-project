import unittest

import numpy as np

from scripts.precision_striker_env import PrecisionStrikerEnv


class PrecisionStrikerTests(unittest.TestCase):
    def setUp(self):
        self.env = PrecisionStrikerEnv(obstacles=1, randomize=False)

    def tearDown(self):
        self.env.close()

    def launch(self, pos, velocity):
        puck = self.env.simulator.pucks["puck_0"]
        puck.position = self.env.simulator.base_coord_to_box2d(pos)
        puck.linearVelocity = self.env.simulator.base_coord_to_box2d(velocity)

    def test_seed_and_obstacle_observability(self):
        a, info = self.env.reset(seed=123)
        b, second_info = self.env.reset(seed=123)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(info["blocks"], second_info["blocks"])
        self.assertTrue(self.env.observation_space.contains(a))
        self.assertEqual(a[11], 1)
        self.assertTrue(np.all(a[12:] == 0))
        c, _ = self.env.reset(seed=123, options={"blocks": [(-0.7, 0.2)]})
        self.assertFalse(np.array_equal(a[8:], c[8:]))

    def test_goal_is_physically_reachable(self):
        self.env.reset(seed=1, options={"blocks": []})
        self.launch((-0.75, 0), (-2, 0))
        for _ in range(15):
            _, reward, terminated, _, info = self.env.step(np.zeros(2))
            if terminated:
                break
        self.assertTrue(terminated)
        self.assertEqual(info["outcome"], "goal")
        self.assertTrue(info["is_success"])
        self.assertGreater(reward, 18)

    def test_end_rail_outside_goal_bounces(self):
        self.env.reset(seed=1, options={"blocks": []})
        self.launch((-0.75, 0.3), (-2, 0))
        for _ in range(10):
            obs, _, _, _, info = self.env.step(np.zeros(2))
            self.assertFalse(info["is_success"])
        self.assertGreater(obs[6], 0)

    def test_puck_crosses_centerline_without_reset_in_both_directions(self):
        for start, speed in [(-0.12, 1.0), (0.12, -1.0)]:
            with self.subTest(start=start, speed=speed):
                obs, _ = self.env.reset(seed=7, options={
                    "puck_position": [start, 0.2], "puck_velocity": [speed, 0], "blocks": []})
                puck = self.env.simulator.pucks["puck_0"]
                previous_x = obs[4]
                for _ in range(8):
                    obs, _, terminated, truncated, info = self.env.step(np.zeros(2))
                    self.assertFalse(terminated)
                    self.assertFalse(truncated)
                    self.assertEqual(info["outcome"], "running")
                    self.assertIs(self.env.simulator.pucks["puck_0"], puck)
                    self.assertGreater((obs[4] - previous_x) * speed, 0)
                    previous_x = obs[4]
                self.assertGreater(obs[4] * speed, 0.2)
                self.assertEqual(self.env._steps, 8)

    def test_collision_is_failure(self):
        self.env.reset(seed=1, options={"blocks": [(-0.4, 0)]})
        self.launch((-0.2, 0), (-2, 0))
        for _ in range(10):
            _, reward, terminated, _, info = self.env.step(np.zeros(2))
            if terminated:
                break
        self.assertEqual(info["outcome"], "obstacle_collision")
        self.assertTrue(terminated)
        self.assertFalse(info["is_success"])
        self.assertLess(reward, -18)

    def test_incoming_speed_range_and_servo_limits(self):
        for seed in range(20):
            obs, _ = self.env.reset(seed=seed)
            self.assertGreater(obs[6], 0)
            self.assertGreaterEqual(np.linalg.norm(obs[6:8]), 0.3 - 1e-6)
            self.assertLessEqual(np.linalg.norm(obs[6:8]), 1.5 + 1e-6)
        for _ in range(100):
            obs, _, terminated, truncated, _ = self.env.step(np.array([-1.0, 1.0]))
            self.assertGreaterEqual(obs[0], self.env.paddle_radius - 1e-6)
            self.assertLessEqual(abs(obs[1]), 0.3501)
            self.assertLessEqual(np.linalg.norm(obs[2:4]), 2.0001)
            if terminated or truncated:
                self.env.reset(seed=20)

    def test_timeout_is_not_terminal(self):
        env = PrecisionStrikerEnv(obstacles=0, max_steps=2)
        try:
            env.reset(seed=2)
            env.step(np.zeros(2))
            _, _, terminated, truncated, info = env.step(np.zeros(2))
            self.assertFalse(terminated)
            self.assertTrue(truncated)
            self.assertEqual(info["outcome"], "timeout")
        finally:
            env.close()

    def test_v3_passive_goal_earns_nothing_and_timeout_cost(self):
        env = PrecisionStrikerEnv(obstacles=1, randomize=False, require_hit=True, timeout_penalty=10)
        try:
            self.assertEqual(env.task_version, "precision-striker-v3-hit-required")
            env.reset(seed=1, options={"blocks": []})
            puck = env.simulator.pucks["puck_0"]
            puck.position = env.simulator.base_coord_to_box2d((-0.75, 0))
            puck.linearVelocity = env.simulator.base_coord_to_box2d((-2, 0))
            for _ in range(15):
                _, reward, terminated, _, info = env.step(np.zeros(2))
                if terminated:
                    break
            self.assertEqual(info["outcome"], "goal")
            self.assertFalse(info["hit"])
            self.assertFalse(info["is_success"])
            self.assertLess(reward, 1)
        finally:
            env.close()
        env = PrecisionStrikerEnv(obstacles=0, max_steps=1, require_hit=True, timeout_penalty=10)
        try:
            env.reset(seed=2)
            _, reward, _, truncated, _ = env.step(np.zeros(2))
            self.assertTrue(truncated)
            self.assertLess(reward, -9)
        finally:
            env.close()

    def test_v4_obstacle_bounce_is_not_terminal(self):
        env = PrecisionStrikerEnv(obstacles=1, randomize=False, require_hit=True, obstacle_bounce=True)
        try:
            self.assertEqual(env.task_version, "precision-striker-v4-obstacle-bounce")
            env.reset(seed=1, options={"blocks": [(-0.4, 0)]})
            puck = env.simulator.pucks["puck_0"]
            puck.position = env.simulator.base_coord_to_box2d((-0.2, 0))
            puck.linearVelocity = env.simulator.base_coord_to_box2d((-2, 0))
            touched = False
            for _ in range(10):
                obs, reward, terminated, _, info = env.step(np.zeros(2))
                touched |= info["obstacle_collision"]
                self.assertFalse(terminated)
                self.assertGreater(reward, -5)
            self.assertTrue(touched)
            self.assertGreater(obs[6], 0)  # bounced back toward the robot
        finally:
            env.close()

    def test_v5_extra_obs_track_hit_and_time(self):
        env = PrecisionStrikerEnv(obstacles=1, randomize=False, require_hit=True, obstacle_bounce=True,
                                  extra_obs=True, goal_shaping=True, max_steps=40)
        try:
            self.assertEqual(env.task_version, "precision-striker-v5-bounce-extra-obs")
            obs, _ = env.reset(seed=3, options={"blocks": [], "puck_position": [0.5, 0.0],
                                                "puck_velocity": [0.4, 0.0]})
            self.assertEqual(obs.shape, (22,))
            self.assertTrue(env.observation_space.contains(obs))
            np.testing.assert_allclose(obs[-2:], [0, 0])
            hit_seen = False
            for step in range(1, 21):
                obs, _, terminated, truncated, info = env.step(np.zeros(2))
                self.assertAlmostEqual(float(obs[-1]), step / 40, places=6)
                self.assertEqual(bool(obs[-2]), info["hit"])
                hit_seen |= info["hit"]
                if terminated or truncated:
                    break
            self.assertTrue(hit_seen)  # the puck drives into the stationary paddle
        finally:
            env.close()

    def test_v4_fast_diagonal_goal_is_counted(self):
        env = PrecisionStrikerEnv(obstacles=1, randomize=False, require_hit=False, obstacle_bounce=True)
        try:
            env.reset(seed=1, options={"blocks": []})
            puck = env.simulator.pucks["puck_0"]
            # Crosses the end line at y ~ -0.08 (inside the 0.12 m half-mouth) heading out sideways.
            puck.position = env.simulator.base_coord_to_box2d((-0.80, 0.0))
            puck.linearVelocity = env.simulator.base_coord_to_box2d((-2.3, -1.1))
            for _ in range(10):
                _, _, terminated, _, info = env.step(np.zeros(2))
                if terminated:
                    break
            self.assertEqual(info["outcome"], "goal")
        finally:
            env.close()

    def test_single_hit_ends_on_second_touch_or_return(self):
        env = PrecisionStrikerEnv(obstacles=1, randomize=False, require_hit=True, obstacle_bounce=True,
                                  single_hit=True)
        try:
            self.assertTrue(env.task_version.endswith("-single"))
            # Puck drives into a stationary paddle, bounces away, and the paddle chases it: second touch.
            env.reset(seed=3, options={"blocks": [], "puck_position": [0.45, 0.0], "puck_velocity": [0.6, 0.0]})
            outcomes = []
            for _ in range(40):
                gap = env.current_state["pucks"][0]["position"][0] - env.current_state["paddles"]["paddle_ego"]["position"][0]
                _, _, terminated, truncated, info = env.step(np.array([np.sign(gap), 0.0]))
                outcomes.append(info["outcome"])
                if terminated or truncated:
                    break
            self.assertIn(info["outcome"], ("double_hit", "missed"))
            self.assertTrue(info["hit"])
            self.assertFalse(info["is_success"])
        finally:
            env.close()

    def test_block_line_prob_blocks_the_straight_shot(self):
        from scripts.scoring_geometry import shot_paths
        env = PrecisionStrikerEnv(obstacles=1, wide_obstacles=True, block_line_prob=1.0)
        try:
            blocked = 0
            for seed in range(20):
                obs, info = env.reset(seed=seed)
                self.assertTrue(info["scorable"])
                strike = (0.45, float(np.clip(obs[5] + obs[7] / obs[6] * (0.45 - obs[4]), -0.35, 0.35)))
                straight = shot_paths(strike, info["blocks"])
                blocked += "direct" not in straight or len(straight) == 1
            self.assertGreaterEqual(blocked, 12)   # most episodes need a bank shot or a narrow aim
        finally:
            env.close()

    def test_scoring_geometry_detects_blocked_and_open_tables(self):
        from scripts.scoring_geometry import scorable, shot_paths
        incoming = ((-0.05, 0.0), (1.0, 0.0))
        self.assertTrue(scorable(*incoming, []))
        self.assertEqual(shot_paths((0.4, 0.0), [(-0.65, 0.0)]), {"bank"})
        wall = [(-0.5, y) for y in (-0.36, -0.27, -0.18, -0.09, 0, 0.09, 0.18, 0.27, 0.36)]
        self.assertFalse(scorable(*incoming, wall))
        self.assertTrue(scorable(*incoming, [b for b in wall if b[1] not in (-0.09, 0)]))
        self.assertFalse(scorable(*incoming, [(-0.90, -0.045), (-0.90, 0.045)]))

    def test_random_episodes_are_scorable(self):
        for obstacles in (1, 3):
            env = PrecisionStrikerEnv(obstacles=obstacles)
            try:
                for seed in range(25):
                    _, info = env.reset(seed=seed)
                    self.assertTrue(info["scorable"])
            finally:
                env.close()

    def test_trajectory_checked_multi_block_layouts_are_makeable(self):
        from scripts.layout_check import verified
        env = PrecisionStrikerEnv(obstacles=2, obstacles_max=3, wide_obstacles=True, block_line_prob=0.5,
                                  paddle_restitution=0.8, layout_check="trajectory")
        try:
            self.assertTrue(env.task_version.endswith("-trajcheck"))
            for seed in range(25):
                _, info = env.reset(seed=seed)
                self.assertIn(len(info["blocks"]), (2, 3))
                check = dict(paddle=env._spawn_paddle, restitution=0.8)
                puck = env._spawn_puck
                self.assertTrue(verified(puck, info["initial_velocity"], info["blocks"], **check)
                                or not verified(puck, info["initial_velocity"], [], **check))
        finally:
            env.close()
        self.assertFalse(verified((0.0, 0.0), (1.0, 0.0), [(-0.5, y) for y in np.arange(-0.32, 0.33, 0.09)]))

    def test_aim_shaping_rewards_clear_shot_lines(self):
        env = PrecisionStrikerEnv(obstacles=1, randomize=False, require_hit=True, obstacle_bounce=True,
                                  aim_shaping=5.0)
        try:
            self.assertTrue(env.task_version.endswith("-aim5"))
            env.reset(seed=1, options={"blocks": [(-0.5, 0.0)]})
            self.assertLess(env._aim_error((0.4, 0.0), (-2.0, 0.0)), 180)
            straight = env._aim_error((0.4, 0.0), (-2.0, 0.0))      # aimed through the block
            from scripts import scoring_geometry as sg
            shots = [s for s in sg._candidate_shots(np.array([0.4, 0.0]), [(-0.5, 0.0)]) if s["kind"] == "bank"]
            bank = env._aim_error((0.4, 0.0), 2.0 * shots[0]["direction"])
            self.assertGreater(straight, 2.0)
            self.assertLess(bank, 0.5)
            self.assertEqual(env._aim_error((0.4, 0.0), (1.0, 0.0)), 180.0)
        finally:
            env.close()

    def test_reject_unreachable_or_overlapping_setup(self):
        with self.assertRaises(ValueError):
            self.env.reset(options={"puck_position": [-0.5, 0]})
        with self.assertRaises(ValueError):
            self.env.reset(options={"blocks": [(-0.4, 0), (-0.4, 0)]})
        with self.assertRaises(ValueError):
            self.env.reset(options={"puck_velocity": [float("nan"), 0]})


if __name__ == "__main__":
    unittest.main()
