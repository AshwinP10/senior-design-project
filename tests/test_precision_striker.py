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
        self.assertGreater(reward, 90)

    def test_end_rail_outside_goal_bounces(self):
        self.env.reset(seed=1, options={"blocks": []})
        self.launch((-0.75, 0.3), (-2, 0))
        for _ in range(10):
            obs, _, _, _, info = self.env.step(np.zeros(2))
            self.assertFalse(info["is_success"])
        self.assertGreater(obs[6], 0)

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
        self.assertLess(reward, -90)

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

    def test_reject_unreachable_or_overlapping_setup(self):
        with self.assertRaises(ValueError):
            self.env.reset(options={"puck_position": [-0.5, 0]})
        with self.assertRaises(ValueError):
            self.env.reset(options={"blocks": [(-0.4, 0), (-0.4, 0)]})
        with self.assertRaises(ValueError):
            self.env.reset(options={"puck_velocity": [float("nan"), 0]})


if __name__ == "__main__":
    unittest.main()
