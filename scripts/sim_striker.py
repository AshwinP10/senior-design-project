"""Physics striker with simulator-in-the-loop aim correction.

The closed-form strike solve in strike_planner.py picks the right shot but executes it a few degrees off
(paddle servo lag, finite paddle mass, contact timing at 20 Hz), and the clear lines to the goal only
tolerate about +-3 degrees. Just before the strike starts, this planner copies the current state into a
private simulator with the same rules, plays the strike there, measures the puck's real outgoing heading,
rotates its aim by the error and tries again (a few iterations). If the corrected strike still does not
score in the copy, it tries the next-best candidate plans. Only observations are used to build the copy
(paddle and puck state, obstacle slots); the puck's sliding friction is estimated from how fast it slows.

    python -m scripts.eval_suite --controller --set sim=1          # see eval_suite.py
"""
from __future__ import annotations

import copy

import numpy as np

from scripts import scoring_geometry as sg
from scripts.strike_planner import DT, StrikePlanner

_MODELS = {}


def _model_env(env_kwargs):
    from scripts.precision_striker_env import PrecisionStrikerEnv
    key = tuple(sorted(env_kwargs.items()))
    if key not in _MODELS:
        _MODELS[key] = PrecisionStrikerEnv(seed=0, randomize=False, **env_kwargs)
    return _MODELS[key]


def _rotate(u, rad):
    c, s = np.cos(rad), np.sin(rad)
    return np.array([c * u[0] - s * u[1], s * u[0] + c * u[1]])


class SimStrikePlanner(StrikePlanner):
    # Relaxed planning settings tried, in order, when the normal settings find no plan: shorter run-up
    # margin (tau), less time reserved to reach it, softer shots. Riskier, which is why every plan is
    # test-fired in the copy before it is played.
    RELAX = ({"tau": 0.06, "reaction": 0.06}, {"tau": 0.02, "reaction": 0.03, "travel_speed": 2.0},
             {"tau": 0.02, "reaction": 0.03, "travel_speed": 2.0, "out_speed": 1.8},
             {"tau": 0.0, "reaction": 0.0, "travel_speed": 2.0, "out_speed": 1.4})

    def __init__(self, env_kwargs, iterations=3, alternatives=8, relax=True, **kwargs):
        super().__init__(**kwargs)
        # Rules for the private copy; obstacles and puck state come from each observation.
        self.env_kwargs = {k: v for k, v in env_kwargs.items()
                           if k not in ("obstacles", "obstacles_max", "block_line_prob", "seed", "randomize",
                                        "layout_check")}
        self.iterations, self.alternatives, self.relax = iterations, alternatives, relax
        self._rollout = False
        self.stats = {"checked": 0, "scored_in_copy": 0, "switched_plan": 0}

    def reset(self):
        super().reset()
        self._corrected = False
        self._locked = False
        self._prev_v = None
        self._damping_samples = []

    # ------------------------------------------------------------------ copy of the world
    def _estimate_damping(self, obs):
        v = np.asarray(obs[6:8], float)
        if self._prev_v is not None and self.struck_at is None and np.linalg.norm(obs[4:6] - obs[:2]) > 0.2:
            a, b = np.linalg.norm(self._prev_v), np.linalg.norm(v)
            same_side = np.sign(v[1]) == np.sign(self._prev_v[1]) or abs(v[1]) < 1e-3
            if a > 0.2 and b > 0 and same_side and b < a:
                ratio = b / a                                    # (1 + h c)^-10 per control step
                self._damping_samples.append(((ratio ** -0.1) - 1) / 0.005)
        self._prev_v = v
        if self._damping_samples:
            self.damping = float(np.clip(np.median(self._damping_samples), 0.05, 0.4))

    def _load(self, obs, steps):
        env = _model_env(self.env_kwargs)
        env.reset(seed=0, options={"blocks": self._blocks(obs), "puck_position": [0.4, 0.0],
                                   "puck_velocity": [0.0, 0.0]})
        sim = env.simulator
        puck, paddle = sim.pucks["puck_0"], sim.paddles["paddle_ego"]
        puck.position = sim.base_coord_to_box2d(tuple(float(x) for x in obs[4:6]))
        puck.linearVelocity = sim.base_coord_to_box2d(tuple(float(x) for x in obs[6:8]))
        paddle.position = sim.base_coord_to_box2d(tuple(float(x) for x in obs[:2]))
        paddle.linearVelocity = sim.base_coord_to_box2d(tuple(float(x) for x in obs[2:4]))
        sim.puck_damping = self.damping
        puck.linearDamping = self.damping
        env.current_state = sim.get_current_state()
        env._steps = steps
        env._previous_distance = env._intercept_distance(env.get_observation(env.current_state))
        return env

    def _play(self, obs, steps, agent):
        """Play the rest of the shot in the copy with a clone of this planner. Returns (heading, outcome)."""
        env = self._load(obs, steps)
        o = env.get_observation(env.current_state)
        heading, done, info = None, False, {"outcome": "running", "is_success": False}
        while not done:
            o, _, terminated, truncated, info = env.step(agent.act(o))
            done = terminated or truncated
            if info["hit"] and heading is None and np.linalg.norm(o[:2] - o[4:6]) > 0.12 and o[6] < 0:
                heading = o[6:8] / np.linalg.norm(o[6:8])
        return heading, info["outcome"], bool(info.get("is_success"))

    def _clone(self):
        agent = copy.copy(self)
        agent.plan = copy.deepcopy(self.plan)
        agent.command = copy.deepcopy(self.command)
        agent._rollout = True
        agent._corrected = True
        agent._locked = self._locked
        return agent

    def _correct(self, obs, steps):
        """Tune the committed plan's aim in the copy; fall back to other plans if it cannot score."""
        self.stats["checked"] += 1
        best = None
        for attempt, plan in enumerate(self._candidates(obs)):   # lazy: alternatives only if needed
            self.plan = plan
            self._locked = attempt > 0               # keep an alternative plan instead of re-planning
            u0 = np.array(plan["shot"]["direction"], float)
            for _ in range(self.iterations):
                agent = self._clone()
                heading, outcome, success = self._play(obs, steps, agent)
                if success:
                    self.stats["scored_in_copy"] += 1
                    self.stats["switched_plan"] += attempt > 0
                    return
                if heading is None:
                    break
                u = np.array(plan["shot"]["direction"], float)
                target = np.array(u0)
                err = np.arctan2(target[0] * heading[1] - target[1] * heading[0], np.dot(target, heading))
                plan["shot"] = dict(plan["shot"], direction=_rotate(u, -err))
                if best is None or abs(err) < best[0]:
                    best = (abs(err), copy.deepcopy(plan))
        if best is not None:
            self.plan = best[1]                              # nothing scored in the copy: least aim error
            self._locked = True

    def _candidates(self, obs):
        """The committed plan first, then the next-best interception plans (different time or shot)."""
        yield self.plan
        if self.alternatives <= 0:
            return
        from scripts.strike_planner import predict
        ts, ps, vs = predict(obs[4:6], obs[6:8], self.damping)
        blocks = self._blocks(obs)
        options = []
        saved = {k: getattr(self, k) for k in ("tau", "reaction", "travel_speed", "out_speed")}
        try:
            # Normal settings first, then the relaxed ones (rank: normal plans before relaxed, then margin).
            for level, setting in enumerate(({},) + (self.RELAX if self.relax else ())):
                for k, v in {**saved, **setting}.items():
                    setattr(self, k, v)
                for i in range(2, len(ts), 5):
                    S = ps[i]
                    if not (self.x_window[0] <= S[0] <= self.x_window[1] and abs(S[1]) <= 0.34):
                        continue
                    for shot in sg.candidate_shots(S, blocks):
                        p = self._make_plan(obs, ts[i], S, vs[i], shot)
                        if p is not None and p["T"] - self.t >= p["tau"] - 1e-9:
                            options.append((level, -sg.angular_margin(S, shot, blocks), p))
        finally:
            for k, v in saved.items():
                setattr(self, k, v)
        options.sort(key=lambda o: o[:2])
        for *_, p in options[: self.alternatives]:
            yield p

    # ------------------------------------------------------------------ control
    def _plan(self, obs):
        if self._locked and self.plan is not None:
            return self.plan                              # a checked plan is kept until it is played
        plan = super()._plan(obs)
        # Only when there is no plan at all: an existing plan is kept rather than swapped for a riskier one.
        if plan is None and self.relax and self.plan is None:
            saved = {k: getattr(self, k) for k in ("tau", "reaction", "travel_speed", "out_speed")}
            try:
                for setting in self.RELAX:
                    for k, v in setting.items():
                        setattr(self, k, v)
                    plan = super()._plan(obs)
                    if plan is not None:
                        break
            finally:
                for k, v in saved.items():
                    setattr(self, k, v)
        return plan

    def act(self, obs):
        obs = np.asarray(obs, dtype=float)
        if not self._rollout:
            self._estimate_damping(obs)
            p = self.plan
            if (p is not None and not self._corrected and self.struck_at is None
                    and self.t >= p["T"] - p["tau"] - 1e-9 and p["T"] - self.t > DT):
                self._correct(obs, int(round(self.t / DT)))
                self._corrected = True
        out = super().act(obs)
        if self.plan is None:                             # puck came back: the next shot gets checked again
            self._corrected = self._locked = False
        return out
