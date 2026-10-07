"""Hand-coded strike controller and geometric planner for the precision-striker task.

The controller only uses the 20 observation values the RL policy sees (paddle and puck state plus
obstacle slots) and emits the same 2D action, so it runs in the unmodified environment.

Pipeline every control step (20 Hz):
  1. Predict the puck path (linear damping, near-elastic long-rail bounces).
  2. Choose an interception time T and an outgoing direction u (from scoring_geometry clear paths,
     or from an external "shot" choice supplied by a learned policy).
  3. Solve the paddle-puck collision so the puck leaves along u at speed k: with the servo-driven
     paddle acting as infinite mass and restitution 1, the outgoing puck velocity is
     v' = v + 2((w - v)·n) n. Choosing the contact normal n ∥ (k u - v) and paddle speed
     s = v·n + |k u - v| / 2 along n gives v' = k u exactly.
  4. Track a reference: wait at a pre-strike point, then move along n at speed s through contact.
"""
from __future__ import annotations

import numpy as np

from scripts import scoring_geometry as sg

PADDLE_R = 0.0508
CONTACT = PADDLE_R + sg.PUCK_R
WORK_X, WORK_Y = (0.06, 0.86), 0.34          # paddle-center limits (env projects to 0.0508..0.88, ±0.35)
DT = 0.05
MAX_PADDLE = 2.0
ACCEL = 10.0

# Settings chosen by scripts/tacc/planner_search.py (Lonestar6 job 3494326, 2026-10-07): best of 270
# combinations on validation seed 50000 (200/200), confirmed on validation seed 60000 (193/200), then
# scored once on the reported 200-shot test (seed 30000): 192/200 = 96.0% under the bounce rule.
TUNED = {"kp_lat": 16.0, "min_slack": 0.0, "out_speed": 2.4, "kp_track": 8.0, "tau": 0.12}


def predict(p, v, damping=0.15, horizon=3.0, dt=0.01):
    """Puck centre path: (times, positions, velocities), vectorised closed form.

    Box2D damping multiplies the velocity by q = 1/(1+h*c) each step, so after k steps
    v_k = v0 q^k and p_k = p0 + h v0 q (1 - q^k)/(1 - q). Long-rail bounces are applied by folding
    the straight path at y = +-RAIL_Y (rail restitution 0.99 is treated as 1). The path stops once the
    puck leaves the table length.
    """
    p, v = np.asarray(p, float), np.asarray(v, float)
    k = np.arange(int(horizon / dt) + 1)
    q = 1.0 / (1.0 + dt * damping)
    qk = q ** k
    travel = dt * q * (1 - qk) / (1 - q)
    xs = p[0] + v[0] * travel
    ys = p[1] + v[1] * travel
    vx, vy = v[0] * qk, v[1] * qk
    period = 4 * sg.RAIL_Y
    shifted = np.mod(ys + sg.RAIL_Y, period)
    mirrored = shifted > period / 2
    ys = np.where(mirrored, period - shifted, shifted) - sg.RAIL_Y
    vy = np.where(mirrored, -vy, vy)
    inside = (xs <= sg.LENGTH / 2 + 0.05) & (xs >= sg.GOAL_X - 0.05)
    n = len(k) if inside.all() else max(int(np.argmin(inside)) + 1, 1)
    return k[:n] * dt, np.stack([xs, ys], 1)[:n], np.stack([vx, vy], 1)[:n]


def solve_strike(v_in, u, k):
    """Contact normal n and paddle speed s that send a puck with velocity v_in along u at speed k."""
    want = k * np.asarray(u) - np.asarray(v_in)
    norm = np.linalg.norm(want)
    n = want / norm
    s = float(np.dot(v_in, n) + norm / 2)
    return n, s


class StrikePlanner:
    def __init__(self, damping=0.15, out_speed=2.4, max_strike=1.7, tau=0.08,
                 kp_move=6.0, kp_track=4.0, kp_lat=4.0, min_slack=0.0, x_window=(0.12, 0.82),
                 shot_chooser=None):
        self.damping, self.out_speed, self.max_strike, self.tau = damping, out_speed, max_strike, tau
        self.kp_move, self.kp_track, self.x_window = kp_move, kp_track, x_window
        # kp_lat: extra gain on tracking error perpendicular to the strike direction (sets aim accuracy).
        # min_slack: prefer interceptions that leave at least this much spare time to line up (s).
        self.kp_lat, self.min_slack = kp_lat, min_slack
        # Optional callable(strike_point, shots, obs) -> shot dict, used by learned shot selection.
        self.shot_chooser = shot_chooser
        self.reset()

    def reset(self):
        self.t = 0.0
        self.plan = None
        self.struck_at = None
        self.command = None          # shot chosen by a learned policy (see set_command)
        self.needs_decision = True   # True at the start and whenever the puck comes back
        self.shots = 0

    def set_command(self, angle, speed, x_pref):
        """Use an external shot: direction angle (rad, from +x), outgoing speed (m/s), and the
        preferred x of the interception point. The controller still handles timing and contact."""
        self.command = {"direction": np.array([np.cos(angle), np.sin(angle)]), "speed": float(speed),
                        "x_pref": float(x_pref)}
        self.needs_decision = False
        self.plan = None

    # ----------------------------------------------------------------- planning
    def _blocks(self, obs):
        slots = np.asarray(obs[8:20]).reshape(3, 4)
        return [(float(x), float(y)) for x, y, _w, present in slots if present > 0.5]

    def _make_plan(self, obs, T, S, v_at, shot, k=None):
        u = shot["direction"]
        k = self.out_speed if k is None else k
        for _ in range(8):                       # lower the shot speed until the paddle can deliver it
            n, s = solve_strike(v_at, u, k)
            if s <= self.max_strike:
                break
            k *= 0.85
        if s <= 0.05 or s > self.max_strike:
            return None
        C = S - n * CONTACT                       # paddle centre at contact
        if not (WORK_X[0] <= C[0] <= WORK_X[1] and abs(C[1]) <= WORK_Y):
            return None
        # Run-up: time for the servo to reach strike speed plus a margin (self.tau).
        tau = s / ACCEL + self.tau
        ramp = s * s / (2 * ACCEL)               # distance lost while the servo accelerates
        Q = C - n * max(s * tau - ramp, 0.02)
        if not (WORK_X[0] <= Q[0] <= WORK_X[1] and abs(Q[1]) <= WORK_Y):
            return None
        travel = np.linalg.norm(Q - obs[:2])
        slack = (T - tau) - (travel / 1.8 + 0.12)
        if slack < 0:
            return None
        return {"T": self.t + T, "S": S, "n": n, "s": s, "k": k, "C": C, "Q": Q, "shot": shot,
                "tau": tau, "slack": slack}

    def _plan(self, obs):
        ts, ps, vs = predict(obs[4:6], obs[6:8], self.damping)
        blocks = self._blocks(obs)
        best = None
        for i in range(2, len(ts), 5):            # candidate interception times every 0.05 s
            S = ps[i]
            if not (self.x_window[0] <= S[0] <= self.x_window[1] and abs(S[1]) <= WORK_Y):
                continue
            if self.command is not None:
                shot = {"kind": "policy", "direction": self.command["direction"], "clearance": 0.0, "goal_y": 0.0}
                plan = self._make_plan(obs, ts[i], S, vs[i], shot, k=self.command["speed"])
                if plan is None:
                    continue
                score = (-abs(S[0] - self.command["x_pref"]),)
                if best is None or score > best[0]:
                    best = (score, plan)
                continue
            shots = sg.candidate_shots(S, blocks)
            if not shots:
                continue
            shot = self.shot_chooser(S, shots, obs) if self.shot_chooser else shots[0]
            plan = self._make_plan(obs, ts[i], S, vs[i], shot)
            if plan is None:
                continue
            score = (shot["kind"] == "direct", min(shot["clearance"], 0.15),
                     min(plan["slack"], self.min_slack), -ts[i])
            if best is None or score > best[0]:
                best = (score, plan)
        return None if best is None else best[1]

    def _refresh(self, obs):
        """Re-aim the committed plan at the latest prediction of where the puck will be at T."""
        plan = self.plan
        remaining = plan["T"] - self.t
        ts, ps, vs = predict(obs[4:6], obs[6:8], self.damping, horizon=max(remaining, 0) + 0.05)
        i = min(int(round(remaining / 0.01)), len(ts) - 1)
        if i <= 0:
            return
        n, s = solve_strike(vs[i], plan["shot"]["direction"], plan["k"])
        if s > MAX_PADDLE or s <= 0:
            return
        plan.update(S=ps[i], n=n, s=s, C=ps[i] - n * CONTACT)

    # ------------------------------------------------------------------ control
    def act(self, obs):
        obs = np.asarray(obs, dtype=float)
        paddle, puck, puck_v = obs[:2], obs[4:6], obs[6:8]
        if self.struck_at is None and self.plan is not None and self.t > self.plan["T"] + 0.05:
            if np.linalg.norm(puck - paddle) < CONTACT + 0.15 or puck_v[0] < 0:
                self.struck_at = self.t
                self.shots += 1
        # After a shot: retreat, and plan again if the puck comes back to the robot side.
        if self.struck_at is not None:
            if puck_v[0] > 0.05 and puck[0] > -0.4 and self.t - self.struck_at > 0.3:
                self.plan, self.struck_at = None, None
                self.command, self.needs_decision = None, True
            else:
                cmd = self.kp_move * (np.array([0.75, 0.0]) - paddle)
                return self._finish(cmd)
        in_approach = self.plan is not None and self.t >= self.plan["T"] - self.plan["tau"] - 1e-9
        if not in_approach:
            new = self._plan(obs)
            if new is not None:
                self.plan = new
        elif self.plan["T"] - self.t > DT:
            self._refresh(obs)
        if self.plan is None:                       # nothing reachable yet: hold a central home
            return self._finish(self.kp_move * (np.array([0.70, np.clip(puck[1], -0.2, 0.2)]) - paddle))
        p = self.plan
        if self.t < p["T"] - p["tau"] - 1e-9:
            cmd = self.kp_move * (p["Q"] - paddle)
        else:
            t_next = self.t + DT
            ref = p["C"] - p["n"] * p["s"] * (p["T"] - t_next)
            err = ref - paddle
            along = np.dot(err, p["n"]) * p["n"]
            cmd = p["s"] * p["n"] + self.kp_track * along + self.kp_lat * (err - along)
            if self.t > p["T"] + 0.1:
                self.struck_at = self.t
        return self._finish(cmd)

    def _finish(self, cmd):
        self.t += DT
        speed = np.linalg.norm(cmd)
        if speed > MAX_PADDLE:
            cmd = cmd * MAX_PADDLE / speed
        return (cmd / MAX_PADDLE).astype(np.float32)    # env maps a -> 2 * a / max(1, |a|)
