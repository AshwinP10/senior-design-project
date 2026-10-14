"""Trajectory-based check that a random obstacle layout can still be scored.

The geometric check (scoring_geometry.scorable) only asks whether a clear straight or bank line exists
from somewhere on the straight incoming path. This stricter check uses the strike controller's
trajectory calculator: it predicts the real incoming puck path (damping and rail bounces), and at each
interception time asks for a shot that
  * clears every obstacle by at least `min_clearance` (puck centre to block edge, so the puck itself has
    min_clearance - puck radius of room for aim error), and
  * the paddle can actually deliver: it can get from its start position to the run-up point in time,
    the contact and run-up points are inside the robot workspace, and the needed paddle speed is within
    the servo limit (solved with the same one-contact collision model the controller uses).
A layout passes if at least `min_window` interception times (0.05 s apart) have such a shot, so the
policy has some timing slack rather than one exact moment.
"""
from __future__ import annotations

import numpy as np

from scripts import scoring_geometry as sg
from scripts.strike_planner import TUNED, StrikePlanner, predict

MIN_CLEARANCE = sg.PUCK_R + 0.02      # 2 cm of room beyond the puck edge (geometric check uses 1 cm)
MIN_WINDOW = 2


def strike_options(puck, velocity, blocks, paddle=(0.67, 0.0), restitution=0.8, damping=0.15,
                   min_clearance=MIN_CLEARANCE):
    """Feasible (time, strike point, shot) triples along the predicted incoming path."""
    planner = StrikePlanner(**{**TUNED, "restitution": 1.0 if restitution is None else restitution})
    obs = np.zeros(20)
    obs[:2] = paddle
    ts, ps, vs = predict(puck, velocity, damping)
    options = []
    for i in range(2, len(ts), 5):
        S = ps[i]
        if not (planner.x_window[0] <= S[0] <= planner.x_window[1] and abs(S[1]) <= 0.34):
            continue
        for shot in sg.candidate_shots(S, blocks):
            if shot["clearance"] < min_clearance:
                continue
            if planner._make_plan(obs, ts[i], S, vs[i], shot) is not None:
                options.append((float(ts[i]), (float(S[0]), float(S[1])), shot))
                break
    return options


def verified(puck, velocity, blocks, min_window=MIN_WINDOW, **kwargs):
    return len(strike_options(puck, velocity, blocks, **kwargs)) >= min_window
