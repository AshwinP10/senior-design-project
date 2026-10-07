"""Geometric scoring feasibility for the precision-striker table.

A shot is scorable if, from some point on the incoming puck's path inside the paddle workspace,
there is a straight line or a single long-rail bank into the goal mouth that keeps the puck
center at least (puck radius + margin) away from every obstacle. Long rails are treated as mirrors
(their restitution is 0.99). Coordinates are base table coordinates: +x robot half, goal at -x.
"""
from __future__ import annotations

import numpy as np

LENGTH, WIDTH = 1.9304, 0.8636
GOAL_WIDTH, PUCK_R, BLOCK_W = 0.24, 0.03175, 0.07
MARGIN = 0.01
SAFE_CLEARANCE = 0.06                           # obstacle clearance beyond which the aim prefers the goal center
PADDLE_X, PADDLE_Y = (0.10, 0.80), 0.35        # region where the paddle can strike
RAIL_Y = WIDTH / 2 - PUCK_R                     # puck-center line at a long rail
GOAL_X = -LENGTH / 2
GOAL_LANE = GOAL_WIDTH / 2 - PUCK_R             # |y| of the puck center that fits in the mouth
GOAL_YS = np.linspace(-0.6 * GOAL_LANE, 0.6 * GOAL_LANE, 7)


def _clear(p0, p1, blocks, clearance, samples=120):
    """True if the segment p0->p1 stays at least `clearance` from every axis-aligned square."""
    if not blocks:
        return True
    t = np.linspace(0.0, 1.0, samples)[:, None]
    pts = p0 + t * (p1 - p0)
    half = BLOCK_W / 2
    for bx, by in blocks:
        dx = np.maximum(np.abs(pts[:, 0] - bx) - half, 0.0)
        dy = np.maximum(np.abs(pts[:, 1] - by) - half, 0.0)
        if np.min(np.hypot(dx, dy)) < clearance:
            return False
    return True


def shot_paths(strike, blocks, clearance=PUCK_R + MARGIN):
    """Kinds of clear paths ('direct', 'bank') from one strike point into the goal."""
    s = np.asarray(strike, dtype=float)
    kinds = set()
    for gy in GOAL_YS:
        g = np.array([GOAL_X, gy])
        if "direct" not in kinds and _clear(s, g, blocks, clearance):
            kinds.add("direct")
        for wall in (RAIL_Y, -RAIL_Y):
            mirrored = np.array([GOAL_X, 2 * wall - gy])
            f = (wall - s[1]) / (mirrored[1] - s[1])          # where the unfolded line meets the rail
            if not 0 < f < 1:
                continue
            bounce = s + f * (mirrored - s)
            if not GOAL_X < bounce[0] < s[0]:
                continue
            if _clear(s, bounce, blocks, clearance) and _clear(bounce, g, blocks, clearance):
                kinds.add("bank")
                break
        if kinds == {"direct", "bank"}:
            break
    return kinds


def _clearance(p0, p1, blocks, samples=120):
    """Smallest distance from the segment p0->p1 to any obstacle square (inf without obstacles)."""
    if not blocks:
        return np.inf
    t = np.linspace(0.0, 1.0, samples)[:, None]
    pts = p0 + t * (p1 - p0)
    half = BLOCK_W / 2
    best = np.inf
    for bx, by in blocks:
        dx = np.maximum(np.abs(pts[:, 0] - bx) - half, 0.0)
        dy = np.maximum(np.abs(pts[:, 1] - by) - half, 0.0)
        best = min(best, float(np.min(np.hypot(dx, dy))))
    return best


_SHOT_CACHE = {}


def candidate_shots(strike, blocks, clearance=PUCK_R + MARGIN):
    """Cached on a 5 mm grid of strike points (the planner asks for nearby points every step)."""
    key = (round(float(strike[0]) / 0.005), round(float(strike[1]) / 0.005),
           tuple((round(x, 4), round(y, 4)) for x, y in blocks), clearance)
    hit = _SHOT_CACHE.get(key)
    if hit is None:
        if len(_SHOT_CACHE) > 200_000:
            _SHOT_CACHE.clear()
        hit = _SHOT_CACHE[key] = _candidate_shots((key[0] * 0.005, key[1] * 0.005), blocks, clearance)
    return hit


def _candidate_shots(strike, blocks, clearance=PUCK_R + MARGIN):
    """All clear shots from a strike point: dicts with kind, unit direction, clearance, aim point.

    Sorted best first: direct before bank, then by obstacle clearance (beyond SAFE_CLEARANCE extra
    clearance does not matter), then by aiming closer to the goal center, which tolerates aim error.
    """
    s = np.asarray(strike, dtype=float)
    shots = []
    for gy in GOAL_YS:
        g = np.array([GOAL_X, gy])
        c = _clearance(s, g, blocks)
        if c >= clearance:
            d = g - s
            shots.append({"kind": "direct", "direction": d / np.linalg.norm(d), "clearance": c,
                          "aim": g, "goal_y": gy})
        for wall in (RAIL_Y, -RAIL_Y):
            mirrored = np.array([GOAL_X, 2 * wall - gy])
            f = (wall - s[1]) / (mirrored[1] - s[1])
            if not 0 < f < 1:
                continue
            bounce = s + f * (mirrored - s)
            if not GOAL_X < bounce[0] < s[0]:
                continue
            c = min(_clearance(s, bounce, blocks), _clearance(bounce, g, blocks))
            if c >= clearance:
                d = bounce - s
                shots.append({"kind": "bank", "direction": d / np.linalg.norm(d), "clearance": c,
                              "aim": bounce, "goal_y": gy})
    shots.sort(key=lambda sh: (sh["kind"] != "direct", -min(sh["clearance"], SAFE_CLEARANCE), abs(sh["goal_y"])))
    return shots


def incoming_strike_points(puck, velocity, n=12):
    """Points on the incoming puck path (folded at the long rails) inside the paddle strike region."""
    p = np.asarray(puck, dtype=float)
    v = np.asarray(velocity, dtype=float)
    if v[0] <= 0:
        return []
    points = []
    for x in np.linspace(*PADDLE_X, n):
        y = p[1] + v[1] / v[0] * (x - p[0])
        period = 4 * RAIL_Y                                  # fold the straight line at both rails
        y = (y + RAIL_Y) % period
        y = (period / 2 - abs(y - period / 2)) - RAIL_Y
        if abs(y) <= PADDLE_Y:
            points.append((x, y))
    return points


def scorable(puck, velocity, blocks):
    """True if at least one strike point on the incoming path has a clear direct or bank shot."""
    return any(shot_paths(s, blocks) for s in incoming_strike_points(puck, velocity))
