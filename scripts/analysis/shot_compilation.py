"""Classify every strict-suite test shot of a policy and render a labelled success / failure video.

    python -m scripts.analysis.shot_compilation --run models/tacc-2026-10/AB-seed3-bank \
        --classify output/report/shot_classes.json --video output/report/strike_compilation.mp4

Shots are the strict suite's test shots (scripts/eval_suite.py, seed 30000): one paddle contact, paddle
restitution 0.8, obstacles anywhere. The simulation is deterministic per seed, so a classified shot replays
exactly when rendered.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from scripts import scoring_geometry as sg
from scripts.eval_suite import CONDITIONS, RULES, _policy
from scripts.precision_striker_env import PrecisionStrikerEnv

COND_TEXT = {"random_1": "1 block, random spot", "blocked_1": "1 block on the straight line to the goal",
             "random_1to3": "1-3 blocks, random spots", "blocked_1to3": "1-3 blocks, straight line blocked"}
KIND_TEXT = {
    "straight": ("SUCCESS - straight shot", "Clear straight line to the goal: the policy hits it straight in."),
    "bank": ("SUCCESS - bank shot", "The straight line is blocked, so the policy banks the puck off the side rail into the goal."),
    "ricochet": ("SUCCESS - ricochet", "The puck glanced off a block and still went in (partly luck)."),
    "clipped_block": ("FAILURE - clipped a block", "Aim a few degrees off: the puck hits a block and bounces back."),
    "bank_wide": ("FAILURE - bank shot off target", "Bounced off the side rail at the wrong angle and missed the goal mouth."),
    "wide": ("FAILURE - just wide", "Reached the far wall beside the goal mouth and bounced back."),
    "double_touch": ("FAILURE - touched twice", "Paddle touched the puck a second time - not allowed (one contact only)."),
    "no_shot": ("FAILURE - no clean hit in time", "Slow or awkward incoming puck; the policy never made a clean strike."),
    "other": ("FAILURE - other", "Shot ended without a goal."),
}


def _run_shot(env, act, reset, seed, record=None):
    """Play one shot. Returns (kind, outcome, steps); record(env, obs, info, t) is called every step."""
    obs, info = env.reset(seed=seed)
    reset()
    hit_seen, bank, block_after_hit, min_x, prev_vy = False, False, False, 9.0, None
    t, done, info = 0, False, {"hit": False, "outcome": "running", "obstacle_collision": False}
    if record:
        record(env, obs, info, t)
    while not done:
        before = info.get("obstacle_collision", False)
        obs, _, terminated, truncated, info = env.step(act(obs))
        t += 1
        done = terminated or truncated
        if info["hit"]:
            if hit_seen and info["obstacle_collision"] and not before:
                block_after_hit = True
            hit_seen = True
            x, vy = float(obs[4]), float(obs[7])
            min_x = min(min_x, x)
            # A side-rail bounce flips the sideways velocity; ignore flips on steps with a new block contact.
            touched_block = info["obstacle_collision"] and not before
            if (prev_vy is not None and abs(vy) > 0.05 and abs(prev_vy) > 0.05 and np.sign(vy) != np.sign(prev_vy)
                    and not touched_block and obs[6] < 0):
                bank = True
            prev_vy = vy
        if record:
            record(env, obs, info, t)
    outcome = info["outcome"]
    if outcome == "goal" and info["hit"]:
        kind = "ricochet" if block_after_hit else "bank" if bank else "straight"
    elif outcome == "double_hit":
        kind = "double_touch"
    elif outcome == "timeout" or not info["hit"]:
        kind = "no_shot"
    elif outcome == "missed":
        kind = "clipped_block" if block_after_hit else "bank_wide" if bank else "wide" if min_x < -0.8 else "clipped_block" \
            if info["obstacle_collision"] else "wide"
    else:
        kind = "other"
    return kind, outcome, t


def classify(job):
    name, run, seed, shots = job
    act, reset, obs_opts = _policy(run, False)
    env = PrecisionStrikerEnv(seed=seed, randomize=False, **{**RULES, **obs_opts, **CONDITIONS[name]})
    rows = []
    for i in range(shots):
        s = seed + 1000 + i
        kind, outcome, steps = _run_shot(env, act, reset, s)
        rows.append({"condition": name, "seed": s, "kind": kind, "outcome": outcome, "steps": steps,
                     "blocks": len(env.user_block_positions)})
    env.close()
    return rows


# ------------------------------------------------------------------------------------------ video
W, H = 1280, 720
FONT = cv2.FONT_HERSHEY_SIMPLEX
INK, MUTED, GOOD, BAD, BG = (31, 31, 31), (100, 100, 100), (80, 150, 34), (45, 55, 200), (255, 255, 255)


def _wrap(text, scale, width, thick=1):
    words, lines, line = text.split(), [], ""
    for w in words:
        trial = (line + " " + w).strip()
        if cv2.getTextSize(trial, FONT, scale, thick)[0][0] > width and line:
            lines.append(line)
            line = w
        else:
            line = trial
    return lines + ([line] if line else [])


def card(title, lines, accent=INK):
    im = np.full((H, W, 3), 255, np.uint8)
    cv2.rectangle(im, (0, 0), (W, 12), accent, -1)
    y = 150
    for ln in _wrap(title, 1.4, W - 160, 3):
        cv2.putText(im, ln, (80, y), FONT, 1.4, accent, 3, cv2.LINE_AA)
        y += 62
    y += 20
    for text in lines:
        for ln in _wrap(text, 0.8, W - 160, 2):
            cv2.putText(im, ln, (80, y), FONT, 0.8, INK, 2, cv2.LINE_AA)
            y += 40
        y += 12
    return im


def render_clip(env, act, reset, row, number, total, canvas, slow=2, hold=30):
    title, explain = KIND_TEXT[row["kind"]]
    good = title.startswith("SUCCESS")
    frames, trail = [], []

    def rec(env, obs, info, t):
        st = env.current_state
        trail.append(st["pucks"][0]["position"])
        im = canvas.frame(st["pucks"][0]["position"], st["paddles"]["paddle_ego"]["position"],
                          list(env.user_block_positions), trail[-40:], f"Case {number}/{total}:  {COND_TEXT[row['condition']]}",
                          f"t = {t * 0.05:.2f} s", info.get("hit", False))
        frames.append(im)

    _run_shot(env, act, reset, row["seed"], rec)
    out = []
    for i, im in enumerate(frames):
        last = i == len(frames) - 1
        f = _compose(im, title, explain, good, banner=last)
        out.extend([f] * (hold if last else slow))
    return out


def _compose(table, title, explain, good, banner=False):
    th, tw = table.shape[:2]
    top = 520                                     # table area height; caption below
    tw2 = int(tw * top / th)
    table = cv2.resize(table, (tw2, top), interpolation=cv2.INTER_AREA)
    im = np.full((H, W, 3), 255, np.uint8)
    x0 = (W - tw2) // 2
    im[:top, x0:x0 + tw2] = table
    colour = GOOD if good else BAD
    cv2.rectangle(im, (0, top), (W, H), (247, 247, 247), -1)
    cv2.rectangle(im, (0, top), (14, H), colour, -1)
    cv2.putText(im, title, (36, top + 44), FONT, 1.05, colour, 3, cv2.LINE_AA)
    y = top + 86
    for ln in _wrap(explain, 0.72, W - 80, 2):
        cv2.putText(im, ln, (36, y), FONT, 0.72, INK, 2, cv2.LINE_AA)
        y += 34
    if banner:
        text = "GOAL" if good else "NO GOAL"
        (bw, bh), _ = cv2.getTextSize(text, FONT, 1.6, 4)
        cv2.rectangle(im, (W // 2 - bw // 2 - 24, 90), (W // 2 + bw // 2 + 24, 90 + bh + 30), colour, -1)
        cv2.putText(im, text, (W // 2 - bw // 2, 90 + bh + 14), FONT, 1.6, (255, 255, 255), 4, cv2.LINE_AA)
    return im


def write_video(frames, path, fps=20):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg failed")


def pick(rows, plan):
    """plan: list of (kind, condition or None, count). Distinct seeds, in plan order."""
    chosen, used = [], set()
    for kind, cond, count in plan:
        cands = [r for r in rows if r["kind"] == kind and (cond is None or r["condition"] == cond)
                 and (r["condition"], r["seed"]) not in used]
        cands.sort(key=lambda r: (-r["blocks"], r["seed"]))
        step = max(1, len(cands) // max(count, 1))
        for r in cands[::step][:count]:
            chosen.append(r)
            used.add((r["condition"], r["seed"]))
    return chosen


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--seed", type=int, default=30000)
    p.add_argument("--shots", type=int, default=200)
    p.add_argument("--classify", type=Path, required=True, help="JSON of per-shot classes (made if missing)")
    p.add_argument("--video", type=Path)
    p.add_argument("--label", default="RL policy")
    args = p.parse_args()
    if args.classify.exists():
        rows = json.loads(args.classify.read_text())
    else:
        with ProcessPoolExecutor(4) as pool:
            rows = [r for part in pool.map(classify, [(n, str(args.run), args.seed, args.shots) for n in CONDITIONS])
                    for r in part]
        args.classify.parent.mkdir(parents=True, exist_ok=True)
        args.classify.write_text(json.dumps(rows, indent=1))
    for name in CONDITIONS:
        print(name, dict(Counter(r["kind"] for r in rows if r["condition"] == name)))
    if not args.video:
        return
    from scripts.demo_live import FastCanvas
    plan_ok = [("straight", "random_1", 2), ("straight", "random_1to3", 1), ("bank", "blocked_1", 3),
               ("bank", "blocked_1to3", 2), ("bank", "random_1to3", 1), ("ricochet", None, 1)]
    plan_bad = [("clipped_block", None, 3), ("wide", None, 2), ("bank_wide", None, 2), ("double_touch", None, 1),
                ("no_shot", None, 1)]
    good, bad = pick(rows, plan_ok), pick(rows, plan_bad)
    total = len(good) + len(bad)
    act, reset, obs_opts = _policy(str(args.run), False)
    canvas = FastCanvas(width=1280)
    envs = {n: PrecisionStrikerEnv(seed=args.seed, randomize=False, **{**RULES, **obs_opts, **CONDITIONS[n]})
            for n in CONDITIONS}
    n_shots = len(rows)
    goals = sum(r["outcome"] == "goal" for r in rows)
    frames = [card("Precision Striker - successful vs unsuccessful strikes", [
        f"Policy: {args.label} (pure reinforcement learning: a neural network outputs the paddle velocity).",
        "Rules: one paddle contact per shot, realistic paddle bounce (no speed-up), blocks anywhere on the far half.",
        f"All clips are real test shots the policy had never seen. Overall it scored {goals} of {n_shots} "
        f"test shots ({100 * goals / n_shots:.1f}%).",
        "Playback is half speed. Grey dots show the puck's recent path."])] * 100
    frames += [card("Part 1 - successful strikes", [f"{len(good)} cases: straight shots, bank shots around blocked "
                                                     "lines, and multi-block layouts."], GOOD)] * 50
    for i, r in enumerate(good, 1):
        frames += render_clip(envs[r["condition"]], act, reset, r, i, total, canvas)
    frames += [card("Part 2 - failure cases, and why they happen", [
        f"{len(bad)} cases covering every failure type we see.",
        "Most misses are aim errors of a few degrees: the policy picks the right kind of shot but the angle is "
        "slightly off, so the puck clips a block or lands just beside the goal mouth."], BAD)] * 70
    for i, r in enumerate(bad, len(good) + 1):
        frames += render_clip(envs[r["condition"]], act, reset, r, i, total, canvas)
    counts = Counter(r["kind"] for r in rows if r["outcome"] != "goal")
    fails = n_shots - goals
    lines = [f"{KIND_TEXT[k][0].split(' - ')[1].capitalize()}: {c} of {fails} failures ({100 * c / fails:.0f}%)"
             for k, c in counts.most_common()]
    frames += [card("What the failures tell us", lines + [
        "Next: keep training from this policy so it keeps the new bank shots but takes the straight shot when it "
        "is open, and sharpen its aim (training already running on TACC)."], INK)] * 140
    args.video.parent.mkdir(parents=True, exist_ok=True)
    write_video(frames, args.video)
    print(f"wrote {args.video}: {len(frames) / 20:.0f} s, {total} cases")


if __name__ == "__main__":
    main()
