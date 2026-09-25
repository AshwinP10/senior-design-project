"""Watch the learned PPO policy in the actual Box2D simulation (no scripted player)."""
from __future__ import annotations

import argparse
from collections import Counter, deque
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from scripts.precision_striker_env import PrecisionStrikerEnv, TASK_VERSION

WINDOW = "RL Air Hockey | Incoming puck + obstacle scoring"


def draw_frame(env, info, trail, counts, steps, seed, trained_steps, paused, rate, training_obstacles, auto_next=False):
    frame = np.full((860, 1000, 3), (24, 19, 15), dtype=np.uint8)
    left, top, width, height = 40, 55, 340, 760
    def pixel(position):
        x, y = position
        return (int(left + width * (y / env.width + 0.5)),
                int(top + height * (x / env.length + 0.5)))
    scale = width / env.width
    cv2.rectangle(frame, (left, top), (left + width, top + height), (60, 45, 30), -1)
    wall = (170, 153, 127)
    cv2.line(frame, (left, top), (left, top+height), wall, 4)
    cv2.line(frame, (left+width, top), (left+width, top+height), wall, 4)
    goal_left = int(left + width/2 - env.simulator.goal_width*scale/2)
    goal_right = int(left + width/2 + env.simulator.goal_width*scale/2)
    for end in (top, top+height):
        cv2.line(frame, (left, end), (goal_left, end), wall, 4)
        cv2.line(frame, (goal_right, end), (left+width, end), wall, 4)
        cv2.line(frame, (goal_left, end), (goal_right, end), (150, 220, 110), 3)
    cv2.line(frame, (left, top+height//2), (left+width, top+height//2), (90, 80, 65), 1)
    for pos in env.user_block_positions:
        cx, cy = pixel(pos)
        half = round(env.block_width*scale/2)
        cv2.rectangle(frame, (cx-half, cy-half), (cx+half, cy+half), (75, 85, 240), -1)
    if len(trail) > 1:
        cv2.polylines(frame, [np.array([pixel(p) for p in trail])], False, (90, 140, 185), 2)
    puck = env.current_state["pucks"][0]
    paddle = env.current_state["paddles"]["paddle_ego"]
    cv2.circle(frame, pixel(puck["position"]), round(env.puck_radius*scale), (60, 185, 255), -1)
    cv2.circle(frame, pixel(paddle["position"]), round(env.paddle_radius*scale), (240, 210, 65), -1)
    def arrow(pos, vec, color):
        endpoint = np.asarray(pos) + 0.12*np.asarray(vec)
        cv2.arrowedLine(frame, pixel(pos), pixel(endpoint), color, 2, tipLength=0.25)
    arrow(puck["position"], puck["velocity"], (80, 220, 255))
    command = getattr(env, "last_command", np.zeros(2))
    arrow(paddle["position"], command, (145, 255, 150))
    def text(value, x, y, size=0.62, color=(225, 227, 230)):
        cv2.putText(frame, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)
    text("OPPONENT GOAL", 112, 30, 0.58)
    text("ROBOT GOAL", 128, 842, 0.58)
    text("LEARNED PPO POLICY", 425, 60, 0.85)
    text("Incoming puck / static obstacle avoidance", 425, 95)
    text(f"Training steps: {trained_steps:,}", 425, 139)
    text(f"Outcome: {info.get('outcome', 'running').upper()}", 425, 179, 0.72)
    text(f"Seed {seed}   |   Sim time {steps*env.control_dt:.2f}s", 425, 217)
    speed = float(np.linalg.norm(puck["velocity"]))
    text(f"Puck speed: {speed:.2f} m/s", 425, 253)
    text(f"Puck vx,vy: {puck['velocity'][0]:+.2f}, {puck['velocity'][1]:+.2f}", 425, 287)
    text(f"Policy vx,vy: {command[0]:+.2f}, {command[1]:+.2f} m/s", 425, 321)
    text(f"Paddle touched puck: {'yes' if info.get('hit') else 'no'}", 425, 355)
    text(f"Obstacles: {len(env.user_block_positions)} (trained with {training_obstacles})", 425, 389)
    text(f"Goals {counts['goal']}   Collisions {counts['obstacle_collision']}", 425, 440)
    text(f"Conceded {counts['conceded']}   Timeouts {counts['timeout']}", 425, 474)
    ended = info.get('outcome', 'running') != 'running'
    status = ('Ended: N next / R repeat' if ended and not auto_next else
              'PAUSED' if paused else f'Playback {rate:g}x')
    text(status, 425, 525, 0.70, (160, 230, 160))
    for i, line in enumerate([
        "SPACE pause   N new scenario   R repeat seed",
        "1/2/3 obstacles   +/- speed   A auto-next",
        "Click reachable table region: new puck start",
        "Slider: incoming speed (0 = randomized)",
        "S save frame   Q/ESC close",
        "Orange: puck / yellow: puck velocity",
        "Cyan: paddle / green: policy velocity",
        "No scripted aiming or obstacle-avoidance player.",
        "Experimental policy; outcomes shown honestly.",
    ]):
        text(line, 425, 575 + i*29, 0.50)
    return frame


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--checkpoint-steps", type=int)
    p.add_argument("--seed", type=int, default=20000)
    p.add_argument("--obstacles", type=int, choices=range(4), default=1)
    p.add_argument("--episodes", type=int, default=0, help="0: keep playing until Q")
    p.add_argument("--record", type=Path, help="Optional MJPG AVI of actual simulation frames")
    p.add_argument("--no-window", action="store_true", help="Render video without opening a window")
    p.add_argument("--snapshot", type=Path)
    p.add_argument("--scenario", type=Path, help="Replay a saved evaluation record (seed and blocks)")
    p.add_argument("--auto-next", action="store_true", help="Advance after a 2-second outcome hold")
    args = p.parse_args()
    if args.no_window and args.episodes < 1:
        p.error("--no-window requires a positive --episodes count")
    manifest = json.loads((args.run / "manifest.json").read_text())
    if manifest["task_version"] != TASK_VERSION:
        p.error("Use a v2 incoming-puck checkpoint, not the superseded stationary-puck model")
    model_path, norm_path = args.run / "policy.zip", args.run / "vecnormalize.pkl"
    trained_steps = manifest.get("completed_timesteps", 0)
    if args.checkpoint_steps:
        trained_steps = args.checkpoint_steps
        model_path = args.run / "checkpoints" / f"policy_{trained_steps}_steps.zip"
        norm_path = args.run / "checkpoints" / f"policy_vecnormalize_{trained_steps}_steps.pkl"
    torch.set_num_threads(1)
    env = PrecisionStrikerEnv(seed=args.seed, obstacles=args.obstacles, randomize=False,
                             speed_min=manifest.get("speed_min", 0.3), speed_max=manifest.get("speed_max", 1.5))
    normalizer = VecNormalize.load(str(norm_path), DummyVecEnv([lambda: env]))
    normalizer.training = False
    normalizer.norm_reward = False
    model = PPO.load(str(model_path), device="cpu")
    clicked = []
    if not args.no_window:
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.createTrackbar("Incoming speed x100 (0=random)", WINDOW, 0, 200, lambda value: None)
        def click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                pos = ((y-55)/760*env.length-env.length/2, (x-40)/340*env.width-env.width/2)
                if -0.15 <= pos[0] <= 0.60 and abs(pos[1]) <= 0.25:
                    clicked.append(pos)
        cv2.setMouseCallback(WINDOW, click)
    writer = None
    if args.record:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(args.record), cv2.VideoWriter_fourcc(*"MJPG"), 20, (1000, 860))
        if not writer.isOpened():
            raise RuntimeError("Could not open video writer")
    seed, completed = args.seed, 0
    counts, trail = Counter(), deque(maxlen=100)
    paused, rate, needs_reset, steps = False, 1.0, True, 0
    info, finished_at = {}, None
    options, last_options = {}, {}
    auto_next = args.auto_next or args.no_window
    if args.scenario:
        scenario = json.loads(args.scenario.read_text())
        seed = int(scenario['seed'])
        options['blocks'] = scenario['blocks']
        print(f'Replaying saved scenario seed {seed}; press N for a new randomized layout.', flush=True)
    try:
        while True:
            started = time.perf_counter()
            if needs_reset:
                if not args.no_window:
                    selected_speed = cv2.getTrackbarPos("Incoming speed x100 (0=random)", WINDOW) / 100
                    if selected_speed > 0:
                        options["puck_velocity"] = [selected_speed, 0]
                last_options = dict(options)
                obs, _ = env.reset(seed=seed, options=options)
                env.last_command = np.zeros(2)
                info, steps, finished_at = {"outcome": "running", "hit": False}, 0, None
                trail.clear()
                needs_reset, options = False, {}
            frame = draw_frame(env, info, trail, counts, steps, seed, trained_steps, paused, rate,
                               manifest.get("obstacles", "unknown"), auto_next)
            if args.snapshot and completed == 0 and steps == 0:
                args.snapshot.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(args.snapshot), frame)
                args.snapshot = None
            if writer is not None:
                writer.write(frame)
            key = -1
            if not args.no_window:
                cv2.imshow(WINDOW, frame)
                elapsed = time.perf_counter()-started
                key = cv2.waitKey(max(1, int((env.control_dt/rate-elapsed)*1000))) & 0xFF
                if key in (27, ord('q')) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
            if key == ord(' '):
                paused = not paused
            if key == ord('a'):
                auto_next = not auto_next
            if key in (ord('+'), ord('=')):
                rate = min(8, rate*2)
            if key == ord('-'):
                rate = max(0.25, rate/2)
            if key == ord('s'):
                cv2.imwrite(str(args.run / f"viewer-{int(time.time())}.png"), frame)
            if key in (ord('1'), ord('2'), ord('3')):
                env.obstacle_count = int(chr(key))
                key = ord('n')
            if key in (ord('n'), ord('r')) or clicked:
                if key != ord('r'):
                    seed += 1
                else:
                    options = dict(last_options)
                if clicked:
                    options["puck_position"] = clicked.pop()
                    clicked.clear()
                needs_reset = True
                continue
            if finished_at is not None:
                if args.no_window or (auto_next and not paused and time.perf_counter()-finished_at > 2.0):
                    if args.episodes and completed >= args.episodes:
                        break
                    seed += 1
                    needs_reset = True
                continue
            if paused:
                continue
            action, _ = model.predict(normalizer.normalize_obs(obs[None, :]), deterministic=True)
            obs, _, terminated, truncated, info = env.step(action[0])
            steps += 1
            trail.append(env.current_state["pucks"][0]["position"])
            if terminated or truncated:
                completed += 1
                counts[info["outcome"]] += 1
                finished_at = time.perf_counter()
                print(json.dumps({"seed": seed, "outcome": info["outcome"], "hit": info["hit"],
                                  "counts": dict(counts)}), flush=True)
    finally:
        normalizer.close()
        if writer is not None:
            writer.release()
        if not args.no_window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
