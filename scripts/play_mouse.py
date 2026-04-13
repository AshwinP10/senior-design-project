"""
Basic air hockey play with mouse control.
Move the mouse over the window to move the paddle. Press R to reset, Q or ESC to quit.

Run from repo root:  python scripts/play_mouse.py
Or with config:      python scripts/play_mouse.py --cfg configs/offline_configs/demonstrate.yaml

Minimal deps (if not already installed):  pip install Box2D opencv-python pyyaml gymnasium
"""
import sys
import os

# Allow running without installing the package (from repo root)
_repo_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import cv2
import numpy as np
import time
import argparse
import yaml

from airhockey import AirHockeyEnv
from airhockey.renderers import AirHockeyRenderer

WINDOW_NAME = "Air Hockey — Mouse to play"
# Shared state for mouse callback (thread-safe enough for demo)
_mouse_xy = [None, None]
_mouse_inside = [False]


def _on_mouse(event, x, y, _unused_a, _unused_b):
    if event in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN):
        _mouse_xy[0], _mouse_xy[1] = x, y
        _mouse_inside[0] = True
    elif event == cv2.EVENT_MOUSELEAVE:
        _mouse_inside[0] = False


def pixel_to_base(mx, my, renderer):
    """Convert window pixel (mx, my) to base coordinates (x, y).
    Screen: right = positive mx, down = positive my. Base: x along table (top negative, bottom positive), y across (left negative, right positive).
    We want: mouse right -> paddle right (base_y increase), mouse down -> paddle toward bottom (base_x increase).
    """
    ppm = renderer.ppm
    length = renderer.length
    width = renderer.width
    render_length = renderer.render_length

    if renderer.orientation == "vertical":
        # Displayed frame is rotated 90 CCW. Align so mouse right = base_y+, mouse down = base_x+.
        pix_row = mx
        pix_col = render_length - 1 - my
    else:
        pix_row = my
        pix_col = mx

    # Draw: pos (render) = (y, -x), pixel = (center[1], center[0])*ppm with center = pos + (width/2, length/2)
    # So pix_col = (-x + length/2)*ppm, pix_row = (y + width/2)*ppm => base_x = length/2 - pix_col/ppm, base_y = pix_row/ppm - width/2
    base_x = length / 2 - pix_col / ppm
    base_y = pix_row / ppm - width / 2
    return np.array([base_x, base_y], dtype=float)


def main():
    parser = argparse.ArgumentParser(description="Play air hockey with mouse.")
    parser.add_argument(
        "--cfg",
        type=str,
        default=None,
        help="Path to config YAML (default: configs/offline_configs/demonstrate.yaml)",
    )
    args = parser.parse_args()

    if args.cfg is None:
        default_path = os.path.join(
            os.path.dirname(__file__), "..", "configs", "offline_configs", "demonstrate.yaml"
        )
        cfg_path = os.path.normpath(default_path)
    else:
        cfg_path = args.cfg

    with open(cfg_path, "r") as f:
        air_hockey_cfg = yaml.safe_load(f)

    air_hockey_params = air_hockey_cfg["air_hockey"].copy()
    air_hockey_params["n_training_steps"] = air_hockey_cfg.get("n_training_steps", 500000)
    air_hockey_params["seed"] = air_hockey_cfg.get("seed", 0)
    if "goal" in air_hockey_params.get("task", ""):
        air_hockey_params["return_goal_obs"] = True
    else:
        air_hockey_params["return_goal_obs"] = False
    # Play mode: end episode only when puck hits a goal (not on puck stop)
    air_hockey_params["terminate_on_puck_stop"] = False

    env = AirHockeyEnv(air_hockey_params)
    renderer = AirHockeyRenderer(env, orientation="vertical", robosuite_view="")
    renderer.show_target_position = False

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, _on_mouse)

    # Action = delta toward mouse; scale so paddle is responsive but stable
    action_scale = 2.0
    max_delta = 0.5

    obs, _ = env.reset()
    start = time.time()
    step_count = 0

    while True:
        # Check if user closed the window (X button)
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

        frame = renderer.get_frame()
        cv2.putText(frame, "Q or ESC to quit | R = reset", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "Q or ESC to quit | R = reset", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(20)

        if key in (ord("q"), ord("Q"), 27):
            break
        if key in (ord("r"), ord("R")):
            obs, _ = env.reset()
            continue

        state = env.current_state
        paddle_pos = np.array(state["paddles"]["paddle_ego"]["position"], dtype=float)

        if _mouse_inside[0] and _mouse_xy[0] is not None and _mouse_xy[1] is not None:
            mouse_base = pixel_to_base(_mouse_xy[0], _mouse_xy[1], renderer)
            mouse_base[0] = np.clip(mouse_base[0], -env.length / 2, env.length / 2)
            mouse_base[1] = np.clip(mouse_base[1], -env.width / 2, env.width / 2)
            delta = mouse_base - paddle_pos
            delta = np.clip(delta, -max_delta, max_delta) * action_scale
            action = delta
        else:
            action = np.array([0.0, 0.0])

        obs, rew, terminated, truncated, info = env.step(action)
        step_count += 1
        if step_count % 500 == 0:
            print(f"fps (approx): {500 / (time.time() - start):.1f}")
            start = time.time()
        if terminated or truncated:
            if info.get("puck_within_ego_goal"):
                print("Goal! You scored. Resetting...")
            elif info.get("puck_within_alt_goal"):
                print("Goal! Scored on you. Resetting...")
            else:
                print("Episode over. Resetting...")
            obs, _ = env.reset()

    cv2.destroyAllWindows()
    try:
        cv2.waitKey(1)
    except Exception:
        pass
    sys.exit(0)


if __name__ == "__main__":
    main()
