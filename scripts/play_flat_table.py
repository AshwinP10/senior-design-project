"""
Flat table: place static cubes with left-click, move paddle with mouse.
Puck spawns at center with zero velocity (no drift). Cubes are immovable obstacles.
R = reset puck (and paddle), keep cubes. C = clear all cubes and reset. Q/ESC = quit.

Run:  python scripts/play_flat_table.py
Or:   python scripts/play_flat_table.py --cfg configs/flat_table_play.yaml
"""
import sys
import os

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

WINDOW_NAME = "Flat table — Left-click: cube | R: reset | C: clear cubes"
_mouse_xy = [None, None]
_mouse_inside = [False]
_click_place = [False]  # True on LBUTTONDOWN so we add one block per click


def _on_mouse(event, x, y, _unused_a, _unused_b):
    global _click_place
    if event in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN):
        _mouse_xy[0], _mouse_xy[1] = x, y
        _mouse_inside[0] = True
        if event == cv2.EVENT_LBUTTONDOWN:
            _click_place[0] = True
    elif event == cv2.EVENT_MOUSELEAVE:
        _mouse_inside[0] = False


def pixel_to_base(mx, my, renderer):
    ppm = renderer.ppm
    length = renderer.length
    width = renderer.width
    render_length = renderer.render_length
    if renderer.orientation == "vertical":
        pix_row = mx
        pix_col = render_length - 1 - my
    else:
        pix_row = my
        pix_col = mx
    base_x = length / 2 - pix_col / ppm
    base_y = pix_row / ppm - width / 2
    return np.array([base_x, base_y], dtype=float)


def main():
    print("Loading config and env...", flush=True)
    parser = argparse.ArgumentParser(description="Flat table: place cubes, play with paddle.")
    parser.add_argument("--cfg", type=str, default=None, help="Config YAML (default: configs/flat_table_play.yaml)")
    args = parser.parse_args()

    cfg_path = args.cfg or os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "configs", "flat_table_play.yaml"))
    with open(cfg_path, "r") as f:
        air_hockey_cfg = yaml.safe_load(f)

    air_hockey_params = air_hockey_cfg["air_hockey"].copy()
    air_hockey_params["n_training_steps"] = air_hockey_cfg.get("n_training_steps", 1)
    air_hockey_params["seed"] = air_hockey_cfg.get("seed", 0)
    air_hockey_params["return_goal_obs"] = False

    print("Creating environment...", flush=True)
    env = AirHockeyEnv(air_hockey_params)
    if not hasattr(env, "user_block_positions"):
        env.user_block_positions = []
    renderer = AirHockeyRenderer(env, orientation="vertical", robosuite_view="")
    renderer.show_target_position = False

    print("Opening window...", flush=True)
    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, _on_mouse)

    # Higher scale = paddle moves faster so you can dodge obstacles
    action_scale = 4.0
    max_delta = 1.0

    obs, _ = env.reset()
    start = time.time()
    step_count = 0
    score = 0

    # Draw and show first frame so the window appears (required on some Windows setups)
    frame = renderer.get_frame()
    cv2.imshow(WINDOW_NAME, frame)
    cv2.waitKey(1)

    print("Flat table sim running. Move mouse = paddle, left-click = place cube, R = reset, C = clear cubes, Q = quit.", flush=True)
    while True:
        # Only exit if user closed the window (0 = closed). -1 can mean "not yet visible" on Windows.
        try:
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) == 0:
                break
        except cv2.error:
            break

        # Left-click: place one static cube at mouse position
        if _click_place[0]:
            _click_place[0] = False
            if _mouse_xy[0] is not None and _mouse_xy[1] is not None:
                base = pixel_to_base(_mouse_xy[0], _mouse_xy[1], renderer)
                base[0] = np.clip(base[0], -env.length / 2 + env.block_width, env.length / 2 - env.block_width)
                base[1] = np.clip(base[1], -env.width / 2 + env.block_width, env.width / 2 - env.block_width)
                env.user_block_positions.append(tuple(base))
                name = "user_block_{}".format(len(env.user_block_positions) - 1)
                env.simulator.spawn_block(base, (0.0, 0.0), name, affected_by_gravity=False, movable=False)
                env.simulator.set_object_links()
                # Refresh state so the cube appears on the next frame (renderer uses env.current_state)
                env.current_state = env.simulator.get_current_state()

        frame = renderer.get_frame()
        cv2.putText(frame, "Left-click: place cube | R: reset puck | C: clear cubes | Q/ESC: quit", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.putText(frame, "Left-click: place cube | R: reset puck | C: clear cubes | Q/ESC: quit", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Scoreboard box
        score_text = f"Goals: {score}"
        (tw, th), _ = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        box_x, box_y = frame.shape[1] // 2 - tw // 2 - 8, 6
        cv2.rectangle(frame, (box_x, box_y), (box_x + tw + 16, box_y + th + 12), (30, 30, 30), -1)
        cv2.rectangle(frame, (box_x, box_y), (box_x + tw + 16, box_y + th + 12), (200, 200, 200), 1)
        cv2.putText(frame, score_text, (box_x + 8, box_y + th + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(20)

        if key in (ord("q"), ord("Q"), 27):
            break
        if key in (ord("r"), ord("R")):
            obs, _ = env.reset()
            continue
        if key in (ord("c"), ord("C")):
            env.user_block_positions.clear()
            obs, _ = env.reset()
            print("Cleared all cubes.")
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
            print("fps (approx): {:.1f}".format(500 / (time.time() - start)))
            start = time.time()
        if terminated or truncated:
            if info.get("puck_within_ego_goal"):
                score += 1
                print(f"Goal! Score: {score}")
            obs, _ = env.reset()

    cv2.destroyAllWindows()
    try:
        cv2.waitKey(1)
    except Exception:
        pass
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error running sim:", e)
        import traceback
        traceback.print_exc()
        sys.exit(1)
