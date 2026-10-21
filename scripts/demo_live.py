"""Live demo: the real Box2D simulation playing shots in a window, in real time.

    python -m scripts.demo_live                 # strike controller (96% on the test shots)
    python -m scripts.demo_live --selftest      # run 6 shots without a window, then exit
    python -m scripts.demo_live --rl-run ... --start-rl --showcase 40
                                                # N cycles through 40 pre-checked shots the RL policy scores

Keys:  N / Space  next shot          R  replay this shot          A  auto-play on/off
       1 2 3      number of obstacles F  fast pucks (1.5-2.0 m/s)  T  puck trail on/off
       O          planning overlay (predicted path, hit point, aim); off by default
       M          switch: strike controller / original RL model    P  pause
       S          slow motion on/off  C  clear tally               Q / Esc  quit
       X          strict rules on/off: one paddle contact, paddle restitution 0.8 (no speed-up),
                  random layouts verified scorable (on by default for models trained that way)
Mouse: left-click the far half to place an obstacle there (up to 3); right-click to go back to random.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from scripts import scoring_geometry as sg
from scripts.analysis.render_episode_gifs import COL, OUTCOME_TEXT, Canvas
from scripts.precision_striker_env import PrecisionStrikerEnv
from scripts.strike_planner import TUNED, StrikePlanner, predict

ROOT = Path(__file__).resolve().parents[1]
WINDOW = "Precision Striker - live demo"
BGR = {k: (v[2], v[1], v[0]) for k, v in COL.items()}


class FastCanvas:
    """OpenCV drawing with the same geometry and colours as the slide GIFs, fast enough for 20 Hz.
    The table, rails and labels are drawn once and cached; moving pieces are drawn per frame."""

    def __init__(self, width=1280, band=58, margin=26):
        self.width, self.band, self.margin = width, band, margin
        self.scale = (width - 2 * margin) / sg.LENGTH
        self.height = int(band + sg.WIDTH * self.scale + 2 * margin)
        static = Canvas(width=width, band=band, margin=margin)
        blank = static.frame((9, 9), (9, 9), [], [], "")          # pieces drawn off-table
        self.background = cv2.cvtColor(np.asarray(blank), cv2.COLOR_RGB2BGR)

    def px(self, x, y):
        return (int(round((self.margin + (x + sg.LENGTH / 2) * self.scale) * 16)),
                int(round((self.band + self.margin + (sg.WIDTH / 2 - y) * self.scale) * 16)))

    def circle(self, im, x, y, r, colour, thickness=-1):
        cv2.circle(im, self.px(x, y), int(r * self.scale * 16), colour, thickness, cv2.LINE_AA, shift=4)

    def frame(self, puck, paddle, blocks, trail, title, status="", hit=False, banner=None,
              pred=None, aim=None, intercept=None):
        im = self.background.copy()
        for bx, by in blocks:
            a, b = self.px(bx - sg.BLOCK_W / 2, by + sg.BLOCK_W / 2), self.px(bx + sg.BLOCK_W / 2, by - sg.BLOCK_W / 2)
            cv2.rectangle(im, (a[0] >> 4, a[1] >> 4), (b[0] >> 4, b[1] >> 4), BGR["obstacle"], -1)
        if pred is not None:
            for q in pred[::4]:
                self.circle(im, q[0], q[1], 0.006, BGR["pred"])
        if aim is not None:
            pts = [np.array(self.px(*q), float) / 16 for q in aim]
            for a, b in zip(pts, pts[1:]):
                n = max(int(np.linalg.norm(b - a) / 14), 1)
                for i in range(0, n, 2):
                    p0, p1 = a + (b - a) * i / n, a + (b - a) * min(i + 1, n) / n
                    cv2.line(im, tuple(int(v) for v in p0), tuple(int(v) for v in p1), BGR["aim"], 3, cv2.LINE_AA)
        if intercept is not None:
            self.circle(im, intercept[0], intercept[1], sg.PUCK_R + 0.012, BGR["pred"], 3)
        for i, q in enumerate(trail):
            shade = int(235 - 75 * (i + 1) / max(1, len(trail)))
            self.circle(im, q[0], q[1], 0.008, (shade + 6, shade, shade))
        self.circle(im, paddle[0], paddle[1], 0.0508, BGR["paddle"])
        self.circle(im, puck[0], puck[1], sg.PUCK_R, BGR["puck"])
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(im, title, (self.margin, 36), font, 0.8, BGR["ink"], 2, cv2.LINE_AA)
        right = status + ("   paddle hit" if hit else "")
        (tw, _), _ = cv2.getTextSize(right, font, 0.6, 1)
        cv2.putText(im, right, (self.width - self.margin - tw, 36), font, 0.6, BGR["muted"], 1, cv2.LINE_AA)
        if banner:
            text, good = banner
            (tw, th), _ = cv2.getTextSize(text, font, 1.2, 3)
            cx, top = self.width // 2, self.band + self.margin + 16
            cv2.rectangle(im, (cx - tw // 2 - 18, top), (cx + tw // 2 + 18, top + th + 26),
                          BGR["good"] if good else BGR["bad"], -1)
            cv2.putText(im, text, (cx - tw // 2, top + th + 12), font, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
        return im
HELP = "N next   R replay   1/2/3 obstacles   F fast   T trail   M switch   X strict rules   A auto   S slow   P pause   Q quit   click: place obstacle"


class Demo:
    def __init__(self, width=1280, rl_run=None, seed=0, rl_label=None, start_rl=False):
        self.canvas = FastCanvas(width=width, band=58, margin=26)
        self.bar_font = cv2.FONT_HERSHEY_SIMPLEX
        self.rng = np.random.default_rng(seed)
        self.obstacles, self.fast, self.overlay, self.mode = 1, False, False, "controller"
        self.show_trail = True
        self.auto, self.paused, self.slow = False, False, False
        self.custom_blocks: list[tuple[float, float]] = []
        self.tally = {"controller": [0, 0], "rl": [0, 0]}
        self.planner = StrikePlanner(**TUNED)
        self.rl_run = Path(rl_run) if rl_run else ROOT / "models" / "precision-striker-v2"
        # Build the simulator with the observation options the RL model was trained on (e.g. the
        # imitation + PPO network also sees "already hit" and "time used"). The controller ignores them.
        manifest = json.loads((self.rl_run / "manifest.json").read_text()) if (self.rl_run / "manifest.json").exists() else {}
        opts = manifest.get("task_options", {})
        self.rl_obs = {k: bool(opts.get(k, False)) for k in ("extra_obs", "goal_shaping")}
        # Random demo layouts use the model's training region, so N shows the setups it trained on.
        if opts.get("wide_obstacles"):
            self.rl_obs["wide_obstacles"] = True
        self.trained_wide = bool(opts.get("wide_obstacles"))
        self.trained_max = int(opts.get("obstacles_max") or 1)
        # Strict rules follow the model's training task; X toggles them.
        self.strict = bool(opts.get("single_hit"))
        # Imitation-based networks use the extra observations; the original September model does not.
        learned_from_controller = bool(manifest.get("bc") or manifest.get("dagger") or self.rl_obs["extra_obs"])
        via_dagger = bool(manifest.get("dagger")) or "dagger" in str(manifest.get("resume", ""))
        self.rl_label = rl_label or ("RL policy (DAgger + PPO)" if via_dagger else
                                     "RL policy (imitation + PPO)" if learned_from_controller else "Original RL model")
        if start_rl:
            self.mode = "rl"
        self.showcase, self.showcase_i = [], 0     # pre-checked (obstacle count, seed) shots, see find_showcase
        self._rl = None
        self.env = None
        self.message, self.message_until = "", 0.0
        try:
            self._rl_policy()          # load the RL model now so pressing M mid-demo does not stall
        except Exception as exc:       # the controller demo still works without it
            self.message, self.message_until = f"RL model not loaded: {exc}", time.time() + 6
        self.new_shot()

    # ------------------------------------------------------------------ setup
    def _make_env(self):
        speed = dict(speed_min=1.5, speed_max=2.0) if self.fast else {}
        if self.env is not None:
            self.env.close()
        # Strict random layouts match that training task: half of them block the straight shot.
        rules = (dict(single_hit=True, paddle_restitution=0.8, layout_check="trajectory",
                      obstacles_max=None, block_line_prob=0.5) if self.strict else {})
        self.planner.restitution = 0.8 if self.strict else 1.0
        self.env = PrecisionStrikerEnv(seed=0, obstacles=self.obstacles, randomize=False, require_hit=True,
                                       timeout_penalty=10, obstacle_bounce=True, **speed, **self.rl_obs, **rules)

    def _rl_policy(self):
        if self._rl is None:
            import torch
            from stable_baselines3 import PPO
            from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
            torch.set_num_threads(1)
            probe = PrecisionStrikerEnv(obstacles=1, require_hit=True, obstacle_bounce=True, **self.rl_obs)
            norm = VecNormalize.load(str(self.rl_run / "vecnormalize.pkl"), DummyVecEnv([lambda: probe]))
            norm.training = False
            model = PPO.load(str(self.rl_run / "policy.zip"), device="cpu")
            self._rl = lambda o: model.predict(norm.normalize_obs(o[None, :]), deterministic=True)[0][0]
        return self._rl

    def new_shot(self, replay=False):
        if not replay and self.showcase and not self.custom_blocks:
            self.obstacles, self.seed = self.showcase[self.showcase_i % len(self.showcase)]
            self.showcase_i += 1
        elif not replay:
            self.seed = int(self.rng.integers(0, 2**31 - 1))
        self._make_env()
        options = {"blocks": self.custom_blocks} if self.custom_blocks else None
        self.obs, info = self.env.reset(seed=self.seed, options=options)
        self.planner.reset()
        self.trail, self.t, self.done, self.info, self.done_at = [], 0, False, {"hit": False}, None
        if not info.get("scorable", True):
            self.flash("This obstacle layout has no clear path to the goal")

    def find_showcase(self, count, obstacle_counts=(1, 1, 2), tries=2000):
        """Search random shots headlessly and keep those the RL policy scores (the sim is deterministic
        for a given seed, so each one replays exactly). Clicked obstacles still show unselected shots."""
        policy, found = self._rl_policy(), []
        for i in range(tries):
            if len(found) >= count:
                break
            self.obstacles = obstacle_counts[i % len(obstacle_counts)]
            seed = int(self.rng.integers(0, 2**31 - 1))
            self._make_env()
            obs, _ = self.env.reset(seed=seed)
            done = False
            while not done:
                obs, _, terminated, truncated, info = self.env.step(policy(obs))
                done = terminated or truncated
            if info["outcome"] == "goal" and info["hit"]:
                found.append((self.obstacles, seed))
        print(f"showcase: {len(found)} scoring shots found in {i + 1} tries")
        self.showcase, self.showcase_i = found, 0
        return found

    def flash(self, text, seconds=2.5):
        self.message, self.message_until = text, time.time() + seconds

    # ------------------------------------------------------------------- step
    def step(self):
        if self.done or self.paused:
            return
        actor = self.planner.act if self.mode == "controller" else self._rl_policy()
        self.obs, _, terminated, truncated, self.info = self.env.step(actor(self.obs))
        self.t += 1
        self.trail.append(self.env.current_state["pucks"][0]["position"])
        if terminated or truncated:
            self.done, self.done_at = True, time.time()
            tally = self.tally[self.mode]
            tally[1] += 1
            tally[0] += int(self.info["outcome"] == "goal" and self.info["hit"])

    # ------------------------------------------------------------------- draw
    def frame(self):
        st = self.env.current_state
        pred = aim = intercept = None
        p = self.planner.plan
        if self.mode == "controller" and self.overlay and p is not None and self.planner.struck_at is None and not self.done:
            _, pred, _ = predict(self.obs[4:6], self.obs[6:8], self.planner.damping,
                                 horizon=max(p["T"] - self.planner.t, 0.0) + 0.02)
            intercept = p["S"]
            aim = [tuple(p["S"]), tuple(p["shot"]["aim"])]
            if p["shot"]["kind"] == "bank":
                aim.append((sg.GOAL_X, p["shot"].get("goal_y", 0.0)))
        who = "Strike controller" if self.mode == "controller" else self.rl_label
        if self.mode == "rl" and self.outside_training():
            who += "  [untrained setup]"
        title = f"{who}   |   {self.obstacles} obstacle{'s' * (self.obstacles > 1)}" + \
                ("   |   custom layout" if self.custom_blocks else "") + ("   |   fast pucks" if self.fast else "")
        speed = float(np.linalg.norm(st["pucks"][0]["velocity"]))
        status = f"t = {self.t * 0.05:.2f} s   puck {speed:.2f} m/s"
        banner = OUTCOME_TEXT.get(self.info["outcome"]) if self.done else None
        im = self.canvas.frame(st["pucks"][0]["position"], st["paddles"]["paddle_ego"]["position"],
                               list(self.env.user_block_positions), self.trail[-30:] if self.show_trail else [], title, status,
                               self.info.get("hit", False), banner, pred, aim, intercept)
        bar = np.zeros((74, im.shape[1], 3), np.uint8)
        bar[:] = (65, 42, 27)
        f = self.bar_font
        goals, shots = self.tally[self.mode]
        rate = f"{100 * goals / shots:.0f}%" if shots else "-"
        cv2.putText(bar, f"Scored {goals} / {shots}  ({rate})", (26, 32), f, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
        flags = [("SELECTED SHOTS", bool(self.showcase) and not self.custom_blocks),
                 ("ONE-CONTACT RULES", self.strict), ("AUTO", self.auto), ("SLOW", self.slow),
                 ("PAUSED", self.paused), ("OVERLAY", self.overlay)]
        cv2.putText(bar, "   ".join(n for n, on in flags if on), (360, 31), f, 0.6, (90, 200, 250), 1, cv2.LINE_AA)
        cv2.putText(bar, HELP, (26, 62), f, 0.45, (215, 190, 170), 1, cv2.LINE_AA)
        if self.message and time.time() < self.message_until:
            (tw, _), _ = cv2.getTextSize(self.message, f, 0.6, 1)
            cv2.putText(bar, self.message, (im.shape[1] - 26 - tw, 31), f, 0.6, (120, 140, 255), 1, cv2.LINE_AA)
        return np.vstack([im, bar])

    def outside_training(self):
        """True when the layout is outside what the loaded RL network was trained on."""
        blocks = list(self.env.user_block_positions)
        if len(blocks) > self.trained_max:
            return True
        if self.trained_wide:
            return False                       # trained on the whole clickable area
        return any(not (-0.65 <= x <= -0.25 and abs(y) <= 0.28) for x, y in blocks)

    # ----------------------------------------------------------------- inputs
    def click(self, event, u, v, flags, param):
        c = self.canvas
        x = (u - c.margin) / c.scale - sg.LENGTH / 2
        y = sg.WIDTH / 2 - (v - c.band - c.margin) / c.scale
        if event == cv2.EVENT_LBUTTONDOWN:
            if not (-0.8 <= x <= -0.22 and abs(y) <= 0.32):
                self.flash("Place obstacles on the far half, in front of the goal")
                return
            if any(max(abs(x - bx), abs(y - by)) < sg.BLOCK_W + 0.02 for bx, by in self.custom_blocks):
                self.flash("Too close to another obstacle")
                return
            self.custom_blocks = (self.custom_blocks + [(x, y)])[-3:]
            self.obstacles = len(self.custom_blocks)
            self.new_shot(replay=True)
            if self.mode == "rl" and self.outside_training():
                self.flash("The RL policy never trained on this setup; expect lower accuracy", 4)
        elif event == cv2.EVENT_RBUTTONDOWN and self.custom_blocks:
            self.custom_blocks = []
            self.flash("Back to random obstacles")
            self.new_shot()

    def key(self, k):
        if k in (ord("q"), 27):
            return False
        if k in (ord("n"), ord(" ")):
            self.new_shot()
        elif k == ord("r"):
            self.new_shot(replay=True)
        elif k in (ord("1"), ord("2"), ord("3")):
            self.obstacles, self.custom_blocks = k - ord("0"), []
            self.new_shot()
            if self.mode == "rl" and self.obstacles > self.trained_max:
                self.flash(f"The RL policy trained with up to {self.trained_max} obstacle(s); expect lower accuracy", 4)
        elif k == ord("f"):
            self.fast = not self.fast
            self.new_shot()
        elif k == ord("o"):           # planning overlay (off by default, not listed in the help line)
            self.overlay = not self.overlay
        elif k == ord("t"):
            self.show_trail = not self.show_trail
        elif k == ord("x"):
            self.strict = not self.strict
            self.flash("Strict rules: one contact, no speed-up" if self.strict else "Original rules")
            self.new_shot(replay=True)
        elif k == ord("m"):
            self.mode = "rl" if self.mode == "controller" else "controller"
            self.new_shot(replay=True)
        elif k == ord("a"):
            self.auto = not self.auto
        elif k == ord("p"):
            self.paused = not self.paused
        elif k == ord("s"):
            self.slow = not self.slow
        elif k == ord("c"):
            self.tally = {"controller": [0, 0], "rl": [0, 0]}
        return True


def run_window(demo):
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    first = demo.frame()
    cv2.resizeWindow(WINDOW, first.shape[1], first.shape[0])
    cv2.setMouseCallback(WINDOW, demo.click)
    while True:
        started = time.perf_counter()
        demo.step()
        cv2.imshow(WINDOW, demo.frame())
        if demo.done and demo.auto and time.time() - demo.done_at > 1.8:
            demo.new_shot()
        period = 0.05 * (3.0 if demo.slow else 1.0)            # 20 Hz control = real time
        wait = max(1, int((period - (time.perf_counter() - started)) * 1000))
        k = cv2.waitKey(wait) & 0xFF
        if k != 255 and not demo.key(k):
            break
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()


def selftest(demo, shots=6):
    for mode in ("controller", "rl"):
        demo.mode = mode
        for i in range(shots):
            demo.new_shot()
            t0, frames = time.perf_counter(), 0
            while not demo.done:
                demo.step(); demo.frame(); frames += 1
            ms = 1000 * (time.perf_counter() - t0) / max(frames, 1)
            print(f"{mode:10s} shot {i + 1}: {demo.info['outcome']:9s} hit={demo.info['hit']}  "
                  f"{ms:.0f} ms per frame (real time needs < 50)")
    print("tally:", demo.tally)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--width", type=int, default=1280, help="Window width in pixels")
    p.add_argument("--rl-run", type=Path, help="Trained PPO run for M (default: models/precision-striker-v2)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rl-label", help="Name shown for the RL model (default: from its manifest)")
    p.add_argument("--start-rl", action="store_true", help="Start in RL mode instead of the strike controller")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--showcase", type=int, default=0,
                   help="Pre-select this many shots (1-2 random obstacles) that the RL policy scores")
    args = p.parse_args()
    demo = Demo(args.width, args.rl_run, args.seed, args.rl_label, args.start_rl)
    if args.showcase:
        demo.find_showcase(args.showcase)
        demo.new_shot()
    selftest(demo) if args.selftest else run_window(demo)


if __name__ == "__main__":
    main()
